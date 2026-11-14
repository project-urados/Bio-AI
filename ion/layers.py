"""ION layers: sparse linear layers whose connections rewire during training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IONConfig


@dataclass
class RewireResult:
    """What happened in one rewiring step of one layer."""

    layer: str
    step: int
    fraction: float
    pruned: int
    grown: int
    # Boolean (out, in) mask of every connection whose weight was reset.
    changed: torch.Tensor


class IONLinear(nn.Module):
    """A linear layer with a sparse, self-rewiring connection mask.

    The layer keeps a dense weight matrix plus a boolean ``mask`` of active
    connections. Only active connections contribute to the output. Every
    ``config.rewire_interval`` steps (see :meth:`rewire`), a fraction of the
    active connections is pruned (weakest ``|w|`` first) and the same number
    of new ones is grown elsewhere (largest accumulated gradient, or random).

    Args:
        in_features: Number of inputs.
        out_features: Number of neurons.
        config: An :class:`IONConfig`. Defaults to ``IONConfig()``.
        **overrides: Any :class:`IONConfig` field, applied on top of ``config``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[IONConfig] = None,
        **overrides,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        cfg = config if config is not None else IONConfig()
        if overrides:
            cfg = cfg.replace(**overrides)
        self.in_features = in_features
        self.out_features = out_features
        self.config = cfg

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if cfg.bias else None
        self.register_buffer("mask", torch.zeros(out_features, in_features, dtype=torch.bool))
        # Rewiring bookkeeping. Gradient accumulation is transient, so it is
        # not saved in state_dict.
        self.register_buffer("rewire_events", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "_grad_sum", torch.zeros(out_features, in_features), persistent=False
        )
        self.register_buffer("_grad_count", torch.zeros((), dtype=torch.long), persistent=False)

        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        """Re-draw the initial wiring and weights from ``config.seed``."""
        gen = None
        if self.config.seed is not None:
            gen = torch.Generator().manual_seed(self.config.seed)
        self.mask.copy_(self._initial_mask(gen))

        bound = 1.0 / math.sqrt(max(1.0, self.config.density * self.in_features))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound, generator=gen)
            self.weight.mul_(self.mask)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound, generator=gen)
        self._grad_sum.zero_()
        self._grad_count.zero_()

    def _initial_mask(self, gen: Optional[torch.Generator]) -> torch.Tensor:
        out_f, in_f = self.out_features, self.in_features
        scores = torch.rand(out_f, in_f, generator=gen)
        if self.config.budget == "per_neuron":
            k = self._per_neuron_k()
            idx = scores.topk(k, dim=1).indices
            mask = torch.zeros(out_f, in_f, dtype=torch.bool)
            mask.scatter_(1, idx, True)
        else:
            n = self._global_n()
            idx = scores.flatten().topk(n).indices
            mask = torch.zeros(out_f * in_f, dtype=torch.bool)
            mask[idx] = True
            mask = mask.view(out_f, in_f)
        return mask

    def _per_neuron_k(self) -> int:
        return min(self.in_features, max(1, round(self.config.density * self.in_features)))

    def _global_n(self) -> int:
        total = self.out_features * self.in_features
        return min(total, max(1, round(self.config.density * total)))

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        y = F.linear(x, w, self.bias)
        if self.training and self.config.rewire_interval > 0 and y.requires_grad:
            # Collect dense gradients (including on inactive connections) so the
            # grow step can pick the connections that would help most.
            xd = x.detach()
            y.register_hook(lambda g: self._accumulate_grad(g, xd))
        return y

    def _accumulate_grad(self, grad_out: torch.Tensor, x: torch.Tensor) -> None:
        g = grad_out.reshape(-1, self.out_features).float()
        xx = x.reshape(-1, self.in_features).to(g.dtype)
        self._grad_sum += g.t() @ xx
        self._grad_count += 1

    # ------------------------------------------------------------- rewiring
    def fraction_at(self, step: int) -> float:
        """Rewire fraction in effect at ``step`` (after applying the decay schedule)."""
        cfg = self.config
        base = cfg.rewire_fraction
        if cfg.rewire_decay == "cosine" and cfg.rewire_stop:
            progress = min(1.0, step / cfg.rewire_stop)
            return base * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base

    def should_rewire(self, step: int) -> bool:
        cfg = self.config
        if cfg.rewire_interval <= 0 or step <= 0 or step % cfg.rewire_interval != 0:
            return False
        return cfg.rewire_stop is None or step < cfg.rewire_stop

    @torch.no_grad()
    def rewire(self, step: int, name: str = "") -> Optional[RewireResult]:
        """Prune weak connections and grow new ones, if ``step`` is a rewire step.

        Returns a :class:`RewireResult`, or ``None`` if nothing was due.
        The per-neuron fan-in (or the global connection count) is preserved.
        """
        if not self.should_rewire(step):
            return None

        frac = self.fraction_at(step)
        mask = self.mask
        w_abs = self.weight.abs()
        grad_ok = self._grad_count.item() > 0 and self.config.grow == "gradient"
        grow_score = self._grad_sum.abs() if grad_ok else torch.rand_like(w_abs)

        if self.config.budget == "per_neuron":
            k = int(mask[0].sum().item())
            # Can't grow more connections than a neuron has inactive inputs.
            p = min(k, self.in_features - k, round(frac * k))
            prune, grow = self._select_per_neuron(mask, w_abs, grow_score, p)
        else:
            n_active = int(mask.sum().item())
            total = mask.numel()
            p = min(n_active, total - n_active, round(frac * n_active))
            prune, grow = self._select_global(mask, w_abs, grow_score, p)

        new_mask = (mask & ~prune) | grow
        changed = prune | grow
        self.mask.copy_(new_mask)
        self.weight.masked_fill_(~new_mask, 0.0)  # pruned and new connections start at 0
        self._grad_sum.zero_()
        self._grad_count.zero_()
        self.rewire_events += 1

        return RewireResult(
            layer=name,
            step=step,
            fraction=frac,
            pruned=int(prune.sum().item()),
            grown=int(grow.sum().item()),
            changed=changed,
        )

    @staticmethod
    def _select_per_neuron(mask, w_abs, grow_score, p):
        if p == 0:
            empty = torch.zeros_like(mask)
            return empty, empty.clone()
        # Prune the p weakest active connections of each neuron.
        prune_score = w_abs.masked_fill(~mask, float("inf"))
        prune_idx = prune_score.topk(p, dim=1, largest=False).indices
        prune = torch.zeros_like(mask).scatter_(1, prune_idx, True)
        # Grow among inactive connections, excluding the ones just pruned.
        candidates = ~mask & ~prune
        grow_idx = grow_score.masked_fill(~candidates, -float("inf")).topk(p, dim=1).indices
        grow = torch.zeros_like(mask).scatter_(1, grow_idx, True)
        return prune & mask, grow

    @staticmethod
    def _select_global(mask, w_abs, grow_score, p):
        if p == 0:
            empty = torch.zeros_like(mask)
            return empty, empty.clone()
        prune_score = w_abs.masked_fill(~mask, float("inf")).flatten()
        prune = torch.zeros(mask.numel(), dtype=torch.bool, device=mask.device)
        prune[prune_score.topk(p, largest=False).indices] = True
        prune = prune.view_as(mask)
        candidates = ~mask & ~prune
        grow_score = grow_score.masked_fill(~candidates, -float("inf")).flatten()
        grow = torch.zeros(mask.numel(), dtype=torch.bool, device=mask.device)
        grow[grow_score.topk(p).indices] = True
        return prune & mask, grow.view_as(mask)

    # ------------------------------------------------------------- inspection
    @torch.no_grad()
    def density(self) -> float:
        """Fraction of connections that are currently active."""
        return self.mask.float().mean().item()

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, density={c.density}, budget={c.budget}, "
            f"rewire_interval={c.rewire_interval}, rewire_fraction={c.rewire_fraction}"
        )
