"""Training-loop helpers that rewire every ION layer in a model in one call."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .layers import IONLinear, RewireResult


def ion_layers(model: nn.Module) -> List[tuple]:
    """Return ``(name, module)`` for every :class:`IONLinear` in ``model``."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, IONLinear)]


@torch.no_grad()
def rewire(
    model: nn.Module,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[RewireResult]:
    """Rewire all ION layers in ``model`` if ``step`` is a rewire step for them.

    Call this once per training step, after ``optimizer.step()``. Layers
    whose ``rewire_interval`` does not divide ``step`` are skipped, so layers
    with different schedules can share one call.

    If ``optimizer`` is given, its state (e.g. Adam moments) is zeroed for
    the connections that changed. Without this, a newly grown connection
    would inherit momentum from the connection it replaced.

    Returns the list of :class:`RewireResult` for layers that rewired.
    """
    results: List[RewireResult] = []
    for name, layer in ion_layers(model):
        res = layer.rewire(step, name=name)
        if res is None:
            continue
        results.append(res)
        if optimizer is not None:
            _reset_optimizer_state(optimizer, layer.weight, res.changed)
    return results


def _reset_optimizer_state(
    optimizer: torch.optim.Optimizer, param: nn.Parameter, changed: torch.Tensor
) -> None:
    state = optimizer.state.get(param, {})
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.shape == changed.shape:
            value.masked_fill_(changed.to(value.device), 0.0)


def rewire_stats(model: nn.Module) -> dict:
    """Current active-connection density of each ION layer, keyed by module name."""
    return {name: layer.density() for name, layer in ion_layers(model)}


def ion_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Parameters belonging to ION layers (handy for separate learning rates)."""
    for _, layer in ion_layers(model):
        yield from layer.parameters(recurse=False)
