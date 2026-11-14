"""Configuration for ION neurons.

An :class:`IONConfig` describes how a layer's connections are wired and how
they rewire during training. Instances can be created in code, loaded from a
dict (e.g. parsed JSON/YAML), and overridden per layer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional

BUDGETS = ("per_neuron", "global")
DECAYS = ("cosine", "constant")
GROW_CRITERIA = ("gradient", "random")


@dataclass
class IONConfig:
    """Wiring and plasticity settings for one :class:`~ion.IONLinear` layer.

    Connectivity
        density: Fraction of all possible input->output connections that are
            active at any time (``0 < density <= 1``).
        budget: How the active-connection budget is enforced.
            ``"per_neuron"`` gives every neuron the same number of inputs
            (like biological synapse budgets). ``"global"`` only fixes the
            total number of connections, so neurons may gain or lose inputs
            freely.
        bias: Whether the layer has a bias term.
        seed: Optional seed for the initial wiring and weights.

    Rewiring
        rewire_fraction: Fraction of active connections swapped (pruned and
            replaced by new ones) each time the layer rewires.
        rewire_interval: Rewire every this many training steps. ``0`` disables
            rewiring.
        rewire_stop: Step after which rewiring stops for good. ``None`` means
            rewiring never stops.
        rewire_decay: ``"cosine"`` shrinks ``rewire_fraction`` smoothly to zero
            at ``rewire_stop`` (needs ``rewire_stop``, otherwise behaves like
            ``"constant"``). ``"constant"`` keeps it fixed.
        grow: How new connections are chosen. ``"gradient"`` picks the
            currently inactive inputs whose gradient is largest, i.e. the
            connections that would reduce the loss most. ``"random"`` picks
            uniformly at random.
    """

    density: float = 0.5
    budget: str = "per_neuron"
    bias: bool = True
    seed: Optional[int] = None

    rewire_fraction: float = 0.1
    rewire_interval: int = 100
    rewire_stop: Optional[int] = None
    rewire_decay: str = "cosine"
    grow: str = "gradient"

    def __post_init__(self) -> None:
        if not (0.0 < self.density <= 1.0):
            raise ValueError(f"density must be in (0, 1], got {self.density}")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}, got {self.budget!r}")
        if not (0.0 <= self.rewire_fraction <= 1.0):
            raise ValueError(
                f"rewire_fraction must be in [0, 1], got {self.rewire_fraction}"
            )
        if self.rewire_interval < 0:
            raise ValueError(
                f"rewire_interval must be >= 0, got {self.rewire_interval}"
            )
        if self.rewire_stop is not None and self.rewire_stop < 0:
            raise ValueError(f"rewire_stop must be >= 0 or None, got {self.rewire_stop}")
        if self.rewire_decay not in DECAYS:
            raise ValueError(
                f"rewire_decay must be one of {DECAYS}, got {self.rewire_decay!r}"
            )
        if self.grow not in GROW_CRITERIA:
            raise ValueError(
                f"grow must be one of {GROW_CRITERIA}, got {self.grow!r}"
            )

    def replace(self, **changes: Any) -> "IONConfig":
        """Return a copy with some fields changed. Unknown names raise ``ValueError``."""
        _check_keys(changes)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IONConfig":
        _check_keys(data)
        return cls(**dict(data))


def _check_keys(keys: Mapping[str, Any]) -> None:
    valid = {f.name for f in dataclasses.fields(IONConfig)}
    unknown = sorted(set(keys) - valid)
    if unknown:
        raise ValueError(f"Unknown IONConfig option(s): {unknown}. Valid: {sorted(valid)}")
