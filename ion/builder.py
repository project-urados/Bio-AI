"""Build PyTorch models from plain dict/JSON specs, with no model script needed.

Example::

    spec = {
        "defaults": {"density": 0.3, "rewire_interval": 50},   # applied to every ion layer
        "layers": [
            {"type": "ion_linear", "in": 784, "out": 256},
            {"type": "relu"},
            {"type": "ion_linear", "in": 256, "out": 10,
             "config": {"density": 0.8, "rewire_fraction": 0.05}},
        ],
    }
    model = build_model(spec)

Layer types: ``ion_linear``, ``linear``, ``dropout``, ``layernorm``, and the
activations ``relu``, ``gelu``, ``tanh``, ``sigmoid``, ``silu``, ``identity``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Union

import torch.nn as nn

from .config import IONConfig
from .layers import IONLinear

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def build_model(spec: Union[Mapping[str, Any], str]) -> nn.Sequential:
    """Build an ``nn.Sequential`` from a spec dict, or from a JSON string/path.

    Top-level keys:
        ``layers`` (required): list of layer dicts, applied in order.
        ``defaults`` (optional): :class:`IONConfig` fields applied to every
            ``ion_linear`` layer. A layer's own ``config`` overrides them.
    """
    if isinstance(spec, str):
        spec = _load_json(spec)
    if "layers" not in spec:
        raise ValueError("model spec must contain a 'layers' list")
    defaults = dict(spec.get("defaults", {}))
    modules = [_build_layer(item, defaults, i) for i, item in enumerate(spec["layers"])]
    return nn.Sequential(*modules)


def _load_json(text_or_path: str) -> Mapping[str, Any]:
    stripped = text_or_path.lstrip()
    if stripped.startswith("{"):
        return json.loads(text_or_path)
    with open(text_or_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_layer(item: Mapping[str, Any], defaults: Mapping[str, Any], index: int) -> nn.Module:
    if not isinstance(item, Mapping) or "type" not in item:
        raise ValueError(f"layer #{index} must be a dict with a 'type' key")
    kind = item["type"]
    args = {k: v for k, v in item.items() if k not in ("type", "config")}

    if kind == "ion_linear":
        cfg_dict = {**defaults, **item.get("config", {})}
        cfg = IONConfig.from_dict(cfg_dict)
        return IONLinear(_req(args, "in", index), _req(args, "out", index), config=cfg)
    if kind == "linear":
        return nn.Linear(_req(args, "in", index), _req(args, "out", index), bias=args.get("bias", True))
    if kind == "dropout":
        return nn.Dropout(args.get("p", 0.5))
    if kind == "layernorm":
        return nn.LayerNorm(_req(args, "size", index))
    if kind in _ACTIVATIONS:
        return _ACTIVATIONS[kind]()
    raise ValueError(
        f"layer #{index}: unknown type {kind!r}. Known: "
        f"ion_linear, linear, dropout, layernorm, {', '.join(sorted(_ACTIVATIONS))}"
    )


def _req(args: Mapping[str, Any], key: str, index: int) -> int:
    if key not in args:
        raise ValueError(f"layer #{index}: missing required key {key!r}")
    return int(args[key])
