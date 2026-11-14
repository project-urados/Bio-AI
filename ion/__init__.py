"""ION AI: PyTorch neurons whose connections rewire during training.

Quick start::

    import ion
    from ion import IONConfig, IONLinear, build_model, rewire

    # 1. Drop-in layer
    layer = IONLinear(784, 256, IONConfig(density=0.3, rewire_interval=50))

    # 2. Whole model from a spec (no custom model script needed)
    model = build_model({
        "defaults": {"density": 0.3, "rewire_interval": 50},
        "layers": [
            {"type": "ion_linear", "in": 784, "out": 256},
            {"type": "relu"},
            {"type": "ion_linear", "in": 256, "out": 10},
        ],
    })

    # 3. Training loop
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for step, (x, y) in enumerate(loader, start=1):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        ion.rewire(model, step, optimizer=opt)
"""

from .config import IONConfig
from .layers import IONLinear, RewireResult
from .rewire import ion_layers, ion_parameters, rewire, rewire_stats
from .builder import build_model

__all__ = [
    "IONConfig",
    "IONLinear",
    "RewireResult",
    "build_model",
    "ion_layers",
    "ion_parameters",
    "rewire",
    "rewire_stats",
]

__version__ = "0.1.0"
