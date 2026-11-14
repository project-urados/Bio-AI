# ION AI v0.1.0

ION AI, formerly BIO AI, is an add-on to PyTorch that allows the use of our all-new ION AI neurons.
These neurons have high efficiency* and have some biological inspiration, hence the old name, "Bio AI".



*Efficiency: A bit more work done for less processing power.


![Visits](https://count.getloli.com/@milestone?theme=asoul&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

## Rewiring neurons

`ion` adds neurons whose connections rewire during training. A layer keeps a
fixed number of active inputs (its *budget*). Every `rewire_interval` steps it
prunes its weakest connections and grows new ones to the inputs with the
largest gradient, so the wiring adapts to the task instead of staying fixed.

```python
import torch
import ion
from ion import IONConfig, IONLinear, build_model

# A single layer
layer = IONLinear(784, 256, IONConfig(density=0.3, rewire_interval=50))

# A whole model from a dict or JSON, no model script needed
model = build_model({
    "defaults": {"density": 0.3, "rewire_interval": 50},
    "layers": [
        {"type": "ion_linear", "in": 784, "out": 256},
        {"type": "relu"},
        {"type": "ion_linear", "in": 256, "out": 10, "config": {"density": 0.8}},
    ],
})

# Training loop: call rewire once per step, after optimizer.step()
opt = torch.optim.Adam(model.parameters(), 1e-3)
for step, (x, y) in enumerate(loader, start=1):
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    ion.rewire(model, step, optimizer=opt)
```

### Config options (`IONConfig`)

| Option | Default | Meaning |
|---|---|---|
| `density` | `0.5` | Fraction of input→output connections active |
| `budget` | `"per_neuron"` | `"per_neuron"` fixes fan-in per neuron; `"global"` fixes the total count |
| `bias` | `True` | Use a bias term |
| `seed` | `None` | Seed for initial wiring and weights |
| `rewire_fraction` | `0.1` | Fraction of active connections swapped per rewire |
| `rewire_interval` | `100` | Rewire every N steps (`0` disables) |
| `rewire_stop` | `None` | Stop rewiring at this step |
| `rewire_decay` | `"cosine"` | `"cosine"` shrinks the fraction to 0 at `rewire_stop`; `"constant"` keeps it |
| `grow` | `"gradient"` | `"gradient"` grows the highest-gradient inputs; `"random"` grows uniformly |

Config can be set per layer (`config=`, keyword overrides, or the spec's
`"config"` key) or for all layers at once (spec `"defaults"`).

### Notes

- Rewiring preserves the budget: fan-in per neuron (or total connections) stays constant.
- Pruned and new connections start at weight 0. Passing `optimizer=` to `ion.rewire` also clears their optimizer state, so a new connection does not inherit momentum from the one it replaced.
- Spec layer types: `ion_linear`, `linear`, `dropout`, `layernorm`, and activations `relu`, `gelu`, `tanh`, `sigmoid`, `silu`, `identity`.

### Tests

```bash
pip install torch pytest
python -m pytest tests
```
