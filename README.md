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

## Training on Hugging Face image datasets

`ion.data` takes the output of `datasets.load_dataset` and returns PyTorch
datasets whose items are dicts: `image` (float tensor in `[0, 1]`) plus the
other columns. It works with the Pixel Art Atlas dataset, which has `image`,
`caption`, `subject` and `row_id` columns:

```python
from torch.utils.data import DataLoader
from ion import IONLinear, build_model, load_image_dataset, rewire

# Same as load_dataset("Angais/pixel-art-atlas-1.4m"), but wrapped for PyTorch
splits = load_image_dataset(
    "Angais/pixel-art-atlas-1.4m",
    streaming=True,       # no 1.4M-image download
    flatten=True,         # (B, 1024) vectors for MLPs; omit for (B, 4, 16, 16)
    keep=["caption"],     # columns to return besides "image"
    shuffle_buffer=10_000,  # train split only; see memory note below
)
train_loader = DataLoader(splits["train"], batch_size=128)
val_loader = DataLoader(splits["validation"], batch_size=128)

for batch in train_loader:
    batch["image"]    # (128, 1024) float tensor, RGBA values in [0, 1]
    batch["caption"]  # list[str]
    ...
```

- Without `streaming=True` you get map-style datasets, which support random access
  and `len()`. Downloading the full set is about 1.2 GB.
- `image_to_tensor` converts a single PIL image or array, and `wrap_image_dataset`
  wraps a dataset you already loaded yourself.
- Memory: `shuffle_buffer` keeps that many decoded examples in RAM (about 4 KB each
  as float32, so 10,000 is roughly 40 MB). It uses a local buffer, not the `datasets`
  library's shuffle, which ran out of memory on a 1 GB machine even at small sizes.
  Streaming without shuffling needs about 0.5 GB on this dataset.
- `datasets` is imported only when you call `load_image_dataset`, so the rest of
  `ion` works without it.
