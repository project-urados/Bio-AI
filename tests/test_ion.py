import json

import pytest
import torch
import torch.nn as nn

import ion
from ion import IONConfig, IONLinear, build_model, rewire, rewire_stats


def _cfg(**kw):
    base = dict(density=0.25, rewire_interval=10, rewire_fraction=0.2, seed=0)
    base.update(kw)
    return IONConfig(**base)


# ---------------------------------------------------------------- config
def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        IONConfig(density=0.0)
    with pytest.raises(ValueError):
        IONConfig(budget="nope")
    with pytest.raises(ValueError):
        IONConfig(rewire_fraction=1.5)
    with pytest.raises(ValueError):
        IONConfig(grow="magic")


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown IONConfig option"):
        IONConfig.from_dict({"densty": 0.5})


def test_config_roundtrip_and_replace():
    c = IONConfig(density=0.4, rewire_stop=100)
    assert IONConfig.from_dict(c.to_dict()) == c
    assert c.replace(density=0.9).density == 0.9
    assert c.density == 0.4  # original untouched


# ---------------------------------------------------------------- layer
def test_initial_fan_in_matches_density_per_neuron():
    layer = IONLinear(40, 16, _cfg(density=0.25))
    assert (layer.mask.sum(dim=1) == 10).all()
    assert layer.density() == pytest.approx(0.25)


def test_inactive_weights_are_zero_and_forward_matches_masked_dense():
    layer = IONLinear(8, 4, _cfg(density=0.5))
    x = torch.randn(3, 8)
    expected = x @ (layer.weight * layer.mask).t() + layer.bias
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert (layer.weight[~layer.mask] == 0).all()


def test_seed_makes_wiring_reproducible():
    a = IONLinear(20, 5, _cfg(seed=7))
    b = IONLinear(20, 5, _cfg(seed=7))
    c = IONLinear(20, 5, _cfg(seed=8))
    assert torch.equal(a.mask, b.mask) and torch.equal(a.weight, b.weight)
    assert not torch.equal(a.mask, c.mask)


def test_overrides_kwargs_apply_on_top_of_config():
    layer = IONLinear(10, 3, _cfg(), density=1.0)
    assert layer.config.density == 1.0
    assert layer.mask.all()


def test_rewire_preserves_per_neuron_fan_in_and_changes_wiring():
    torch.manual_seed(0)
    layer = IONLinear(30, 6, _cfg(density=0.3, rewire_fraction=0.3))
    before = layer.mask.clone()
    x = torch.randn(16, 30)
    layer.train()
    layer(x).sum().backward()
    res = layer.rewire(step=10)
    assert res is not None
    assert (layer.mask.sum(dim=1) == before.sum(dim=1)).all()
    assert res.pruned == res.grown > 0
    assert not torch.equal(before, layer.mask)
    # Pruned connections are off, and weights of inactive connections are zero.
    assert (layer.weight[~layer.mask] == 0).all()


def test_rewire_preserves_global_connection_count():
    torch.manual_seed(0)
    layer = IONLinear(20, 10, _cfg(budget="global", density=0.3, rewire_fraction=0.25))
    n = int(layer.mask.sum())
    layer(torch.randn(8, 20)).sum().backward()
    layer.rewire(step=10)
    assert int(layer.mask.sum()) == n


def test_gradient_growth_picks_highest_gradient_inactive_connections():
    # One output, 4 inputs, budget of 1. Input 3 has the biggest gradient
    # signal but is inactive, so growth should pick it.
    layer = IONLinear(4, 1, _cfg(density=0.25, rewire_fraction=1.0, bias=False))
    with torch.no_grad():
        layer.mask[:] = torch.tensor([[True, False, False, False]])
        layer.weight[:] = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    x = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    layer.train()
    layer(x).sum().backward()
    layer.rewire(step=10)
    assert layer.mask[0].tolist() == [False, False, False, True]


def test_no_rewire_outside_interval_or_when_disabled():
    layer = IONLinear(8, 2, _cfg(rewire_interval=10))
    assert layer.rewire(step=5) is None
    assert layer.rewire(step=0) is None
    off = IONLinear(8, 2, _cfg(rewire_interval=0))
    assert off.rewire(step=10) is None


def test_rewire_stop_halts_rewiring():
    layer = IONLinear(8, 2, _cfg(rewire_interval=5, rewire_stop=20, rewire_decay="constant"))
    assert layer.should_rewire(15)
    assert not layer.should_rewire(20)
    assert not layer.should_rewire(25)


def test_cosine_decay_shrinks_fraction_to_zero():
    layer = IONLinear(8, 2, _cfg(rewire_fraction=0.4, rewire_stop=100, rewire_interval=10))
    assert layer.fraction_at(0) == pytest.approx(0.4)
    assert layer.fraction_at(50) == pytest.approx(0.2)
    assert layer.fraction_at(100) == pytest.approx(0.0, abs=1e-9)


def test_random_growth_mode_runs():
    layer = IONLinear(16, 4, _cfg(grow="random"))
    layer(torch.randn(4, 16)).sum().backward()
    res = layer.rewire(step=10)
    assert res is not None and res.grown == res.pruned


def test_full_density_layer_does_not_crash_on_rewire():
    layer = IONLinear(6, 3, _cfg(density=1.0))
    layer(torch.randn(2, 6)).sum().backward()
    res = layer.rewire(step=10)
    assert res.pruned == 0 and res.grown == 0


def test_rewire_events_persist_in_state_dict_but_grad_buffers_do_not():
    layer = IONLinear(8, 2, _cfg())
    layer(torch.randn(2, 8)).sum().backward()
    layer.rewire(step=10)
    sd = layer.state_dict()
    assert "rewire_events" in sd
    assert "_grad_sum" not in sd
    other = IONLinear(8, 2, _cfg(seed=99))
    other.load_state_dict(sd)
    assert torch.equal(other.mask, layer.mask)


# ---------------------------------------------------------------- module-level API
def test_rewire_helper_rewires_all_layers_and_resets_optimizer_state():
    torch.manual_seed(0)
    model = nn.Sequential(
        IONLinear(12, 8, _cfg(density=0.3)),
        nn.ReLU(),
        IONLinear(8, 3, _cfg(density=0.5, rewire_interval=10)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(32, 12)
    model(x).pow(2).mean().backward()
    opt.step()
    results = rewire(model, step=10, optimizer=opt)
    assert len(results) == 2
    first = model[0].weight
    state = opt.state[first]
    assert (state["exp_avg"][results[0].changed] == 0).all()


def test_build_model_from_dict_and_defaults_override():
    spec = {
        "defaults": {"density": 0.5, "rewire_interval": 7},
        "layers": [
            {"type": "ion_linear", "in": 10, "out": 6},
            {"type": "gelu"},
            {"type": "ion_linear", "in": 6, "out": 2, "config": {"density": 1.0}},
        ],
    }
    model = build_model(spec)
    layers = [m for m in model if isinstance(m, IONLinear)]
    assert layers[0].config.density == 0.5 and layers[0].config.rewire_interval == 7
    assert layers[1].config.density == 1.0 and layers[1].config.rewire_interval == 7
    assert model(torch.randn(4, 10)).shape == (4, 2)


def test_build_model_from_json_string_and_file(tmp_path):
    spec = {"layers": [{"type": "ion_linear", "in": 4, "out": 2}]}
    assert len(build_model(json.dumps(spec))) == 1
    p = tmp_path / "m.json"
    p.write_text(json.dumps(spec))
    assert len(build_model(str(p))) == 1


def test_build_model_errors_are_clear():
    with pytest.raises(ValueError, match="'layers'"):
        build_model({})
    with pytest.raises(ValueError, match="unknown type"):
        build_model({"layers": [{"type": "lstm"}]})
    with pytest.raises(ValueError, match="missing required key 'out'"):
        build_model({"layers": [{"type": "ion_linear", "in": 3}]})


def test_mixed_model_trains_and_loss_decreases():
    torch.manual_seed(0)
    model = build_model({
        "defaults": {"density": 0.5, "rewire_interval": 20, "rewire_fraction": 0.1},
        "layers": [
            {"type": "ion_linear", "in": 2, "out": 32},
            {"type": "tanh"},
            {"type": "ion_linear", "in": 32, "out": 1},
        ],
    })
    x = torch.rand(256, 2) * 2 - 1
    y = (x[:, :1] * x[:, 1:]).sign()  # XOR-like target
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    losses = []
    for step in range(1, 201):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rewire(model, step, optimizer=opt)
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8
    stats = rewire_stats(model)
    assert set(stats) == {"0", "2"}
    assert all(0 < d <= 1 for d in stats.values())


def test_public_api_exports():
    for name in ["IONConfig", "IONLinear", "build_model", "rewire", "rewire_stats", "ion_layers"]:
        assert hasattr(ion, name)
