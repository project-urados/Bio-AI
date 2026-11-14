import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import ion
from ion import IONLinear, IONConfig
from ion.data import ImageDataset, ImageIterableDataset, image_to_tensor, wrap_image_dataset

datasets = pytest.importorskip("datasets")


def _sprite(seed):
    rng = np.random.default_rng(seed)
    rgba = rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(rng.random((16, 16)) < 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


@pytest.fixture
def hf_split():
    images = [_sprite(i) for i in range(6)]
    return datasets.Dataset.from_dict({
        "image": images,
        "caption": [f"sprite {i}" for i in range(6)],
        "subject": [f"s{i}" for i in range(6)],
        "row_id": list(range(6)),
    }).cast_column("image", datasets.Image())


def test_image_to_tensor_pil_rgba_shape_and_range():
    t = image_to_tensor(_sprite(0), "RGBA")
    assert t.shape == (4, 16, 16)
    assert t.dtype == torch.float32
    assert 0.0 <= t.min() and t.max() <= 1.0


def test_image_to_tensor_flatten_and_modes():
    assert image_to_tensor(_sprite(0), "RGBA", flatten=True).shape == (1024,)
    assert image_to_tensor(_sprite(0), "RGB").shape == (3, 16, 16)
    assert image_to_tensor(_sprite(0), "L").shape == (1, 16, 16)


def test_image_to_tensor_array_input_and_bad_shape():
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    assert image_to_tensor(arr).shape == (4, 16, 16)
    with pytest.raises(ValueError):
        image_to_tensor(np.zeros((16, 16, 3), dtype=np.uint8), "RGBA")
    with pytest.raises(ValueError):
        image_to_tensor(arr, "CMYK")


def test_map_dataset_returns_image_and_metadata(hf_split):
    ds = wrap_image_dataset(hf_split)
    assert isinstance(ds, ImageDataset)
    assert len(ds) == 6
    item = ds[2]
    assert item["image"].shape == (4, 16, 16)
    assert item["caption"] == "sprite 2"
    assert item["subject"] == "s2"
    assert item["row_id"] == 2


def test_keep_filters_columns_and_flatten_works(hf_split):
    ds = wrap_image_dataset(hf_split, flatten=True, keep=["caption"])
    item = ds[0]
    assert set(item) == {"image", "caption"}
    assert item["image"].shape == (1024,)


def test_dataloader_batches_and_collates_strings(hf_split):
    loader = DataLoader(wrap_image_dataset(hf_split), batch_size=4)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 4, 16, 16)
    assert len(batch["caption"]) == 4 and all(isinstance(c, str) for c in batch["caption"])


def test_iterable_dataset_streaming_matches_map(hf_split):
    stream = hf_split.to_iterable_dataset()
    ds = wrap_image_dataset(stream)
    assert isinstance(ds, ImageIterableDataset)
    items = list(ds)
    assert len(items) == 6
    assert items[0]["image"].shape == (4, 16, 16)
    assert [i["row_id"] for i in items] == list(range(6))


def test_iterable_shuffle_keeps_wrapper_settings(hf_split):
    ds = wrap_image_dataset(hf_split.to_iterable_dataset(), flatten=True, keep=["caption"])
    shuffled = ds.shuffle(seed=0, buffer_size=6)
    assert isinstance(shuffled, ImageIterableDataset)
    item = next(iter(shuffled))
    assert item["image"].shape == (1024,) and set(item) == {"image", "caption"}


def test_missing_image_column_is_a_clear_error(hf_split):
    with pytest.raises(ValueError, match="not found"):
        wrap_image_dataset(hf_split, image_column="picture")


def test_transform_is_applied(hf_split):
    ds = wrap_image_dataset(hf_split, transform=lambda t: t * 0)
    assert ds[0]["image"].abs().sum() == 0


def test_end_to_end_training_on_image_batches(hf_split):
    torch.manual_seed(0)
    loader = DataLoader(wrap_image_dataset(hf_split, flatten=True), batch_size=3)
    model = torch.nn.Sequential(
        IONLinear(1024, 32, IONConfig(density=0.2, rewire_interval=2, seed=0)),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    step = 0
    for _ in range(3):
        for batch in loader:
            step += 1
            loss = model(batch["image"]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ion.rewire(model, step, optimizer=opt)
    assert step == 6
    assert int(model[0].rewire_events) >= 1


def test_load_image_dataset_wraps_splits_and_shuffles_only_train(monkeypatch, hf_split):
    from ion import data as data_mod

    def fake_load(name, split=None, streaming=False, **kw):
        ds = hf_split.to_iterable_dataset() if streaming else hf_split
        return {"train": ds, "validation": ds}

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    splits = data_mod.load_image_dataset("fake/name", streaming=True, shuffle_buffer=6)
    assert set(splits) == {"train", "validation"}
    assert all(isinstance(v, ImageIterableDataset) for v in splits.values())
    assert splits["train"].data is not hf_split  # shuffled wrapper
    assert splits["validation"].data is not None


def test_shuffle_buffer_requires_streaming(monkeypatch, hf_split):
    from ion import data as data_mod
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: hf_split)
    with pytest.raises(ValueError, match="streaming"):
        data_mod.load_image_dataset("fake/name", shuffle_buffer=10)
