"""Feed Hugging Face ``datasets`` image datasets into ION / PyTorch training.

Works with the output of ``load_dataset`` directly, for both downloaded and
streaming datasets, including image datasets such as
``Angais/pixel-art-atlas-1.4m`` (columns: ``image``, ``caption``, ``subject``,
``row_id``)::

    from ion.data import load_image_dataset

    splits = load_image_dataset("Angais/pixel-art-atlas-1.4m", streaming=True,
                                shuffle_buffer=10_000)  # shuffle only the train split
    loader = DataLoader(splits["train"], batch_size=128)

    for batch in loader:
        batch["image"]    # float tensor (B, 4, 16, 16) in [0, 1], RGBA
        batch["caption"]  # list[str]
        batch["subject"]  # list[str]

Pass ``flatten=True`` to get ``(B, 1024)`` vectors for MLPs.

``datasets`` is imported lazily, so the rest of ``ion`` works without it.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

_CHANNELS = {"L": 1, "RGB": 3, "RGBA": 4}


def image_to_tensor(image: Any, mode: str = "RGBA", flatten: bool = False) -> torch.Tensor:
    """Convert a PIL image or a uint8 HWC array to a float tensor in ``[0, 1]``.

    Returns shape ``(C, H, W)``, or ``(C*H*W,)`` when ``flatten=True``.
    Float arrays are assumed to be in ``[0, 1]`` already and are not rescaled.
    """
    if mode not in _CHANNELS:
        raise ValueError(f"mode must be one of {sorted(_CHANNELS)}, got {mode!r}")

    if hasattr(image, "convert"):  # PIL image: let PIL handle mode conversion
        arr = np.asarray(image.convert(mode))
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale: add the channel axis
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[-1] != _CHANNELS[mode]:
        raise ValueError(
            f"expected an HWC array with {_CHANNELS[mode]} channel(s) for mode {mode!r}, "
            f"got shape {arr.shape}"
        )

    t = torch.from_numpy(np.array(arr, copy=True))
    t = t.to(torch.float32)
    if arr.dtype == np.uint8:
        t = t / 255.0
    t = t.permute(2, 0, 1).contiguous()  # HWC -> CHW
    return t.reshape(-1) if flatten else t


class _ImageConverter:
    """Shared per-example conversion used by both dataset wrappers."""

    def __init__(
        self,
        image_column: str,
        mode: str,
        flatten: bool,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
        keep: Optional[Sequence[str]],
    ) -> None:
        self.image_column = image_column
        self.mode = mode
        self.flatten = flatten
        self.transform = transform
        self.keep = None if keep is None else tuple(keep)

    def __call__(self, example: Mapping[str, Any]) -> dict:
        image = image_to_tensor(example[self.image_column], self.mode, self.flatten)
        if self.transform is not None:
            image = self.transform(image)
        out = {"image": image}
        if self.keep is None:
            others = (k for k in example if k != self.image_column)
        else:
            others = (k for k in self.keep if k in example)
        for key in others:
            out[key] = example[key]
        return out


class ImageDataset(Dataset):
    """Map-style PyTorch dataset over a downloaded Hugging Face split.

    Each item is a dict with ``"image"`` (tensor) plus the other columns
    (for example ``"caption"``, ``"subject"``, ``"row_id"``).
    """

    def __init__(self, hf_dataset: Any, image_column: str = "image", mode: str = "RGBA",
                 flatten: bool = False, transform: Optional[Callable] = None,
                 keep: Optional[Sequence[str]] = None) -> None:
        _check_column(hf_dataset, image_column)
        self.data = hf_dataset
        self._convert = _ImageConverter(image_column, mode, flatten, transform, keep)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return self._convert(self.data[index])


def _buffered_shuffle(items: Iterator[Any], size: int, seed: int) -> Iterator[Any]:
    """Approximate shuffle of a stream using a fixed-size buffer.

    Memory is bounded by ``size`` items, unlike ``datasets``' built-in shuffle
    buffer, which ran out of memory on small machines.
    """
    rng = random.Random(seed)
    buf: list = []
    for item in items:
        if len(buf) < size:
            buf.append(item)
            continue
        i = rng.randrange(size)
        yield buf[i]
        buf[i] = item
    rng.shuffle(buf)
    yield from buf


class ImageIterableDataset(IterableDataset):
    """Streaming PyTorch dataset over a Hugging Face ``IterableDataset``.

    The Hugging Face iterable splits its shards across DataLoader workers
    itself, so ``num_workers > 0`` does not duplicate examples.

    ``shuffle_buffer`` shuffles through a local buffer of converted examples.
    Each pass over the data uses the same order for a given ``shuffle_seed``.
    """

    def __init__(self, hf_iterable: Any, image_column: str = "image", mode: str = "RGBA",
                 flatten: bool = False, transform: Optional[Callable] = None,
                 keep: Optional[Sequence[str]] = None,
                 shuffle_buffer: Optional[int] = None, shuffle_seed: int = 0) -> None:
        _check_column(hf_iterable, image_column)
        if shuffle_buffer is not None and shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be >= 1 or None")
        self.data = hf_iterable
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        self._convert = _ImageConverter(image_column, mode, flatten, transform, keep)

    def __iter__(self) -> Iterator[dict]:
        items = (self._convert(example) for example in self.data)
        if self.shuffle_buffer is None:
            return items
        return _buffered_shuffle(items, self.shuffle_buffer, self.shuffle_seed)

    def shuffle(self, seed: int = 0, buffer_size: int = 1000) -> "ImageIterableDataset":
        """Return a copy that shuffles through a local buffer of ``buffer_size`` examples."""
        c = self._convert
        return ImageIterableDataset(
            self.data, image_column=c.image_column, mode=c.mode, flatten=c.flatten,
            transform=c.transform, keep=c.keep,
            shuffle_buffer=buffer_size, shuffle_seed=seed,
        )


def wrap_image_dataset(
    data: Any,
    image_column: str = "image",
    mode: str = "RGBA",
    flatten: bool = False,
    transform: Optional[Callable] = None,
    keep: Optional[Sequence[str]] = None,
    shuffle_buffer: Optional[int] = None,
    shuffle_seed: int = 0,
) -> Union[ImageDataset, ImageIterableDataset]:
    """Wrap a Hugging Face ``Dataset`` or streaming ``IterableDataset`` for PyTorch.

    Streaming datasets (``IterableDataset``) are detected automatically.
    """
    if _is_streaming(data):
        return ImageIterableDataset(
            data, image_column=image_column, mode=mode, flatten=flatten,
            transform=transform, keep=keep,
            shuffle_buffer=shuffle_buffer, shuffle_seed=shuffle_seed,
        )
    if shuffle_buffer is not None:
        raise ValueError("shuffle_buffer only applies to streaming datasets; "
                         "use hf_dataset.shuffle() for downloaded datasets")
    return ImageDataset(data, image_column=image_column, mode=mode, flatten=flatten,
                        transform=transform, keep=keep)


def load_image_dataset(
    name: str,
    split: Optional[str] = None,
    streaming: bool = False,
    image_column: str = "image",
    mode: str = "RGBA",
    flatten: bool = False,
    transform: Optional[Callable] = None,
    keep: Optional[Sequence[str]] = None,
    shuffle_buffer: Optional[int] = None,
    shuffle_seed: int = 0,
    **load_kwargs: Any,
) -> Any:
    """Load a Hugging Face image dataset and wrap it for PyTorch in one call.

    With ``split=None`` (the default) this returns a ``dict`` of wrapped splits,
    for example ``{"train": ..., "validation": ...}``, which matches
    ``load_dataset(name)``. With a split name it returns one wrapped split.

    ``shuffle_buffer`` (streaming only) shuffles the train split through a
    local buffer of that many converted examples. Memory is about
    ``buffer_size * image bytes`` (a 16x16 RGBA sprite is 1 KB as uint8, 4 KB
    as float32). Use ``None`` to stream in dataset order.

    Extra keyword arguments go to ``datasets.load_dataset``
    (for example ``data_files`` or ``revision``).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("load_image_dataset needs the 'datasets' package: pip install datasets") from exc

    if shuffle_buffer is not None and not streaming:
        raise ValueError("shuffle_buffer only applies when streaming=True; "
                         "call .shuffle() on a downloaded dataset instead")
    raw = load_dataset(name, split=split, streaming=streaming, **load_kwargs)

    def prepare(ds: Any, shuffle: bool) -> Any:
        return wrap_image_dataset(
            ds, image_column=image_column, mode=mode, flatten=flatten,
            transform=transform, keep=keep,
            shuffle_buffer=shuffle_buffer if shuffle else None, shuffle_seed=shuffle_seed,
        )

    if isinstance(raw, Mapping):  # DatasetDict / IterableDatasetDict
        # Shuffle only the training split; validation stays in order.
        return {k: prepare(v, shuffle=(k == "train")) for k, v in raw.items()}
    return prepare(raw, shuffle=True)


def _is_streaming(data: Any) -> bool:
    try:
        from datasets import IterableDataset as HFIterable
    except ImportError:  # pragma: no cover
        HFIterable = ()
    return isinstance(data, IterableDataset) or (HFIterable and isinstance(data, HFIterable))


def _check_column(data: Any, column: str) -> None:
    columns = getattr(data, "column_names", None)
    # Streaming datasets may not know their columns until the first example.
    if columns is not None and column not in columns:
        raise ValueError(f"image column {column!r} not found. Available columns: {list(columns)}")
