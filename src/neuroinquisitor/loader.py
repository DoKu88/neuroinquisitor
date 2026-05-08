"""Standalone loader — single entry point for reading snapshot collections."""

from __future__ import annotations

import os
from pathlib import Path

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.backends.local import LocalBackend
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.formats.base import Format
from neuroinquisitor.formats.hdf5_format import HDF5Format
from neuroinquisitor.index.json_index import JSONIndex

_BACKENDS: dict[str, type[Backend]] = {
    "local": LocalBackend,
}
_FORMATS: dict[str, type[Format]] = {
    "hdf5": HDF5Format,
}


def resolve_backend(spec: str | Backend, log_dir: Path) -> Backend:
    if isinstance(spec, Backend):
        return spec
    if spec not in _BACKENDS:
        raise ValueError(f"Unknown backend {spec!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[spec](log_dir)


def resolve_format(spec: str | Format) -> Format:
    if isinstance(spec, Format):
        return spec
    if spec not in _FORMATS:
        raise ValueError(f"Unknown format {spec!r}. Available: {sorted(_FORMATS)}")
    return _FORMATS[spec]()


def load(
    log_dir: str | os.PathLike[str],
    backend: str | Backend = "local",
    format: str | Format = "hdf5",
    epochs: int | list[int] | range | None = None,
    layers: str | list[str] | None = None,
) -> SnapshotCollection:
    """Load snapshots from *log_dir* and return a lazy :class:`~neuroinquisitor.collection.SnapshotCollection`.

    No tensor data is read from disk until the caller requests it via
    :meth:`~neuroinquisitor.collection.SnapshotCollection.by_epoch` or
    :meth:`~neuroinquisitor.collection.SnapshotCollection.by_layer`.

    Parameters
    ----------
    log_dir:
        Directory containing snapshots and the index file.
    backend:
        Storage backend — ``"local"`` (default) or a
        :class:`~neuroinquisitor.backends.base.Backend` instance.
    format:
        Snapshot file format — ``"hdf5"`` (default) or a
        :class:`~neuroinquisitor.formats.base.Format` instance.
    epochs:
        Restrict the collection to a single epoch index, a list of indices,
        or a :class:`range`.  ``None`` (default) includes all epochs.
    layers:
        Restrict the collection to a single layer name or a list of names.
        ``None`` (default) includes all layers.
    """
    root = Path(log_dir)
    _backend = resolve_backend(backend, root)
    _format = resolve_format(format)
    _index = JSONIndex.load(_backend)
    col = SnapshotCollection(_backend, _format, _index)
    if epochs is not None or layers is not None:
        col = col.select(epochs=epochs, layers=layers)
    return col
