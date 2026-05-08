"""NeuroInquisitor — top-level orchestrator."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.backends.local import LocalBackend
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.formats.base import Format
from neuroinquisitor.formats.hdf5_format import HDF5Format
from neuroinquisitor.formats.safetensors_format import SafeTensorsFormat
from neuroinquisitor.index.base import IndexEntry
from neuroinquisitor.index.json_index import JSONIndex

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[Backend]] = {
    "local": LocalBackend,
}
_FORMATS: dict[str, type[Format]] = {
    "safetensors": SafeTensorsFormat,
    "hdf5": HDF5Format,
}


def _resolve_backend(spec: str | Backend, log_dir: Path) -> Backend:
    if isinstance(spec, Backend):
        return spec
    if spec not in _BACKENDS:
        raise ValueError(f"Unknown backend {spec!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[spec](log_dir)


def _resolve_format(spec: str | Format) -> Format:
    if isinstance(spec, Format):
        return spec
    if spec not in _FORMATS:
        raise ValueError(f"Unknown format {spec!r}. Available: {sorted(_FORMATS)}")
    return _FORMATS[spec]()


class NeuroInquisitor:
    """Attach to a PyTorch ``nn.Module`` to snapshot weights during training.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose parameters are snapshotted.
    log_dir:
        Directory where snapshots and the index are written.
    compress:
        Hint to the format to use compression.  Has no effect for
        ``SafeTensorsFormat`` (which does not support compression).
    create_new:
        ``True``  → start a fresh run; raises :exc:`FileExistsError` if
        an index already exists in *log_dir*.
        ``False`` → append to an existing run; raises
        :exc:`FileNotFoundError` if no index is found.
    backend:
        Storage backend.  Pass ``"local"`` (default) or a
        :class:`~neuroinquisitor.backends.base.Backend` instance.
    format:
        Snapshot file format.  Pass ``"safetensors"`` (default) or a
        :class:`~neuroinquisitor.formats.base.Format` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: str | os.PathLike[str] = ".",
        compress: bool = False,
        create_new: bool = True,
        backend: str | Backend = "local",
        format: str | Format = "safetensors",
    ) -> None:
        self._model = model
        self._compress = compress
        self._closed = False

        self._log_dir = Path(log_dir)
        self._backend = _resolve_backend(backend, self._log_dir)
        self._format = _resolve_format(format)

        index_exists = self._backend.exists("index.json")

        if create_new:
            if index_exists:
                raise FileExistsError(
                    f"A snapshot index already exists in {self._log_dir!r}. "
                    "Pass create_new=False to append to it, or use a different log_dir."
                )
            self._index = JSONIndex(self._backend)
            self._index.save()
        else:
            if not index_exists:
                raise FileNotFoundError(
                    f"No snapshot index found in {self._log_dir!r}. "
                    "Pass create_new=True to start a new run."
                )
            self._index = JSONIndex.load(self._backend)

        logger.info(
            "NeuroInquisitor opened %s (create_new=%s, backend=%s, format=%s)",
            self._log_dir,
            create_new,
            type(self._backend).__name__,
            type(self._format).__name__,
        )

    # ------------------------------------------------------------------
    # Snapshot key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_key(epoch: int | None, step: int | None, extension: str) -> str:
        parts: list[str] = []
        if epoch is not None:
            parts.append(f"epoch_{epoch:04d}")
        if step is not None:
            parts.append(f"step_{step:06d}")
        return "_".join(parts) + extension

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        epoch: int | None = None,
        step: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Snapshot all model parameters and persist to the backend.

        Parameters
        ----------
        epoch:
            Training epoch index.
        step:
            Training step index.  At least one of *epoch* or *step* must
            be supplied.
        metadata:
            Arbitrary scalar values stored alongside the snapshot (e.g.
            ``{"loss": 0.42, "lr": 1e-3}``).  The keys ``"epoch"`` and
            ``"step"`` are reserved.
        """
        if self._closed:
            raise RuntimeError("Cannot snapshot: NeuroInquisitor is closed.")
        if epoch is None and step is None:
            raise ValueError("At least one of epoch or step must be provided.")

        if metadata is not None:
            reserved = frozenset({"epoch", "step"}) & metadata.keys()
            if reserved:
                raise ValueError(
                    f"Metadata contains reserved key(s) {sorted(reserved)}. "
                    "Choose different key names."
                )

        file_key = self._snapshot_key(epoch, step, self._format.extension)
        if self._index.contains_key(file_key):
            label = self._snapshot_key(epoch, step, "")
            raise ValueError(
                f"Snapshot {label!r} already exists. "
                "Each (epoch, step) combination must be unique."
            )

        params: dict[str, np.ndarray] = {
            name: param.detach().cpu().numpy()
            for name, param in self._model.named_parameters()
        }

        file_metadata: dict[str, object] = {}
        if epoch is not None:
            file_metadata["epoch"] = epoch
        if step is not None:
            file_metadata["step"] = step
        if metadata:
            file_metadata.update(metadata)

        data = self._format.write(params, file_metadata, compress=self._compress)
        self._backend.write(file_key, data)

        layer_names = list(params.keys())
        self._index.add(
            IndexEntry(
                epoch=epoch,
                step=step,
                file_key=file_key,
                layers=layer_names,
                metadata=metadata or {},
            )
        )
        logger.debug("Snapshot written: %s", file_key)

    # ------------------------------------------------------------------
    # Read-back
    # ------------------------------------------------------------------

    def load_snapshot(self, epoch: int) -> dict[str, np.ndarray]:
        """Load a single snapshot by epoch index."""
        entry = self._index.get_by_epoch(epoch)
        if entry is None:
            available = [e.epoch for e in self._index.all() if e.epoch is not None]
            raise KeyError(
                f"No snapshot for epoch {epoch}. Available epochs: {sorted(available)}"
            )
        path = self._backend.read_path(entry.file_key)
        return self._format.read(path)

    def load_all_snapshots(self) -> SnapshotCollection:
        """Return a lazy :class:`~neuroinquisitor.collection.SnapshotCollection`.

        Only the in-memory index is consulted — no tensor files are opened
        until the caller requests actual weight data.
        """
        return SnapshotCollection(self._backend, self._format, self._index)

    @classmethod
    def load(
        cls,
        log_dir: str | os.PathLike[str],
        backend: str | Backend = "local",
        format: str | Format = "safetensors",
    ) -> SnapshotCollection:
        """Open a run directory and return a lazy :class:`~neuroinquisitor.collection.SnapshotCollection`.

        Does not require a model — useful for post-training analysis.
        """
        root = Path(log_dir)
        _backend = _resolve_backend(backend, root)
        _format = _resolve_format(format)
        _index = JSONIndex.load(_backend)
        return SnapshotCollection(_backend, _format, _index)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalise the run.  Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        logger.info("NeuroInquisitor closed %s", self._log_dir)

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            warnings.warn(
                f"NeuroInquisitor for {str(self._log_dir)!r} was not explicitly closed. "
                "Call .close() when done.",
                ResourceWarning,
                stacklevel=2,
            )
            self.close()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (
            f"NeuroInquisitor("
            f"model={type(self._model).__name__}, "
            f"log_dir={str(self._log_dir)!r}, "
            f"backend={type(self._backend).__name__}, "
            f"format={type(self._format).__name__}, "
            f"compress={self._compress}, "
            f"status={status!r})"
        )
