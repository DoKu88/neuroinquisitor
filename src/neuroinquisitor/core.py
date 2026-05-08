"""NeuroInquisitor — top-level orchestrator."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.formats.base import Format
from neuroinquisitor.index.base import IndexEntry
from neuroinquisitor.index.json_index import JSONIndex
from neuroinquisitor.loader import load as _load
from neuroinquisitor.loader import resolve_backend as _resolve_backend
from neuroinquisitor.loader import resolve_format as _resolve_format

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


class NeuroInquisitor:
    """Attach to a PyTorch ``nn.Module`` to snapshot weights during training.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose parameters are snapshotted.
    log_dir:
        Directory where snapshots and the index are written.
    compress:
        Hint to the format to use compression.
    create_new:
        ``True``  → start a fresh run; raises :exc:`FileExistsError` if
        an index already exists in *log_dir*.
        ``False`` → append to an existing run; raises
        :exc:`FileNotFoundError` if no index is found.
    backend:
        Storage backend.  Pass ``"local"`` (default) or a
        :class:`~neuroinquisitor.backends.base.Backend` instance.
    format:
        Snapshot file format.  Pass ``"hdf5"`` (default) or a
        :class:`~neuroinquisitor.formats.base.Format` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: str | os.PathLike[str] = ".",
        compress: bool = False,
        create_new: bool = True,
        backend: str | Backend = "local",
        format: str | Format = "hdf5",
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

    @classmethod
    def load(
        cls,
        log_dir: str | os.PathLike[str],
        backend: str | Backend = "local",
        format: str | Format = "hdf5",
        epochs: int | list[int] | range | None = None,
        layers: str | list[str] | None = None,
    ) -> SnapshotCollection:
        """Open a run directory and return a lazy :class:`~neuroinquisitor.collection.SnapshotCollection`.

        Does not require a model — useful for post-training analysis.

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
            Restrict to a single epoch, a list of indices, or a :class:`range`.
            ``None`` (default) includes all epochs.
        layers:
            Restrict to a single layer name or a list of names.
            ``None`` (default) includes all layers.
        """
        return _load(log_dir, backend=backend, format=format, epochs=epochs, layers=layers)

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
