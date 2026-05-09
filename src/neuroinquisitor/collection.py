"""SnapshotCollection — lazy, filterable view over a set of snapshots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.formats.base import Format
from neuroinquisitor.index.base import Index, IndexEntry

if TYPE_CHECKING:
    pass


class SnapshotCollection:
    """A lazy, filterable view over snapshots stored by a NeuroInquisitor.

    Obtain an instance via
    :meth:`~neuroinquisitor.core.NeuroInquisitor.load_all_snapshots` or
    :meth:`~neuroinquisitor.core.NeuroInquisitor.load`.

    No tensor data is read from disk until you call :meth:`by_epoch` or
    :meth:`by_layer`.  :meth:`select` composes filter sets and returns a
    new :class:`SnapshotCollection` — zero I/O.

    Access patterns
    ---------------
    - ``col.by_epoch(3)``            → ``dict[str, np.ndarray]`` for epoch 3
    - ``col.by_layer("fc1.weight")`` → ``dict[int, np.ndarray]`` mapping epoch → array
                                       (reads are issued in parallel)
    - ``col.select(epochs=range(0, 10), layers=["fc1.weight", "fc1.bias"])``
      → new :class:`SnapshotCollection` covering only the requested subset
    """

    def __init__(
        self,
        backend: Backend,
        format: Format,
        index: Index,
        epoch_filter: set[int] | None = None,
        layer_filter: set[str] | None = None,
    ) -> None:
        self._backend = backend
        self._format = format
        self._index = index
        self._epoch_filter = epoch_filter
        self._layer_filter = layer_filter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_entries(self) -> list[IndexEntry]:
        """Index entries that pass the current epoch filter."""
        entries = self._index.all()
        if self._epoch_filter is not None:
            entries = [e for e in entries if e.epoch in self._epoch_filter]
        return entries

    # ------------------------------------------------------------------
    # Properties — metadata only, no tensor I/O
    # ------------------------------------------------------------------

    @property
    def epochs(self) -> list[int]:
        """Sorted list of epoch indices present in this collection."""
        return sorted(e.epoch for e in self._active_entries() if e.epoch is not None)

    @property
    def layers(self) -> list[str]:
        """Parameter names available in this collection.

        Reads only the header of the first matching snapshot
        (no tensor data) and applies the layer filter.
        """
        active = self._active_entries()
        if not active:
            return []
        # Use cached layer names from the index — no file I/O at all.
        names = list(active[0].layers)
        if self._layer_filter is not None:
            names = [n for n in names if n in self._layer_filter]
        return names

    # ------------------------------------------------------------------
    # Single-axis access — reads exactly the data requested
    # ------------------------------------------------------------------

    def by_epoch(self, epoch: int) -> dict[str, np.ndarray]:
        """Load all (or filtered) parameters for a single epoch.

        Opens the snapshot file for *epoch* once, reads the requested
        layers, and closes the file.
        """
        if self._epoch_filter is not None and epoch not in self._epoch_filter:
            raise KeyError(
                f"Epoch {epoch} is excluded by the current selection. "
                f"Available: {self.epochs}"
            )
        entry = self._index.get_by_epoch(epoch)
        if entry is None:
            raise KeyError(
                f"No snapshot for epoch {epoch}. Available: {self.epochs}"
            )
        path = self._backend.read_path(entry.file_key)
        return self._format.read(path, layers=self._layer_filter)

    def by_layer(
        self,
        name: str,
        max_workers: int = 8,
    ) -> dict[int, np.ndarray]:
        """Load a single parameter across all (or filtered) epochs.

        Each epoch's file is read independently, so reads are issued in
        parallel via :class:`~concurrent.futures.ThreadPoolExecutor`.
        Only the requested tensor is read from each file.

        Parameters
        ----------
        name:
            Parameter name, e.g. ``"fc1.weight"``.
        max_workers:
            Maximum parallel reader threads.

        Returns
        -------
        dict[int, np.ndarray]
            ``{epoch: array}`` for every epoch in this collection.
        """
        if self._layer_filter is not None and name not in self._layer_filter:
            raise KeyError(
                f"Layer {name!r} is excluded by the current selection. "
                f"Available: {self.layers}"
            )

        entries = self._active_entries()
        if entries and name not in entries[0].layers:
            raise KeyError(
                f"Layer {name!r} not found. Available: {self.layers}"
            )

        def _read_one(entry: IndexEntry) -> tuple[int, np.ndarray]:
            path = self._backend.read_path(entry.file_key)
            tensor = self._format.read(path, layers={name})[name]
            key = entry.epoch if entry.epoch is not None else entry.step
            assert key is not None
            return key, tensor

        n_workers = min(max_workers, len(entries) or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_read_one, entries))

        return dict(results)

    # ------------------------------------------------------------------
    # Standard export — no NI wrapper types in return values
    # ------------------------------------------------------------------

    def to_state_dict(self, epoch: int) -> dict[str, torch.Tensor]:
        """Return a standard PyTorch state dict for *epoch*.

        The returned dict is loadable directly with ``model.load_state_dict()``.
        Includes both parameters and any buffers (e.g. BatchNorm running stats)
        that were captured at snapshot time.  No NI types appear in the return
        value.

        Parameters
        ----------
        epoch:
            Epoch index to load.
        """
        if self._epoch_filter is not None and epoch not in self._epoch_filter:
            raise KeyError(
                f"Epoch {epoch} is excluded by the current selection. "
                f"Available: {self.epochs}"
            )
        entry = self._index.get_by_epoch(epoch)
        if entry is None:
            raise KeyError(
                f"No snapshot for epoch {epoch}. Available: {self.epochs}"
            )
        path = self._backend.read_path(entry.file_key)
        arrays: dict[str, np.ndarray] = self._format.read(path, layers=self._layer_filter)

        if entry.buffers:
            buf_names = (
                set(entry.buffers) & self._layer_filter
                if self._layer_filter is not None
                else None
            )
            arrays = {**arrays, **self._format.read_buffers(path, names=buf_names)}

        return {name: torch.from_numpy(arr.copy()) for name, arr in arrays.items()}

    def to_numpy(
        self,
        epoch: int,
        layers: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return parameter tensors as plain NumPy arrays for *epoch*.

        No NI types appear in the return value.

        Parameters
        ----------
        epoch:
            Epoch index to load.
        layers:
            Optional list of layer names to include.  ``None`` returns all
            layers (subject to the collection's existing filter).
        """
        col = self.select(layers=layers) if layers is not None else self
        return col.by_epoch(epoch)

    # ------------------------------------------------------------------
    # Filter composition — zero I/O
    # ------------------------------------------------------------------

    def select(
        self,
        epochs: int | list[int] | range | None = None,
        layers: str | list[str] | None = None,
    ) -> SnapshotCollection:
        """Narrow to *epochs* and/or *layers*; return a new :class:`SnapshotCollection`.

        Composes with existing filters — calling ``select`` twice is safe
        and equivalent to calling it once with the intersection.  No files
        are opened.

        Parameters
        ----------
        epochs:
            A single epoch index, a list of indices, or a :class:`range`.
        layers:
            A single parameter name or a list of names.
        """
        new_epoch_filter = self._epoch_filter
        if epochs is not None:
            incoming: set[int] = {epochs} if isinstance(epochs, int) else set(epochs)
            new_epoch_filter = (
                incoming if new_epoch_filter is None else new_epoch_filter & incoming
            )

        new_layer_filter = self._layer_filter
        if layers is not None:
            incoming_l: set[str] = (
                {layers} if isinstance(layers, str) else set(layers)
            )
            new_layer_filter = (
                incoming_l
                if new_layer_filter is None
                else new_layer_filter & incoming_l
            )

        return SnapshotCollection(
            self._backend,
            self._format,
            self._index,
            new_epoch_filter,
            new_layer_filter,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._active_entries())

    def __repr__(self) -> str:
        return (
            f"SnapshotCollection("
            f"snapshots={len(self)}, "
            f"epochs={self.epochs}, "
            f"backend={type(self._backend).__name__}, "
            f"format={type(self._format).__name__})"
        )
