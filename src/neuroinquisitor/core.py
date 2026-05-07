"""Core NeuroInquisitor class."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


class NeuroInquisitor:
    """Attach to a PyTorch nn.Module to snapshot weights during training.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose parameters are snapshotted.
    log_dir:
        Directory where the HDF5 file is written.
    filename:
        HDF5 filename (e.g. ``"weights.h5"``).
    compress:
        Store datasets with gzip compression when ``True``.
    create_new:
        ``True``  → create a fresh file; raises ``FileExistsError`` if one
        already exists.
        ``False`` → open an existing file in append mode; raises
        ``FileNotFoundError`` if the file is absent.
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: str | os.PathLike[str] = ".",
        filename: str = "weights.h5",
        compress: bool = False,
        create_new: bool = True,
    ) -> None:
        self._model = model
        self._compress = compress
        self._closed = False

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self._filepath = log_path / filename

        if create_new:
            if self._filepath.exists():
                raise FileExistsError(
                    f"HDF5 file already exists: {self._filepath}. "
                    "Pass create_new=False to append to it, or delete it first."
                )
            mode = "w"
        else:
            if not self._filepath.exists():
                raise FileNotFoundError(
                    f"HDF5 file not found: {self._filepath}. "
                    "Pass create_new=True to create a new file."
                )
            mode = "a"

        self._file: h5py.File = h5py.File(self._filepath, mode)
        logger.info("NeuroInquisitor opened %s (mode=%r)", self._filepath, mode)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _group_key(self, epoch: int | None, step: int | None) -> str:
        parts: list[str] = []
        if epoch is not None:
            parts.append(f"epoch_{epoch:04d}")
        if step is not None:
            parts.append(f"step_{step:06d}")
        return "_".join(parts)

    def snapshot(
        self,
        epoch: int | None = None,
        step: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Snapshot all model parameters to HDF5 and flush to disk."""
        if self._closed:
            raise RuntimeError("Cannot snapshot: NeuroInquisitor is closed.")
        if epoch is None and step is None:
            raise ValueError("At least one of epoch or step must be provided.")

        # Validate everything before touching the file.
        if metadata is not None:
            reserved = frozenset({"epoch", "step"}) & metadata.keys()
            if reserved:
                raise ValueError(
                    f"Metadata contains reserved key(s) {sorted(reserved)}. "
                    "Choose different key names to avoid collision with "
                    "built-in snapshot attributes."
                )

        group_key = self._group_key(epoch, step)
        if group_key in self._file:
            raise ValueError(
                f"Snapshot {group_key!r} already exists in {self._filepath}. "
                "Each (epoch, step) combination must be unique."
            )

        grp = self._file.create_group(group_key)

        if epoch is not None:
            grp.attrs["epoch"] = epoch
        if step is not None:
            grp.attrs["step"] = step
        if metadata is not None:
            for key, value in metadata.items():
                grp.attrs[key] = value

        dataset_kwargs: dict[str, object] = {}
        if self._compress:
            dataset_kwargs["compression"] = "gzip"
            dataset_kwargs["chunks"] = True

        for name, param in self._model.named_parameters():
            data = param.detach().cpu().numpy()
            grp.create_dataset(name, data=data, **dataset_kwargs)

        self._file.flush()
        logger.debug("Snapshot written: %s", group_key)

    def load_snapshot(self, epoch: int) -> dict[str, np.ndarray]:
        """Load a snapshot by epoch from the HDF5 file."""
        group_key = f"epoch_{epoch:04d}"
        with h5py.File(self._filepath, "r") as f:
            if group_key not in f:
                raise KeyError(
                    f"No snapshot found for epoch {epoch} ({group_key!r}) "
                    f"in {self._filepath}."
                )
            grp = f[group_key]
            return {name: grp[name][()] for name in grp}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalize and close the HDF5 file. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_file"):
            self._file.close()
        logger.info("NeuroInquisitor closed %s", self._filepath)

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            import warnings

            warnings.warn(
                f"NeuroInquisitor for {self._filepath!r} was not explicitly closed. "
                "Call .close() when done to avoid resource leaks.",
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
            f"file={self._filepath!r}, "
            f"compress={self._compress}, "
            f"status={status!r})"
        )
