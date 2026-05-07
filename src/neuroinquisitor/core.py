"""Core NeuroInquisitor class."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]

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
