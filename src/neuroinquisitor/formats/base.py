"""Format ABC — snapshot file format abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Format(ABC):
    """Abstracts how parameter tensors are serialised to/from bytes.

    Implementors
    ------------
    :class:`~neuroinquisitor.formats.hdf5_format.HDF5Format`
    """

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension including the leading dot, e.g. ``".h5"``."""

    @abstractmethod
    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,
        buffers: dict[str, np.ndarray] | None = None,
    ) -> bytes:
        """Serialise *params* and *metadata* to a byte string.

        If *buffers* is provided, implementations should store the buffer
        tensors under a namespace that is distinguishable from parameters.
        Callers that do not need buffers omit the argument.
        """

    @abstractmethod
    def read(
        self,
        path: Path,
        layers: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Deserialise tensors from *path*.

        If *layers* is given, only those parameter names are loaded —
        implementations should avoid reading the rest from disk.
        """

    def read_buffers(
        self,
        path: Path,
        names: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read buffer tensors stored in *path*.

        Implementations that store buffers separately (e.g. under a
        ``"buffers/"`` group) should override this.  The default returns an
        empty dict so formats that do not support buffers remain unchanged.
        """
        return {}

    @abstractmethod
    def list_layers(self, path: Path) -> list[str]:
        """Return all parameter names stored in *path* without reading tensors."""
