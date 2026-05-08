"""Format ABC — snapshot file format abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Format(ABC):
    """Abstracts how parameter tensors are serialised to/from bytes.

    Implementors
    ------------
    :class:`~neuroinquisitor.formats.safetensors.SafeTensorsFormat`
    """

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension including the leading dot, e.g. ``".safetensors"``."""

    @abstractmethod
    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
    ) -> bytes:
        """Serialise *params* and *metadata* to a byte string."""

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

    @abstractmethod
    def list_layers(self, path: Path) -> list[str]:
        """Return all parameter names stored in *path* without reading tensors."""
