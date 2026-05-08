"""Backend ABC — storage location abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    """Abstracts where snapshot files are stored (local filesystem, S3, …).

    Each snapshot is stored under an opaque *key* string (e.g.
    ``"epoch_0000.h5"``).  The backend is responsible for
    translating keys to actual storage locations.

    Implementors
    ------------
    :class:`~neuroinquisitor.backends.local.LocalBackend`
    """

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Persist *data* under *key*, overwriting any existing content."""

    @abstractmethod
    def read_path(self, key: str) -> Path:
        """Return a local :class:`~pathlib.Path` for *key*.

        For remote backends this may download the file to a temporary
        location first.  Raises :exc:`FileNotFoundError` if the key does
        not exist.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists in this backend."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* from this backend.  No-op if it does not exist."""
