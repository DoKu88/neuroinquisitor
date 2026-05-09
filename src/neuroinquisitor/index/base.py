"""Index ABC + IndexEntry — snapshot metadata catalog abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroinquisitor.backends.base import Backend
    from neuroinquisitor.schema import CapturePolicy


@dataclass
class IndexEntry:
    """Lightweight metadata record for one snapshot.

    No weight data is stored here — only the information needed to
    locate and filter snapshots without opening any tensor file.
    """

    epoch: int | None = None
    step: int | None = None
    file_key: str = ""
    layers: list[str] = field(default_factory=list)
    buffers: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    capture_policy: CapturePolicy | None = None


class Index(ABC):
    """Abstracts the snapshot metadata catalog.

    The index is the only thing :class:`~neuroinquisitor.collection.SnapshotCollection`
    reads when evaluating filters — no tensor files are opened until the
    caller requests actual weight data.

    Implementors
    ------------
    :class:`~neuroinquisitor.index.json_index.JSONIndex`
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @abstractmethod
    def add(self, entry: IndexEntry) -> None:
        """Add *entry* and persist the updated index immediately."""

    @abstractmethod
    def all(self) -> list[IndexEntry]:
        """Return all entries in insertion order."""

    @abstractmethod
    def get_by_epoch(self, epoch: int) -> IndexEntry | None:
        """Return the entry for *epoch*, or ``None`` if not found."""

    @abstractmethod
    def contains_key(self, file_key: str) -> bool:
        """Return ``True`` if a snapshot with *file_key* is already recorded."""

    @abstractmethod
    def save(self) -> None:
        """Persist the current state of the index via the backend."""

    @classmethod
    @abstractmethod
    def load(cls, backend: Backend) -> Index:
        """Construct an index by reading from *backend*."""
