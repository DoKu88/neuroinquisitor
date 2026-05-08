"""JSONIndex — index.json backed catalog."""

from __future__ import annotations

import json

from neuroinquisitor.backends.base import Backend

from .base import Index, IndexEntry

_KEY = "index.json"


class JSONIndex(Index):
    """Stores the snapshot catalog as ``index.json`` via the backend.

    The file is rewritten on every :meth:`add` so the catalog is always
    up to date on disk.  Future implementors may swap this for a
    :class:`SQLiteIndex` or a remote SQL catalog without changing
    anything above the :class:`~neuroinquisitor.index.base.Index` interface.
    """

    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        self._entries: list[IndexEntry] = []

    # ------------------------------------------------------------------
    # Index interface
    # ------------------------------------------------------------------

    def add(self, entry: IndexEntry) -> None:
        self._entries.append(entry)
        self.save()

    def all(self) -> list[IndexEntry]:
        return list(self._entries)

    def get_by_epoch(self, epoch: int) -> IndexEntry | None:
        for entry in self._entries:
            if entry.epoch == epoch:
                return entry
        return None

    def contains_key(self, file_key: str) -> bool:
        return any(e.file_key == file_key for e in self._entries)

    def save(self) -> None:
        payload = {
            "snapshots": [
                {
                    "epoch": e.epoch,
                    "step": e.step,
                    "file_key": e.file_key,
                    "layers": e.layers,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ]
        }
        self._backend.write(_KEY, json.dumps(payload, indent=2).encode())

    @classmethod
    def load(cls, backend: Backend) -> JSONIndex:
        instance = cls(backend)
        if not backend.exists(_KEY):
            return instance
        raw = backend.read_path(_KEY).read_bytes()
        data = json.loads(raw)
        instance._entries = [
            IndexEntry(
                epoch=s["epoch"],
                step=s["step"],
                file_key=s["file_key"],
                layers=s.get("layers", []),
                metadata=s.get("metadata", {}),
            )
            for s in data["snapshots"]
        ]
        return instance
