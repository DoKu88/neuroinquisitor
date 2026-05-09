"""JSONIndex — index.json backed catalog."""

from __future__ import annotations

import json

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.schema import (
    CapturePolicy,
    RunManifest,
    RunMetadata,
    SnapshotRef,
)

from .base import Index, IndexEntry

_KEY = "index.json"


def _entry_to_ref(entry: IndexEntry) -> SnapshotRef:
    return SnapshotRef(
        epoch=entry.epoch,
        step=entry.step,
        file_key=entry.file_key,
        layers=entry.layers,
        buffers=entry.buffers,
        metadata=entry.metadata,
        capture_policy=entry.capture_policy,
    )


def _ref_to_entry(ref: SnapshotRef) -> IndexEntry:
    return IndexEntry(
        epoch=ref.epoch,
        step=ref.step,
        file_key=ref.file_key,
        layers=ref.layers,
        buffers=ref.buffers,
        metadata=ref.metadata,
        capture_policy=ref.capture_policy,
    )


class JSONIndex(Index):
    """Stores the snapshot catalog as ``index.json`` via the backend.

    The file is rewritten on every :meth:`add` so the catalog is always
    up to date on disk.  The on-disk format is a :class:`~neuroinquisitor.schema.RunManifest`
    serialised as JSON.  Future implementors may swap this for a
    :class:`SQLiteIndex` or a remote SQL catalog without changing anything
    above the :class:`~neuroinquisitor.index.base.Index` interface.
    """

    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        self._manifest: RunManifest = RunManifest()
        self._entries: list[IndexEntry] = []

    # ------------------------------------------------------------------
    # Manifest-level helpers
    # ------------------------------------------------------------------

    def set_run_metadata(self, metadata: RunMetadata) -> None:
        """Attach run-level provenance metadata (call before first save)."""
        self._manifest = self._manifest.model_copy(update={"run_metadata": metadata})

    def set_capture_policy(self, policy: CapturePolicy) -> None:
        """Attach the run-level capture policy (call before first save)."""
        self._manifest = self._manifest.model_copy(update={"capture_policy": policy})

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
        manifest = self._manifest.model_copy(
            update={"snapshots": [_entry_to_ref(e) for e in self._entries]}
        )
        self._backend.write(_KEY, manifest.model_dump_json(indent=2).encode())

    @classmethod
    def load(cls, backend: Backend) -> JSONIndex:
        instance = cls(backend)
        if not backend.exists(_KEY):
            return instance
        raw = json.loads(backend.read_path(_KEY).read_bytes())
        manifest = RunManifest.model_validate(raw)
        instance._manifest = manifest.model_copy(update={"snapshots": []})
        instance._entries = [_ref_to_entry(ref) for ref in manifest.snapshots]
        return instance
