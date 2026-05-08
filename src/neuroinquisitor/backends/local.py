"""LocalBackend — plain filesystem storage."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Backend


class LocalBackend(Backend):
    """Stores snapshot files in a directory on the local filesystem."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def write(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_path(self, key: str) -> Path:
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(
                f"Key {key!r} not found under {self._root}"
            )
        return path

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def delete(self, key: str) -> None:
        (self._root / key).unlink(missing_ok=True)

    def __repr__(self) -> str:
        return f"LocalBackend(root={self._root!r})"
