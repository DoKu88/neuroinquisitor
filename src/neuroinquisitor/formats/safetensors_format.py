"""SafeTensorsFormat — one .safetensors file per snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save as _st_save

from .base import Format


class SafeTensorsFormat(Format):
    """Serialises snapshots using the SafeTensors format.

    Each snapshot is a self-contained ``.safetensors`` file.  The format
    supports selective tensor loading via :func:`safe_open`, so
    :meth:`read` with *layers* specified performs only the I/O necessary
    for the requested parameters.

    Notes
    -----
    SafeTensors does not support compression.  The *compress* parameter on
    :class:`~neuroinquisitor.core.NeuroInquisitor` is accepted for API
    consistency but has no effect with this format.
    """

    @property
    def extension(self) -> str:
        return ".safetensors"

    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
    ) -> bytes:
        # SafeTensors metadata values must be strings.
        str_meta: dict[str, str] = {k: str(v) for k, v in metadata.items()}
        # Arrays must be C-contiguous.
        contiguous = {k: np.ascontiguousarray(v) for k, v in params.items()}
        return _st_save(contiguous, metadata=str_meta or None)

    def read(
        self,
        path: Path,
        layers: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys()) if layers is None else [k for k in f.keys() if k in layers]
            return {k: f.get_tensor(k) for k in keys}

    def list_layers(self, path: Path) -> list[str]:
        with safe_open(str(path), framework="numpy") as f:
            return list(f.keys())
