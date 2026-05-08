"""HDF5Format — one .h5 file per snapshot, with optional gzip compression."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np

from .base import Format

_GZIP_LEVEL = 4


class HDF5Format(Format):
    """Serialises snapshots using HDF5 via h5py.

    Each snapshot is a self-contained ``.h5`` file.  When *compress* is
    ``True``, each dataset is written with gzip compression (level
    ``_GZIP_LEVEL``).  Selective layer loading is supported via dataset-level
    reads — only the requested tensors are decompressed.

    Metadata is stored as HDF5 file-level attributes.  All values are
    coerced to strings to avoid HDF5 type ambiguity.
    """

    @property
    def extension(self) -> str:
        return ".h5"

    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,
    ) -> bytes:
        buf = io.BytesIO()
        compression = "gzip" if compress else None
        compression_opts = _GZIP_LEVEL if compress else None

        with h5py.File(buf, "w") as f:
            for name, array in params.items():
                f.create_dataset(
                    name,
                    data=np.ascontiguousarray(array),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            for k, v in metadata.items():
                f.attrs[k] = str(v)

        return buf.getvalue()

    def read(
        self,
        path: Path,
        layers: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        with h5py.File(str(path), "r") as f:
            keys = list(f.keys()) if layers is None else [k for k in f.keys() if k in layers]
            return {k: f[k][()] for k in keys}

    def list_layers(self, path: Path) -> list[str]:
        with h5py.File(str(path), "r") as f:
            return list(f.keys())
