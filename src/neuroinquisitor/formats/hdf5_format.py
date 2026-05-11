"""HDF5Format — one .h5 file per snapshot, with optional gzip compression."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np

from .base import Format

_GZIP_LEVEL = 4
_BUFFERS_GROUP = "buffers"


class HDF5Format(Format):
    """Serialises snapshots using HDF5 via h5py.

    Each snapshot is a self-contained ``.h5`` file.  When *compress* is
    ``True``, each dataset is written with gzip compression (level
    ``_GZIP_LEVEL``).  Selective layer loading is supported via dataset-level
    reads — only the requested tensors are decompressed.

    Parameters are stored as top-level HDF5 datasets.  When buffer capture
    is active, buffer tensors are stored under the ``"buffers/"`` group so
    they are always distinguishable from parameters without reading metadata.

    Scalar metadata is stored as HDF5 file-level attributes (coerced to strings).
    """

    @property
    def extension(self) -> str:
        return ".h5"

    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,
        buffers: dict[str, np.ndarray] | None = None,
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
            if buffers:
                grp = f.require_group(_BUFFERS_GROUP)
                for name, array in buffers.items():
                    grp.create_dataset(
                        name,
                        data=np.ascontiguousarray(array),
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            for k, v in metadata.items():
                f.attrs[k] = str(v)

        return buf.getvalue()

    def write_to_path(
        self,
        dest: Path,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,
        buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Write snapshot directly to *dest* via h5py, with no BytesIO buffer."""
        compression = "gzip" if compress else None
        compression_opts = _GZIP_LEVEL if compress else None

        with h5py.File(str(dest), "w") as f:
            for name, array in params.items():
                f.create_dataset(
                    name,
                    data=np.ascontiguousarray(array),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            if buffers:
                grp = f.require_group(_BUFFERS_GROUP)
                for name, array in buffers.items():
                    grp.create_dataset(
                        name,
                        data=np.ascontiguousarray(array),
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            for k, v in metadata.items():
                f.attrs[k] = str(v)

    def read(
        self,
        path: Path,
        layers: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        with h5py.File(str(path), "r") as f:
            param_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            if layers is not None:
                param_keys = [k for k in param_keys if k in layers]
            return {k: f[k][()] for k in param_keys}

    def read_buffers(
        self,
        path: Path,
        names: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read buffer tensors from the ``"buffers/"`` group of *path*.

        Returns an empty dict if the snapshot has no buffers.
        """
        with h5py.File(str(path), "r") as f:
            if _BUFFERS_GROUP not in f:
                return {}
            grp = f[_BUFFERS_GROUP]
            keys = list(grp.keys()) if names is None else [k for k in grp.keys() if k in names]
            return {k: grp[k][()] for k in keys}

    def list_layers(self, path: Path) -> list[str]:
        with h5py.File(str(path), "r") as f:
            return [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
