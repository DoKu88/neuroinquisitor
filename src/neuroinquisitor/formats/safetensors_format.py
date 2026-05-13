"""SafetensorsFormat — fast, memory-mappable weight storage with bfloat16 support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from safetensors import numpy as _stnp
from safetensors import safe_open
from safetensors import torch as _sttorch

from .base import Format

if TYPE_CHECKING:
    import torch

_BUFFER_PREFIX = "__buffers__/"


def _stringify_metadata(metadata: dict[str, object]) -> dict[str, str]:
    return {k: str(v) for k, v in metadata.items()}


def _merge_buffers(
    params: dict[str, np.ndarray],
    buffers: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if not buffers:
        return dict(params)
    merged: dict[str, np.ndarray] = dict(params)
    for name, arr in buffers.items():
        merged[f"{_BUFFER_PREFIX}{name}"] = arr
    return merged


class SafetensorsFormat(Format):
    """Serialises snapshots using safetensors.

    Each snapshot is a self-contained ``.safetensors`` file.  Reads use the
    memory-mapped ``safe_open`` API so selective layer loading reads only the
    requested layer's bytes from disk.

    Buffers are stored under an ``__buffers__/`` name prefix to keep them
    distinguishable from parameters without inspecting metadata.

    Parameters and buffers are written natively as bfloat16 when supplied via
    :meth:`write_tensors_to_path` — there is no numpy round-trip.
    """

    @property
    def extension(self) -> str:
        return ".safetensors"

    # ------------------------------------------------------------------
    # Numpy write paths
    # ------------------------------------------------------------------

    def write(
        self,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,  # noqa: ARG002 - safetensors does not compress
        buffers: dict[str, np.ndarray] | None = None,
    ) -> bytes:
        merged = _merge_buffers(params, buffers)
        return _stnp.save(merged, metadata=_stringify_metadata(metadata))

    def write_to_path(
        self,
        dest: Path,
        params: dict[str, np.ndarray],
        metadata: dict[str, object],
        compress: bool = False,  # noqa: ARG002
        buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        merged = _merge_buffers(params, buffers)
        _stnp.save_file(merged, str(dest), metadata=_stringify_metadata(metadata))

    # ------------------------------------------------------------------
    # Torch write path (bfloat16-safe)
    # ------------------------------------------------------------------

    def write_tensors_to_path(
        self,
        dest: Path,
        tensors: dict[str, "torch.Tensor"],
        metadata: dict[str, object],
        buffers: dict[str, "torch.Tensor"] | None = None,
    ) -> None:
        """Write *tensors* (and optional torch *buffers*) directly to *dest*.

        Uses ``safetensors.torch.save_file`` so dtypes like ``bfloat16`` are
        preserved without a numpy conversion step.
        """
        merged: dict[str, torch.Tensor] = dict(tensors)
        if buffers:
            for name, t in buffers.items():
                merged[f"{_BUFFER_PREFIX}{name}"] = t
        _sttorch.save_file(merged, str(dest), metadata=_stringify_metadata(metadata))

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def read(
        self,
        path: Path,
        layers: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="np") as f:
            for key in f.keys():
                if key.startswith(_BUFFER_PREFIX):
                    continue
                if layers is not None and key not in layers:
                    continue
                out[key] = f.get_tensor(key)
        return out

    def read_buffers(
        self,
        path: Path,
        names: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="np") as f:
            for key in f.keys():
                if not key.startswith(_BUFFER_PREFIX):
                    continue
                short = key[len(_BUFFER_PREFIX):]
                if names is not None and short not in names:
                    continue
                out[short] = f.get_tensor(key)
        return out

    def list_layers(self, path: Path) -> list[str]:
        with safe_open(str(path), framework="np") as f:
            return [k for k in f.keys() if not k.startswith(_BUFFER_PREFIX)]
