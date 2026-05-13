"""Tests for SafetensorsFormat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

safetensors = pytest.importorskip("safetensors")
import torch  # noqa: E402

from neuroinquisitor.formats.safetensors_format import (  # noqa: E402
    SafetensorsFormat,
    _BUFFER_PREFIX,
)


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "fc1.weight": rng.standard_normal((8, 4)).astype(np.float32),
        "fc1.bias": rng.standard_normal((8,)).astype(np.float32),
    }


def test_extension() -> None:
    assert SafetensorsFormat().extension == ".safetensors"


def test_write_to_path_round_trip(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    arrays = _arrays()
    dest = tmp_path / "snap.safetensors"
    fmt.write_to_path(dest, arrays, {"epoch": 1})
    loaded = fmt.read(dest)
    assert set(loaded.keys()) == set(arrays.keys())
    for name, arr in arrays.items():
        np.testing.assert_array_equal(loaded[name], arr)


def test_write_bytes_round_trip(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    arrays = _arrays()
    data = fmt.write(arrays, {})
    path = tmp_path / "snap.safetensors"
    path.write_bytes(data)
    loaded = fmt.read(path)
    for name, arr in arrays.items():
        np.testing.assert_array_equal(loaded[name], arr)


def test_selective_read_only_returns_requested(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    arrays = _arrays()
    dest = tmp_path / "snap.safetensors"
    fmt.write_to_path(dest, arrays, {})
    loaded = fmt.read(dest, layers={"fc1.weight"})
    assert set(loaded.keys()) == {"fc1.weight"}
    np.testing.assert_array_equal(loaded["fc1.weight"], arrays["fc1.weight"])


def test_list_layers_excludes_buffers(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    arrays = _arrays()
    buffers = {"running_mean": np.zeros(8, dtype=np.float32)}
    dest = tmp_path / "snap.safetensors"
    fmt.write_to_path(dest, arrays, {}, buffers=buffers)
    layers = set(fmt.list_layers(dest))
    assert layers == set(arrays.keys())


def test_read_buffers(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    arrays = _arrays()
    buffers = {"bn.running_mean": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    dest = tmp_path / "snap.safetensors"
    fmt.write_to_path(dest, arrays, {}, buffers=buffers)
    loaded_bufs = fmt.read_buffers(dest)
    np.testing.assert_array_equal(loaded_bufs["bn.running_mean"], buffers["bn.running_mean"])


def test_write_tensors_to_path_bfloat16_round_trip(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    t = torch.randn(4, 4).to(torch.bfloat16)
    dest = tmp_path / "snap.safetensors"
    fmt.write_tensors_to_path(dest, {"w": t}, {"epoch": 0})

    # Read back through safetensors.torch to verify dtype preservation.
    from safetensors import safe_open

    with safe_open(str(dest), framework="pt") as f:
        out = f.get_tensor("w")
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, t)


def test_write_tensors_to_path_with_buffers(tmp_path: Path) -> None:
    fmt = SafetensorsFormat()
    t = torch.randn(4, 4)
    buf = torch.ones(4)
    dest = tmp_path / "snap.safetensors"
    fmt.write_tensors_to_path(dest, {"w": t}, {}, buffers={"running_mean": buf})

    layers = fmt.list_layers(dest)
    assert layers == ["w"]
    bufs = fmt.read_buffers(dest)
    assert set(bufs.keys()) == {"running_mean"}


def test_buffer_prefix_kept_internal(tmp_path: Path) -> None:
    """Buffer keys should be stored under the internal prefix but never surface from read()."""
    fmt = SafetensorsFormat()
    buffers = {"x": np.zeros(2, dtype=np.float32)}
    dest = tmp_path / "snap.safetensors"
    fmt.write_to_path(dest, {}, {}, buffers=buffers)

    from safetensors import safe_open

    with safe_open(str(dest), framework="np") as f:
        all_keys = list(f.keys())
    assert all_keys == [f"{_BUFFER_PREFIX}x"]
    assert fmt.read(dest) == {}


def test_import_error_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import neuroinquisitor.formats.safetensors_format as mod

    monkeypatch.setattr(mod, "_stnp", None)
    fmt = SafetensorsFormat()
    with pytest.raises(ImportError, match="neuroinquisitor\\[safetensors\\]"):
        fmt.write_to_path(tmp_path / "snap.safetensors", _arrays(), {})
