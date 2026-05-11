"""Tests for HDF5Format and NeuroInquisitor with format="hdf5"."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroinquisitor import HDF5Format, NeuroInquisitor, SnapshotCollection
from neuroinquisitor.formats.hdf5_format import _GZIP_LEVEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture()
def obs(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    observer = NeuroInquisitor(mlp, log_dir=tmp_path, format="hdf5")
    yield observer
    if not observer._closed:
        observer.close()


@pytest.fixture()
def obs_compressed(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    observer = NeuroInquisitor(mlp, log_dir=tmp_path, format="hdf5", compress=True)
    yield observer
    if not observer._closed:
        observer.close()


# ---------------------------------------------------------------------------
# HDF5Format unit tests (format in isolation)
# ---------------------------------------------------------------------------


class TestHDF5FormatUnit:
    def _arrays(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {
            "fc1.weight": rng.standard_normal((8, 4)).astype(np.float32),
            "fc1.bias": rng.standard_normal((8,)).astype(np.float32),
        }

    def test_extension(self) -> None:
        assert HDF5Format().extension == ".h5"

    def test_write_returns_bytes(self) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {})
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_write_produces_valid_hdf5(self) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {})
        with h5py.File(io.BytesIO(data), "r") as f:
            assert set(f.keys()) == {"fc1.weight", "fc1.bias"}

    def test_round_trip_values(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        data = fmt.write(arrays, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        for name, arr in arrays.items():
            np.testing.assert_array_equal(loaded[name], arr)

    def test_round_trip_dtype_preserved(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        data = fmt.write(arrays, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        for name, arr in arrays.items():
            assert loaded[name].dtype == arr.dtype

    def test_round_trip_shape_preserved(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        data = fmt.write(arrays, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        for name, arr in arrays.items():
            assert loaded[name].shape == arr.shape

    def test_metadata_stored_as_attributes(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        meta = {"epoch": 3, "loss": 0.42}
        data = fmt.write(self._arrays(), meta)
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert f.attrs["epoch"] == "3"
            assert f.attrs["loss"] == "0.42"

    def test_metadata_values_coerced_to_str(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {"x": 1, "y": True})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert isinstance(f.attrs["x"], str)
            assert isinstance(f.attrs["y"], str)

    def test_empty_metadata(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert len(dict(f.attrs)) == 0

    def test_selective_read(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path, layers={"fc1.weight"})
        assert set(loaded.keys()) == {"fc1.weight"}

    def test_selective_read_returns_correct_values(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        data = fmt.write(arrays, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path, layers={"fc1.bias"})
        np.testing.assert_array_equal(loaded["fc1.bias"], arrays["fc1.bias"])

    def test_list_layers(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        assert set(fmt.list_layers(path)) == {"fc1.weight", "fc1.bias"}

    # --- compression ---

    def test_compress_false_no_compression(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {}, compress=False)
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert f["fc1.weight"].compression is None

    def test_compress_true_uses_gzip(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {}, compress=True)
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert f["fc1.weight"].compression == "gzip"

    def test_compress_true_uses_correct_level(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write(self._arrays(), {}, compress=True)
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        with h5py.File(str(path), "r") as f:
            assert f["fc1.weight"].compression_opts == _GZIP_LEVEL

    def test_compressed_round_trip_values(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        data = fmt.write(arrays, {}, compress=True)
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        for name, arr in arrays.items():
            np.testing.assert_array_equal(loaded[name], arr)

    def test_compressed_file_is_smaller_for_large_arrays(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        rng = np.random.default_rng(42)
        # Large arrays of zeros compress very well
        arrays = {"w": np.zeros((256, 256), dtype=np.float32)}
        uncompressed = fmt.write(arrays, {}, compress=False)
        compressed = fmt.write(arrays, {}, compress=True)
        assert len(compressed) < len(uncompressed)

    def test_empty_params(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        data = fmt.write({}, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        assert loaded == {}

    def test_fortran_order_array_stored_contiguously(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arr = np.asfortranarray(np.ones((4, 4), dtype=np.float32))
        data = fmt.write({"w": arr}, {})
        path = tmp_path / "snap.h5"
        path.write_bytes(data)
        loaded = fmt.read(path)
        np.testing.assert_array_equal(loaded["w"], arr)

    # --- write_to_path (NI-LLM-003) ---

    def test_write_to_path_round_trip(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        dest = tmp_path / "snap.h5"
        fmt.write_to_path(dest, arrays, {})
        loaded = fmt.read(dest)
        for name, arr in arrays.items():
            np.testing.assert_array_equal(loaded[name], arr)

    def test_write_to_path_numerically_identical_to_write(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        meta = {"epoch": 1}

        via_bytes = tmp_path / "via_bytes.h5"
        via_bytes.write_bytes(fmt.write(arrays, meta))

        via_path = tmp_path / "via_path.h5"
        fmt.write_to_path(via_path, arrays, meta)

        loaded_bytes = fmt.read(via_bytes)
        loaded_path = fmt.read(via_path)
        for name in arrays:
            np.testing.assert_array_equal(loaded_path[name], loaded_bytes[name])

    def test_write_to_path_creates_file(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        dest = tmp_path / "snap.h5"
        assert not dest.exists()
        fmt.write_to_path(dest, self._arrays(), {})
        assert dest.exists()

    def test_write_to_path_with_compression(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        arrays = self._arrays()
        dest = tmp_path / "snap.h5"
        fmt.write_to_path(dest, arrays, {}, compress=True)
        with h5py.File(str(dest), "r") as f:
            assert f["fc1.weight"].compression == "gzip"

    def test_write_to_path_with_buffers(self, tmp_path: Path) -> None:
        fmt = HDF5Format()
        rng = np.random.default_rng(1)
        buffers = {"bn.running_mean": rng.standard_normal((8,)).astype(np.float32)}
        dest = tmp_path / "snap.h5"
        fmt.write_to_path(dest, self._arrays(), {}, buffers=buffers)
        loaded_bufs = fmt.read_buffers(dest)
        np.testing.assert_array_equal(loaded_bufs["bn.running_mean"], buffers["bn.running_mean"])

    def test_base_format_write_to_path_fallback(self, tmp_path: Path) -> None:
        """A Format subclass that does not override write_to_path falls back to write()."""
        from neuroinquisitor.formats.base import Format

        class _StubFormat(Format):
            @property
            def extension(self) -> str:
                return ".h5"

            def write(self, params, metadata, compress=False, buffers=None) -> bytes:
                return HDF5Format().write(params, metadata, compress=compress, buffers=buffers)

            def read(self, path, layers=None):
                return HDF5Format().read(path, layers=layers)

            def list_layers(self, path):
                return HDF5Format().list_layers(path)

        stub = _StubFormat()
        arrays = self._arrays()
        dest = tmp_path / "stub.h5"
        stub.write_to_path(dest, arrays, {})  # must not raise
        loaded = stub.read(dest)
        for name, arr in arrays.items():
            np.testing.assert_array_equal(loaded[name], arr)


# ---------------------------------------------------------------------------
# NeuroInquisitor integration with format="hdf5"
# ---------------------------------------------------------------------------


class TestNeuroInquisitorHDF5:
    def test_h5_file_created_on_snapshot(
        self, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        assert (tmp_path / "epoch_0000.h5").exists()

    def test_index_json_exists(self, obs: NeuroInquisitor, tmp_path: Path) -> None:
        obs.snapshot(epoch=0)
        assert (tmp_path / "index.json").exists()

    def test_multiple_h5_files_created(
        self, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        obs.snapshot(epoch=1)
        assert (tmp_path / "epoch_0000.h5").exists()
        assert (tmp_path / "epoch_0001.h5").exists()

    def test_load_snapshot_values_match(
        self, mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        original = {
            name: param.detach().cpu().numpy().copy()
            for name, param in mlp.named_parameters()
        }
        obs.snapshot(epoch=0)
        loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
        for name, arr in original.items():
            np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)

    def test_load_snapshot_shapes(
        self, mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
        for name, param in mlp.named_parameters():
            assert loaded[name].shape == tuple(param.shape)

    def test_load_snapshot_returns_numpy_arrays(
        self, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
        for arr in loaded.values():
            assert isinstance(arr, np.ndarray)

    def test_duplicate_epoch_raises(self, obs: NeuroInquisitor) -> None:
        obs.snapshot(epoch=0)
        with pytest.raises(ValueError, match="epoch_0000"):
            obs.snapshot(epoch=0)

    def test_step_only_creates_h5_file(
        self, obs: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs.snapshot(step=10)
        assert (tmp_path / "step_000010.h5").exists()

    def test_compressed_h5_file_created(
        self, obs_compressed: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs_compressed.snapshot(epoch=0)
        assert (tmp_path / "epoch_0000.h5").exists()

    def test_compressed_values_match(
        self, mlp: nn.Module, obs_compressed: NeuroInquisitor, tmp_path: Path
    ) -> None:
        original = {
            name: param.detach().cpu().numpy().copy()
            for name, param in mlp.named_parameters()
        }
        obs_compressed.snapshot(epoch=0)
        loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
        for name, arr in original.items():
            np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)

    def test_compressed_file_uses_gzip(
        self, obs_compressed: NeuroInquisitor, tmp_path: Path
    ) -> None:
        obs_compressed.snapshot(epoch=0)
        path = tmp_path / "epoch_0000.h5"
        with h5py.File(str(path), "r") as f:
            first_key = next(iter(f.keys()))
            assert f[first_key].compression == "gzip"

    def test_format_instance_accepted(self, mlp: nn.Module, tmp_path: Path) -> None:
        obs = NeuroInquisitor(mlp, log_dir=tmp_path, format=HDF5Format())
        obs.snapshot(epoch=0)
        obs.close()
        assert (tmp_path / "epoch_0000.h5").exists()

    def test_load_returns_collection(
        self, obs: NeuroInquisitor, mlp: nn.Module, tmp_path: Path
    ) -> None:
        for epoch in range(3):
            with torch.no_grad():
                for p in mlp.parameters():
                    p.add_(0.1)
            obs.snapshot(epoch=epoch)
        col = NeuroInquisitor.load(tmp_path)
        assert isinstance(col, SnapshotCollection)
        assert len(col) == 3

    def test_collection_by_epoch(
        self, obs: NeuroInquisitor, mlp: nn.Module, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        col = NeuroInquisitor.load(tmp_path)
        params = col.by_epoch(0)
        assert set(params.keys()) == {name for name, _ in mlp.named_parameters()}

    def test_collection_by_layer(
        self, obs: NeuroInquisitor, mlp: nn.Module, tmp_path: Path
    ) -> None:
        for epoch in range(3):
            obs.snapshot(epoch=epoch)
        col = NeuroInquisitor.load(tmp_path)
        result = col.by_layer("0.weight")
        assert set(result.keys()) == {0, 1, 2}

    def test_neuro_inquisitor_load_classmethod(
        self, obs: NeuroInquisitor, mlp: nn.Module, tmp_path: Path
    ) -> None:
        obs.snapshot(epoch=0)
        obs.close()
        col = NeuroInquisitor.load(tmp_path, format="hdf5")
        assert isinstance(col, SnapshotCollection)
        assert col.epochs == [0]

    def test_append_mode_hdf5(self, mlp: nn.Module, tmp_path: Path) -> None:
        obs1 = NeuroInquisitor(mlp, log_dir=tmp_path, format="hdf5", create_new=True)
        obs1.snapshot(epoch=0)
        obs1.close()

        obs2 = NeuroInquisitor(mlp, log_dir=tmp_path, format="hdf5", create_new=False)
        obs2.snapshot(epoch=1)
        obs2.close()

        col = NeuroInquisitor.load(tmp_path, format="hdf5")
        assert col.epochs == [0, 1]
