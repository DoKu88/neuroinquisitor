"""Tests for HDF5 storage correctness — Sprint 3."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))


@pytest.fixture()
def obs(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    observer = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5")
    yield observer
    if not observer._closed:
        observer.close()


# ---------------------------------------------------------------------------
# Group hierarchy
# ---------------------------------------------------------------------------


def test_group_hierarchy_created_correctly(
    obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    obs.snapshot(epoch=1)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert "epoch_0000" in f
        assert "epoch_0001" in f


def test_parameter_datasets_inside_group(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    obs.close()

    expected = {name for name, _ in mlp.named_parameters()}
    with h5py.File(tmp_path / "weights.h5", "r") as f:
        actual = set(f["epoch_0000"].keys())
    assert actual == expected


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_weights_saved_with_correct_shapes(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name, param in mlp.named_parameters():
            assert grp[name].shape == tuple(param.shape), f"Shape mismatch: {name}"


def test_weights_saved_with_correct_dtype(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name, param in mlp.named_parameters():
            expected_dtype = param.detach().cpu().numpy().dtype
            assert grp[name].dtype == expected_dtype, f"Dtype mismatch: {name}"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def test_compression_gzip_set_when_compress_true(
    mlp: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, compress=True)
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name in grp:
            assert grp[name].compression == "gzip", f"Expected gzip for {name}"


def test_no_compression_when_compress_false(mlp: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, compress=False)
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name in grp:
            assert grp[name].compression is None, f"Expected no compression for {name}"


# ---------------------------------------------------------------------------
# load_snapshot round-trip
# ---------------------------------------------------------------------------


def test_load_snapshot_values_match_original(
    mlp: nn.Module, obs: NeuroInquisitor
) -> None:
    original = {
        name: param.detach().cpu().numpy() for name, param in mlp.named_parameters()
    }
    obs.snapshot(epoch=7)
    loaded = obs.load_snapshot(epoch=7)

    for name, arr in original.items():
        np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)


def test_load_snapshot_raises_for_missing_epoch(obs: NeuroInquisitor) -> None:
    with pytest.raises(KeyError, match="epoch 99"):
        obs.load_snapshot(epoch=99)


# ---------------------------------------------------------------------------
# Multiple snapshots coexist
# ---------------------------------------------------------------------------


def test_multiple_snapshots_coexist(mlp: nn.Module, obs: NeuroInquisitor) -> None:
    for epoch in range(5):
        # mutate weights slightly so values differ
        with torch.no_grad():
            for param in mlp.parameters():
                param.add_(0.01)
        obs.snapshot(epoch=epoch)

    for epoch in range(5):
        loaded = obs.load_snapshot(epoch=epoch)
        assert set(loaded.keys()) == {name for name, _ in mlp.named_parameters()}


# ---------------------------------------------------------------------------
# Crash-recovery: flush guarantees all prior snapshots survive
# ---------------------------------------------------------------------------


def test_crash_recovery_all_snapshots_readable(
    mlp: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5")
    for epoch in range(3):
        obs.snapshot(epoch=epoch)

    # Simulate crash: close the HDF5 file object directly, bypassing observer.close()
    obs._file.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        for epoch in range(3):
            group_key = f"epoch_{epoch:04d}"
            assert group_key in f, f"Missing {group_key} after simulated crash"
            assert len(f[group_key]) > 0


# ---------------------------------------------------------------------------
# Append mode: create_new=False + snapshot
# ---------------------------------------------------------------------------


def test_append_mode_adds_snapshots_to_existing_file(
    mlp: nn.Module, tmp_path: Path
) -> None:
    obs1 = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5", create_new=True)
    obs1.snapshot(epoch=0)
    obs1.close()

    obs2 = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5", create_new=False)
    obs2.snapshot(epoch=1)
    obs2.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert "epoch_0000" in f
        assert "epoch_0001" in f


def test_append_mode_preserves_existing_snapshots(
    mlp: nn.Module, tmp_path: Path
) -> None:
    original = {
        name: param.detach().cpu().numpy().copy()
        for name, param in mlp.named_parameters()
    }

    obs1 = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5")
    obs1.snapshot(epoch=0)
    obs1.close()

    obs2 = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5", create_new=False)
    obs2.snapshot(epoch=1)
    loaded = obs2.load_snapshot(epoch=0)
    obs2.close()

    for name, arr in original.items():
        np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)


# ---------------------------------------------------------------------------
# Nested parameter names with dots
# ---------------------------------------------------------------------------


class _NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8))
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def test_dotted_parameter_names_round_trip(tmp_path: Path) -> None:
    model = _NestedModel()
    param_names = {name for name, _ in model.named_parameters()}
    assert any("." in name for name in param_names), "fixture has no dotted names"

    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    loaded = obs.load_snapshot(epoch=0)
    obs.close()

    assert set(loaded.keys()) == param_names
    for name, param in model.named_parameters():
        np.testing.assert_allclose(
            loaded[name], param.detach().cpu().numpy(), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Compression: chunks attribute
# ---------------------------------------------------------------------------


def test_chunks_not_none_when_compress_true(mlp: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, compress=True)
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name in grp:
            assert grp[name].chunks is not None, f"Expected chunks for {name}"


def test_chunks_none_when_compress_false(mlp: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, compress=False)
    obs.snapshot(epoch=0)
    obs.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0000"]
        for name in grp:
            assert grp[name].chunks is None, f"Expected no chunks for {name}"


# ---------------------------------------------------------------------------
# load_snapshot return type
# ---------------------------------------------------------------------------


def test_load_snapshot_returns_numpy_arrays(
    mlp: nn.Module, obs: NeuroInquisitor
) -> None:
    obs.snapshot(epoch=0)
    loaded = obs.load_snapshot(epoch=0)
    for name, arr in loaded.items():
        assert isinstance(arr, np.ndarray), f"{name} is {type(arr)}, expected ndarray"


