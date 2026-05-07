"""Tests for snapshot() and load_snapshot() — Sprint 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


@pytest.fixture()
def observer(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path, filename="weights.h5")
    yield obs
    if not obs._closed:
        obs.close()


# ---------------------------------------------------------------------------
# Basic snapshot correctness
# ---------------------------------------------------------------------------


def test_snapshot_saves_all_parameter_names(
    mlp: nn.Module, observer: NeuroInquisitor
) -> None:
    observer.snapshot(epoch=0)
    loaded = observer.load_snapshot(epoch=0)
    expected_names = {name for name, _ in mlp.named_parameters()}
    assert set(loaded.keys()) == expected_names


def test_snapshot_saves_correct_shapes(
    mlp: nn.Module, observer: NeuroInquisitor
) -> None:
    observer.snapshot(epoch=0)
    loaded = observer.load_snapshot(epoch=0)
    for name, param in mlp.named_parameters():
        assert loaded[name].shape == tuple(param.shape), f"Shape mismatch for {name}"


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_metadata_stored_as_group_attributes(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    import h5py

    meta = {"lr": 0.001, "optimizer": "adam"}
    observer.snapshot(epoch=1, metadata=meta)
    observer.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0001"]
        assert grp.attrs["lr"] == pytest.approx(0.001)
        assert grp.attrs["optimizer"] == "adam"
        assert grp.attrs["epoch"] == 1


def test_metadata_none_stores_epoch_attr(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    import h5py

    observer.snapshot(epoch=3)
    observer.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert f["epoch_0003"].attrs["epoch"] == 3


# ---------------------------------------------------------------------------
# Duplicate snapshot prevention
# ---------------------------------------------------------------------------


def test_duplicate_epoch_raises_value_error(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=5)
    with pytest.raises(ValueError, match="epoch_0005"):
        observer.snapshot(epoch=5)


def test_duplicate_epoch_step_raises_value_error(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=2, step=10)
    with pytest.raises(ValueError, match="epoch_0002_step_000010"):
        observer.snapshot(epoch=2, step=10)


def test_different_steps_same_epoch_allowed(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=0, step=0)
    observer.snapshot(epoch=0, step=1)  # should not raise


# ---------------------------------------------------------------------------
# GPU snapshot
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_snapshot_moves_to_cpu(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).cuda()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    loaded = obs.load_snapshot(epoch=0)
    obs.close()

    for arr in loaded.values():
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Shape preservation after save and load
# ---------------------------------------------------------------------------


def test_shapes_preserved_after_round_trip(
    mlp: nn.Module, observer: NeuroInquisitor
) -> None:
    observer.snapshot(epoch=0)
    loaded = observer.load_snapshot(epoch=0)
    for name, param in mlp.named_parameters():
        assert loaded[name].shape == tuple(param.shape)


# ---------------------------------------------------------------------------
# Error when closed
# ---------------------------------------------------------------------------


def test_snapshot_raises_when_closed(observer: NeuroInquisitor) -> None:
    observer.close()
    with pytest.raises(RuntimeError, match="closed"):
        observer.snapshot(epoch=0)


def test_snapshot_requires_epoch_or_step(observer: NeuroInquisitor) -> None:
    with pytest.raises(ValueError, match="epoch or step"):
        observer.snapshot()


# ---------------------------------------------------------------------------
# Step-only snapshot
# ---------------------------------------------------------------------------


def test_step_only_snapshot_creates_correct_group(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    import h5py

    observer.snapshot(step=42)
    observer.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert "step_000042" in f
        assert f["step_000042"].attrs["step"] == 42
        assert "epoch" not in f["step_000042"].attrs


def test_step_only_snapshot_saves_parameters(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    import h5py

    observer.snapshot(step=0)
    observer.close()

    expected = {name for name, _ in mlp.named_parameters()}
    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert set(f["step_000000"].keys()) == expected


def test_duplicate_step_only_raises_value_error(observer: NeuroInquisitor) -> None:
    observer.snapshot(step=7)
    with pytest.raises(ValueError, match="step_000007"):
        observer.snapshot(step=7)


# ---------------------------------------------------------------------------
# load_snapshot after observer.close()
# ---------------------------------------------------------------------------


def test_load_snapshot_works_after_close(
    mlp: nn.Module, observer: NeuroInquisitor
) -> None:
    original = {
        name: param.detach().cpu().numpy().copy()
        for name, param in mlp.named_parameters()
    }
    observer.snapshot(epoch=0)
    observer.close()

    loaded = observer.load_snapshot(epoch=0)
    for name, arr in original.items():
        np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)


# ---------------------------------------------------------------------------
# Metadata reserved key collision
# ---------------------------------------------------------------------------


def test_metadata_epoch_key_raises_value_error(observer: NeuroInquisitor) -> None:
    with pytest.raises(ValueError, match="reserved key"):
        observer.snapshot(epoch=0, metadata={"epoch": 999})


def test_metadata_step_key_raises_value_error(observer: NeuroInquisitor) -> None:
    with pytest.raises(ValueError, match="reserved key"):
        observer.snapshot(epoch=0, metadata={"step": 5})


def test_metadata_both_reserved_keys_raises(observer: NeuroInquisitor) -> None:
    with pytest.raises(ValueError, match="reserved key"):
        observer.snapshot(epoch=0, metadata={"epoch": 0, "step": 0})


def test_snapshot_still_usable_after_reserved_key_error(
    observer: NeuroInquisitor,
) -> None:
    # Bug regression: reserved key error must not poison the epoch slot.
    with pytest.raises(ValueError, match="reserved key"):
        observer.snapshot(epoch=0, metadata={"epoch": 99})
    # Same epoch must succeed now that validation fires before group creation.
    observer.snapshot(epoch=0)


def test_metadata_empty_dict_is_noop(observer: NeuroInquisitor, tmp_path: Path) -> None:
    import h5py

    observer.snapshot(epoch=0, metadata={})
    observer.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        # Only the built-in "epoch" attr should be present; no extras.
        assert set(f["epoch_0000"].attrs.keys()) == {"epoch"}


# ---------------------------------------------------------------------------
# Step attr stored for epoch+step snapshots
# ---------------------------------------------------------------------------


def test_epoch_and_step_both_stored_as_attrs(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    import h5py

    observer.snapshot(epoch=3, step=50)
    observer.close()

    with h5py.File(tmp_path / "weights.h5", "r") as f:
        grp = f["epoch_0003_step_000050"]
        assert grp.attrs["epoch"] == 3
        assert grp.attrs["step"] == 50


# ---------------------------------------------------------------------------
# load_snapshot for step-only snapshot (documents unsupported path)
# ---------------------------------------------------------------------------


def test_load_snapshot_raises_for_step_only_snapshot(
    observer: NeuroInquisitor,
) -> None:
    # step-only snapshots are stored under "step_XXXXXX", not "epoch_XXXX",
    # so load_snapshot(epoch=...) cannot retrieve them.
    observer.snapshot(step=5)
    with pytest.raises(KeyError):
        observer.load_snapshot(epoch=5)


# ---------------------------------------------------------------------------
# Model with no parameters
# ---------------------------------------------------------------------------


def test_snapshot_model_with_no_parameters(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    loaded = obs.load_snapshot(epoch=0)
    obs.close()
    assert loaded == {}
