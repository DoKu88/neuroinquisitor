"""Tests for snapshot()."""

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
    obs = NeuroInquisitor(mlp, log_dir=tmp_path)
    yield obs
    if not obs._closed:
        obs.close()


# ---------------------------------------------------------------------------
# Basic snapshot correctness
# ---------------------------------------------------------------------------


def test_snapshot_saves_all_parameter_names(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    assert set(loaded.keys()) == {name for name, _ in mlp.named_parameters()}


def test_snapshot_saves_correct_shapes(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for name, param in mlp.named_parameters():
        assert loaded[name].shape == tuple(param.shape)


def test_snapshot_values_match_model(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for name, param in mlp.named_parameters():
        np.testing.assert_allclose(loaded[name], param.detach().cpu().numpy(), rtol=1e-6)


# ---------------------------------------------------------------------------
# Metadata round-trip via index
# ---------------------------------------------------------------------------


def test_metadata_stored_in_index(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=1, metadata={"lr": 0.001, "optimizer": "adam"})
    entry = observer._index.get_by_epoch(1)
    assert entry is not None
    assert entry.metadata["lr"] == 0.001
    assert entry.metadata["optimizer"] == "adam"


def test_metadata_none_stores_epoch_in_index(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=3)
    entry = observer._index.get_by_epoch(3)
    assert entry is not None
    assert entry.epoch == 3


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
    observer.snapshot(epoch=0, step=1)  # must not raise


# ---------------------------------------------------------------------------
# GPU snapshot
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_snapshot_moves_to_cpu(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).cuda()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for arr in loaded.values():
        assert isinstance(arr, np.ndarray)


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


def test_step_only_snapshot_creates_file(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(step=42)
    assert (tmp_path / "step_000042.h5").exists()


def test_step_only_snapshot_recorded_in_index(observer: NeuroInquisitor) -> None:
    observer.snapshot(step=42)
    entries = observer._index.all()
    assert any(e.step == 42 and e.epoch is None for e in entries)


def test_step_only_snapshot_saves_parameters(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(step=0)
    col = NeuroInquisitor.load(tmp_path)
    # step-only entry has no epoch; read via the index directly
    entry = col._index.all()[0]
    path = col._backend.read_path(entry.file_key)
    loaded = col._format.read(path)
    assert set(loaded.keys()) == {name for name, _ in mlp.named_parameters()}


def test_duplicate_step_only_raises_value_error(observer: NeuroInquisitor) -> None:
    observer.snapshot(step=7)
    with pytest.raises(ValueError, match="step_000007"):
        observer.snapshot(step=7)


# ---------------------------------------------------------------------------
# Load works after observer.close()
# ---------------------------------------------------------------------------


def test_load_works_after_close(
    mlp: nn.Module, observer: NeuroInquisitor, tmp_path: Path
) -> None:
    original = {
        name: param.detach().cpu().numpy().copy()
        for name, param in mlp.named_parameters()
    }
    observer.snapshot(epoch=0)
    observer.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
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
    with pytest.raises(ValueError, match="reserved key"):
        observer.snapshot(epoch=0, metadata={"epoch": 99})
    observer.snapshot(epoch=0)  # same epoch must succeed


def test_metadata_empty_dict_stores_no_extras(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=0, metadata={})
    entry = observer._index.get_by_epoch(0)
    assert entry is not None
    assert entry.metadata == {}


# ---------------------------------------------------------------------------
# Both epoch + step stored
# ---------------------------------------------------------------------------


def test_epoch_and_step_both_in_index(observer: NeuroInquisitor) -> None:
    observer.snapshot(epoch=3, step=50)
    entry = observer._index.get_by_epoch(3)
    assert entry is not None
    assert entry.epoch == 3
    assert entry.step == 50


def test_epoch_and_step_snapshot_file_named_correctly(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(epoch=3, step=50)
    assert (tmp_path / "epoch_0003_step_000050.h5").exists()


# ---------------------------------------------------------------------------
# Load raises for step-only snapshot when queried by epoch
# ---------------------------------------------------------------------------


def test_load_raises_for_step_only_snapshot(
    observer: NeuroInquisitor, tmp_path: Path
) -> None:
    observer.snapshot(step=5)
    with pytest.raises(KeyError):
        NeuroInquisitor.load(tmp_path).by_epoch(5)


# ---------------------------------------------------------------------------
# Model with no parameters
# ---------------------------------------------------------------------------


def test_snapshot_model_with_no_parameters(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    assert loaded == {}
