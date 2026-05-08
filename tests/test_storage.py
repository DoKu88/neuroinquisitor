"""Tests for storage correctness and SnapshotCollection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor, SnapshotCollection


@pytest.fixture()
def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))


@pytest.fixture()
def obs(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    observer = NeuroInquisitor(mlp, log_dir=tmp_path)
    yield observer
    if not observer._closed:
        observer.close()


# ---------------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------------


def test_epoch_file_created(obs: NeuroInquisitor, tmp_path: Path) -> None:
    obs.snapshot(epoch=0)
    assert (tmp_path / "epoch_0000.h5").exists()


def test_multiple_epoch_files_created(obs: NeuroInquisitor, tmp_path: Path) -> None:
    obs.snapshot(epoch=0)
    obs.snapshot(epoch=1)
    assert (tmp_path / "epoch_0000.h5").exists()
    assert (tmp_path / "epoch_0001.h5").exists()


def test_index_json_exists(obs: NeuroInquisitor, tmp_path: Path) -> None:
    obs.snapshot(epoch=0)
    assert (tmp_path / "index.json").exists()


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_weights_saved_with_correct_shapes(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for name, param in mlp.named_parameters():
        assert loaded[name].shape == tuple(param.shape)


def test_weights_saved_with_correct_dtype(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for name, param in mlp.named_parameters():
        expected = param.detach().cpu().numpy().dtype
        assert loaded[name].dtype == expected


# ---------------------------------------------------------------------------
# Load round-trip
# ---------------------------------------------------------------------------


def test_load_values_match_original(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    original = {name: param.detach().cpu().numpy() for name, param in mlp.named_parameters()}
    obs.snapshot(epoch=7)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(7)
    for name, arr in original.items():
        np.testing.assert_allclose(loaded[name], arr, rtol=1e-6)


def test_load_raises_for_missing_epoch(obs: NeuroInquisitor, tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="epoch 99"):
        NeuroInquisitor.load(tmp_path).by_epoch(99)


def test_load_returns_numpy_arrays(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    obs.snapshot(epoch=0)
    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for arr in loaded.values():
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Multiple snapshots coexist
# ---------------------------------------------------------------------------


def test_multiple_snapshots_coexist(
    mlp: nn.Module, obs: NeuroInquisitor, tmp_path: Path
) -> None:
    for epoch in range(5):
        with torch.no_grad():
            for p in mlp.parameters():
                p.add_(0.01)
        obs.snapshot(epoch=epoch)
    col = NeuroInquisitor.load(tmp_path)
    for epoch in range(5):
        loaded = col.by_epoch(epoch)
        assert set(loaded.keys()) == {name for name, _ in mlp.named_parameters()}


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------


def test_append_mode_adds_snapshots(mlp: nn.Module, tmp_path: Path) -> None:
    obs1 = NeuroInquisitor(mlp, log_dir=tmp_path, create_new=True)
    obs1.snapshot(epoch=0)
    obs1.close()

    obs2 = NeuroInquisitor(mlp, log_dir=tmp_path, create_new=False)
    obs2.snapshot(epoch=1)
    obs2.close()

    col = NeuroInquisitor.load(tmp_path)
    assert col.epochs == [0, 1]


def test_append_mode_preserves_existing_snapshots(
    mlp: nn.Module, tmp_path: Path
) -> None:
    original = {
        name: param.detach().cpu().numpy().copy()
        for name, param in mlp.named_parameters()
    }
    obs1 = NeuroInquisitor(mlp, log_dir=tmp_path)
    obs1.snapshot(epoch=0)
    obs1.close()

    obs2 = NeuroInquisitor(mlp, log_dir=tmp_path, create_new=False)
    obs2.snapshot(epoch=1)
    obs2.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
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
    assert any("." in name for name in param_names)

    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    assert set(loaded.keys()) == param_names
    for name, param in model.named_parameters():
        np.testing.assert_allclose(
            loaded[name], param.detach().cpu().numpy(), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# SnapshotCollection
# ---------------------------------------------------------------------------


@pytest.fixture()
def multi_obs(mlp: nn.Module, tmp_path: Path) -> NeuroInquisitor:
    obs = NeuroInquisitor(mlp, log_dir=tmp_path)
    for epoch in range(4):
        with torch.no_grad():
            for p in mlp.parameters():
                p.add_(0.1)
        obs.snapshot(epoch=epoch)
    obs.close()
    return NeuroInquisitor(mlp, log_dir=tmp_path, create_new=False)


def test_load_returns_collection(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    assert isinstance(col, SnapshotCollection)
    assert len(col) == 4


def test_load_epochs_property(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    assert col.epochs == [0, 1, 2, 3]


def test_load_layers_property(
    mlp: nn.Module, multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    assert set(col.layers) == {name for name, _ in mlp.named_parameters()}


def test_by_epoch_returns_correct_params(
    mlp: nn.Module, multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    params = col.by_epoch(0)
    assert set(params.keys()) == {name for name, _ in mlp.named_parameters()}
    for arr in params.values():
        assert isinstance(arr, np.ndarray)


def test_by_epoch_raises_for_missing(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    with pytest.raises(KeyError, match="epoch 99"):
        col.by_epoch(99)


def test_by_layer_returns_all_epochs(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    result = col.by_layer("0.weight")
    assert set(result.keys()) == {0, 1, 2, 3}
    for arr in result.values():
        assert isinstance(arr, np.ndarray)


def test_by_layer_reads_in_parallel(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    result = col.by_layer("0.weight", max_workers=4)
    assert set(result.keys()) == {0, 1, 2, 3}


def test_by_layer_values_differ_across_epochs(
    multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    result = col.by_layer("0.weight")
    assert not np.allclose(result[0], result[3])


def test_by_layer_raises_for_missing(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    with pytest.raises(KeyError, match="not found"):
        col.by_layer("nonexistent.weight")


def test_select_by_epoch_range(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=range(1, 3))
    assert sub.epochs == [1, 2]
    assert len(sub) == 2


def test_select_by_epoch_list(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=[0, 3])
    assert sub.epochs == [0, 3]


def test_select_by_single_epoch(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=2)
    assert sub.epochs == [2]
    assert len(sub) == 1


def test_select_by_layers(
    mlp: nn.Module, multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(layers=["0.weight", "0.bias"])
    assert set(sub.layers) == {"0.weight", "0.bias"}


def test_select_by_single_layer(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(layers="0.weight")
    assert sub.layers == ["0.weight"]


def test_select_combined_epochs_and_layers(
    multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=range(0, 2), layers="0.weight")
    assert sub.epochs == [0, 1]
    assert sub.layers == ["0.weight"]


def test_select_returns_new_collection(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=range(0, 2))
    assert sub is not col
    assert len(col) == 4


def test_select_is_composable(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    sub = col.select(epochs=range(0, 3)).select(epochs=range(1, 4))
    assert sub.epochs == [1, 2]


def test_collection_repr(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()
    r = repr(col)
    assert "SnapshotCollection" in r
    assert "snapshots=4" in r


def test_load_is_lazy(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    """NeuroInquisitor.load must not read tensor files — only the index."""
    import unittest.mock as mock

    col = NeuroInquisitor.load(tmp_path)
    multi_obs.close()

    with mock.patch.object(col._format, "read", wraps=col._format.read) as mock_read:
        _ = col.epochs
        _ = col.layers
        _ = len(col)
        mock_read.assert_not_called()


# ---------------------------------------------------------------------------
# load with epoch/layer filters
# ---------------------------------------------------------------------------


def test_load_with_epoch_filter(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path, epochs=range(1, 3))
    multi_obs.close()
    assert col.epochs == [1, 2]


def test_load_with_single_epoch(multi_obs: NeuroInquisitor, tmp_path: Path) -> None:
    col = NeuroInquisitor.load(tmp_path, epochs=2)
    multi_obs.close()
    assert col.epochs == [2]
    assert len(col) == 1


def test_load_with_layer_filter(
    mlp: nn.Module, multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path, layers=["0.weight", "0.bias"])
    multi_obs.close()
    assert set(col.layers) == {"0.weight", "0.bias"}


def test_load_with_epoch_and_layer_filter(
    multi_obs: NeuroInquisitor, tmp_path: Path
) -> None:
    col = NeuroInquisitor.load(tmp_path, epochs=range(0, 2), layers="0.weight")
    multi_obs.close()
    assert col.epochs == [0, 1]
    assert col.layers == ["0.weight"]
