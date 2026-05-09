"""Tests for built-in analyzers (NI-GAMMA-001 through NI-GAMMA-005)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.analyzers import (
    probe_linear,
    projection_embed,
    similarity_compare,
    spectrum_rank,
    trajectory_stats,
)
from neuroinquisitor.replay import ReplaySession


# ---------------------------------------------------------------------------
# Shared toy model and data helpers
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _make_weights_history(n_epochs: int = 5) -> dict[int, np.ndarray]:
    """Synthetic per-epoch weight arrays (4×8 linear layer)."""
    torch.manual_seed(0)
    base = torch.randn(8, 4)
    history: dict[int, np.ndarray] = {}
    for e in range(n_epochs):
        w = base + 0.1 * e * torch.randn_like(base)
        history[e] = w.numpy()
    return history


def _make_layer_snapshot(n_layers: int = 3) -> dict[str, np.ndarray]:
    """Synthetic layer-keyed weight arrays for one epoch."""
    torch.manual_seed(1)
    return {f"layer_{i}": torch.randn(8, 4).numpy() for i in range(n_layers)}


def _make_activations(
    n_samples: int = 16, n_features: int = 8, n_layers: int = 2
) -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    return {f"fc{i}": torch.randn(n_samples, n_features) for i in range(n_layers)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def multi_epoch_run(tmp_path: Path) -> tuple[Path, Callable[[], nn.Module]]:
    torch.manual_seed(0)
    model = TinyMLP()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    for ep in range(4):
        nn.CrossEntropyLoss()(model(X), y).backward()
        opt.step()
        opt.zero_grad()
        obs.snapshot(epoch=ep)
    obs.close()
    return tmp_path, TinyMLP


@pytest.fixture()
def small_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(3)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)


# ---------------------------------------------------------------------------
# NI-GAMMA-001: trajectory_stats
# ---------------------------------------------------------------------------


def test_trajectory_stats_returns_dataframe() -> None:
    df = trajectory_stats(_make_weights_history())
    assert isinstance(df, pd.DataFrame)


def test_trajectory_stats_column_schema() -> None:
    df = trajectory_stats(_make_weights_history())
    expected = {
        "epoch", "l2_from_init", "cosine_to_init", "cosine_to_final",
        "update_norm", "velocity", "acceleration",
    }
    assert set(df.columns) == expected


def test_trajectory_stats_no_ni_types() -> None:
    df = trajectory_stats(_make_weights_history())
    for col in df.columns:
        assert df[col].dtype in (np.dtype("float64"), np.dtype("int64"), object), (
            f"Column {col!r} has unexpected dtype {df[col].dtype}"
        )


def test_trajectory_stats_row_count() -> None:
    history = _make_weights_history(n_epochs=5)
    df = trajectory_stats(history)
    assert len(df) == 5


def test_trajectory_stats_epochs_sorted() -> None:
    df = trajectory_stats(_make_weights_history(5))
    assert list(df["epoch"]) == [0, 1, 2, 3, 4]


def test_trajectory_stats_l2_init_is_zero_at_epoch_0() -> None:
    df = trajectory_stats(_make_weights_history(3))
    assert df.loc[0, "l2_from_init"] == pytest.approx(0.0)


def test_trajectory_stats_cosine_init_is_one_at_epoch_0() -> None:
    df = trajectory_stats(_make_weights_history(3))
    assert df.loc[0, "cosine_to_init"] == pytest.approx(1.0)


def test_trajectory_stats_cosine_final_is_one_at_last_epoch() -> None:
    df = trajectory_stats(_make_weights_history(3))
    assert df.iloc[-1]["cosine_to_final"] == pytest.approx(1.0)


def test_trajectory_stats_update_norm_nan_at_epoch_0() -> None:
    df = trajectory_stats(_make_weights_history(3))
    assert pd.isna(df.loc[0, "update_norm"])


def test_trajectory_stats_update_norm_positive_after_epoch_0() -> None:
    df = trajectory_stats(_make_weights_history(3))
    assert (df["update_norm"].iloc[1:] > 0).all()


def test_trajectory_stats_velocity_equals_update_norm() -> None:
    df = trajectory_stats(_make_weights_history(3))
    pd.testing.assert_series_equal(
        df["velocity"].reset_index(drop=True),
        df["update_norm"].reset_index(drop=True),
        check_names=False,
    )


def test_trajectory_stats_acceleration_nan_first_two() -> None:
    df = trajectory_stats(_make_weights_history(5))
    assert pd.isna(df.loc[0, "acceleration"])
    assert pd.isna(df.loc[1, "acceleration"])


def test_trajectory_stats_empty_input() -> None:
    df = trajectory_stats({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_trajectory_stats_regression() -> None:
    # Fixed seed: deterministic values
    weights = {0: np.zeros(4), 1: np.array([1.0, 0.0, 0.0, 0.0])}
    df = trajectory_stats(weights)
    assert df.loc[1, "l2_from_init"] == pytest.approx(1.0)
    assert df.loc[1, "update_norm"] == pytest.approx(1.0)
    assert df.loc[1, "cosine_to_init"] == pytest.approx(float("nan"), nan_ok=True)


# ---------------------------------------------------------------------------
# NI-GAMMA-002: spectrum_rank
# ---------------------------------------------------------------------------


def test_spectrum_rank_returns_dataframe() -> None:
    df = spectrum_rank(_make_layer_snapshot())
    assert isinstance(df, pd.DataFrame)


def test_spectrum_rank_column_schema() -> None:
    df = spectrum_rank(_make_layer_snapshot())
    assert set(df.columns) == {"layer", "epoch", "spectral_norm", "frobenius_norm",
                                "stable_rank", "effective_rank"}


def test_spectrum_rank_no_ni_types() -> None:
    df = spectrum_rank(_make_layer_snapshot(3))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_spectrum_rank_one_row_per_layer() -> None:
    weights = _make_layer_snapshot(n_layers=3)
    df = spectrum_rank(weights)
    assert len(df) == 3
    assert set(df["layer"]) == set(weights.keys())


def test_spectrum_rank_epoch_column() -> None:
    df = spectrum_rank(_make_layer_snapshot(), epoch=7)
    assert (df["epoch"] == 7).all()


def test_spectrum_rank_default_epoch_zero() -> None:
    df = spectrum_rank(_make_layer_snapshot())
    assert (df["epoch"] == 0).all()


def test_spectrum_rank_spectral_norm_positive() -> None:
    df = spectrum_rank(_make_layer_snapshot())
    assert (df["spectral_norm"] > 0).all()


def test_spectrum_rank_frobenius_norm_positive() -> None:
    df = spectrum_rank(_make_layer_snapshot())
    assert (df["frobenius_norm"] > 0).all()


def test_spectrum_rank_stable_rank_leq_rank() -> None:
    weights = {"w": np.eye(4, dtype=np.float32)}
    df = spectrum_rank(weights)
    # Identity matrix: spectral_norm=1, frob^2=4, stable_rank=4
    assert df.loc[0, "stable_rank"] == pytest.approx(4.0)


def test_spectrum_rank_effective_rank_identity() -> None:
    # Identity: all singular values equal → max entropy → effective_rank = n
    weights = {"w": np.eye(4, dtype=np.float32)}
    df = spectrum_rank(weights)
    assert df.loc[0, "effective_rank"] == pytest.approx(4.0, rel=1e-4)


def test_spectrum_rank_regression() -> None:
    np.random.seed(42)
    w = np.random.randn(8, 4).astype(np.float32)
    df = spectrum_rank({"fc": w})
    sv = np.linalg.svd(w, compute_uv=False)
    assert df.loc[0, "spectral_norm"] == pytest.approx(float(sv[0]), rel=1e-5)
    assert df.loc[0, "frobenius_norm"] == pytest.approx(float(np.linalg.norm(w, "fro")), rel=1e-5)


def test_spectrum_rank_zero_weight_matrix() -> None:
    weights = {"zero": np.zeros((4, 4), dtype=np.float32)}
    df = spectrum_rank(weights)
    assert df.loc[0, "spectral_norm"] == pytest.approx(0.0)
    assert df.loc[0, "frobenius_norm"] == pytest.approx(0.0)
    assert pd.isna(df.loc[0, "stable_rank"])
    assert pd.isna(df.loc[0, "effective_rank"])


def test_spectrum_rank_conv_weights() -> None:
    # Conv weights: (out_ch, in_ch, kH, kW) → reshape to (out_ch, -1)
    w = np.random.randn(8, 4, 3, 3).astype(np.float32)
    df = spectrum_rank({"conv": w})
    assert len(df) == 1
    assert df.loc[0, "spectral_norm"] > 0


# ---------------------------------------------------------------------------
# NI-GAMMA-003: projection_embed
# ---------------------------------------------------------------------------


def test_projection_embed_returns_dataframe() -> None:
    df = projection_embed(_make_activations())
    assert isinstance(df, pd.DataFrame)


def test_projection_embed_column_schema_2d() -> None:
    df = projection_embed(_make_activations())
    assert set(df.columns) == {"layer", "sample_idx", "component_0", "component_1"}


def test_projection_embed_column_schema_3d() -> None:
    df = projection_embed(_make_activations(), n_components=3)
    assert "component_2" in df.columns


def test_projection_embed_no_ni_types() -> None:
    df = projection_embed(_make_activations())
    assert isinstance(df, pd.DataFrame)
    for col in ["component_0", "component_1"]:
        assert df[col].dtype == np.float64


def test_projection_embed_row_count() -> None:
    acts = _make_activations(n_samples=16, n_layers=2)
    df = projection_embed(acts)
    # 2 layers × 16 samples
    assert len(df) == 32


def test_projection_embed_sample_idx_range() -> None:
    acts = _make_activations(n_samples=16, n_layers=1)
    df = projection_embed(acts)
    assert set(df["sample_idx"]) == set(range(16))


def test_projection_embed_pca_shape() -> None:
    acts = {"fc": torch.randn(10, 6)}
    df = projection_embed(acts, n_components=2)
    assert len(df) == 10
    assert "component_0" in df.columns
    assert "component_1" in df.columns


def test_projection_embed_umap_raises_without_umap() -> None:
    try:
        import umap  # type: ignore[import-untyped]  # noqa: F401
        pytest.skip("umap-learn is installed; skip ImportError test")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="umap-learn"):
        projection_embed(_make_activations(), method="umap")


def test_projection_embed_pca_deterministic() -> None:
    acts = _make_activations(n_samples=8, n_layers=1)
    df1 = projection_embed(acts, n_components=2)
    df2 = projection_embed(acts, n_components=2)
    pd.testing.assert_frame_equal(df1, df2)


def test_projection_embed_fewer_features_than_components() -> None:
    # Only 2 features but requesting 3 components; should pad, not crash.
    acts = {"tiny": torch.randn(5, 2)}
    df = projection_embed(acts, n_components=3)
    assert len(df) == 5
    assert "component_2" in df.columns


# ---------------------------------------------------------------------------
# NI-GAMMA-004: similarity_compare
# ---------------------------------------------------------------------------


def test_similarity_compare_returns_dataframe() -> None:
    a = _make_activations(n_samples=8, n_layers=2)
    b = _make_activations(n_samples=8, n_layers=2)
    df = similarity_compare(a, b)
    assert isinstance(df, pd.DataFrame)


def test_similarity_compare_column_schema() -> None:
    a = _make_activations(n_samples=8, n_layers=2)
    b = _make_activations(n_samples=8, n_layers=2)
    df = similarity_compare(a, b)
    assert set(df.columns) == {"layer_a", "layer_b", "cka"}


def test_similarity_compare_no_ni_types() -> None:
    a = _make_activations(n_samples=8, n_layers=1)
    b = _make_activations(n_samples=8, n_layers=1)
    df = similarity_compare(a, b)
    assert df["cka"].dtype == np.float64


def test_similarity_compare_cross_product_size() -> None:
    a = _make_activations(n_samples=8, n_layers=2)
    b = _make_activations(n_samples=8, n_layers=3)
    df = similarity_compare(a, b)
    assert len(df) == 2 * 3


def test_similarity_compare_self_cka_is_one() -> None:
    acts = _make_activations(n_samples=16, n_layers=1)
    df = similarity_compare(acts, acts)
    diag = df[df["layer_a"] == df["layer_b"]]
    for val in diag["cka"].values:
        assert val == pytest.approx(1.0, rel=1e-5)


def test_similarity_compare_within_run(
    multi_epoch_run: tuple[Path, Callable[[], nn.Module]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """Integration: replay two epochs and compare activations with CKA."""
    run_dir, factory = multi_epoch_run

    def replay(ep: int) -> dict[str, torch.Tensor]:
        return ReplaySession(
            run=run_dir,
            checkpoint=ep,
            model_factory=factory,
            dataloader=small_dataloader,
            modules=["fc1", "fc2"],
            capture=["activations"],
        ).run().activations

    df = similarity_compare(replay(0), replay(3))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"layer_a", "layer_b", "cka"}
    assert len(df) == 4  # 2×2 cross-product


def test_similarity_compare_zero_activations_returns_nan() -> None:
    a = {"fc": torch.zeros(8, 4)}
    b = {"fc": torch.randn(8, 4)}
    df = similarity_compare(a, b)
    assert pd.isna(df.loc[0, "cka"])


def test_similarity_compare_mismatched_samples_raises() -> None:
    a = {"fc": torch.randn(8, 4)}
    b = {"fc": torch.randn(10, 4)}
    with pytest.raises(ValueError, match="Sample dimension mismatch"):
        similarity_compare(a, b)


# ---------------------------------------------------------------------------
# NI-GAMMA-005: probe_linear
# ---------------------------------------------------------------------------


def test_probe_linear_returns_dataframe() -> None:
    acts = _make_activations(n_samples=20, n_layers=2)
    labels = torch.randint(0, 2, (20,))
    df = probe_linear(acts, labels)
    assert isinstance(df, pd.DataFrame)


def test_probe_linear_column_schema() -> None:
    acts = _make_activations(n_samples=20, n_layers=2)
    labels = torch.randint(0, 2, (20,))
    df = probe_linear(acts, labels)
    assert set(df.columns) == {"layer", "train_accuracy", "val_accuracy"}


def test_probe_linear_no_ni_types() -> None:
    acts = _make_activations(n_samples=20, n_layers=1)
    labels = torch.randint(0, 2, (20,))
    df = probe_linear(acts, labels)
    assert df["train_accuracy"].dtype == np.float64
    assert df["val_accuracy"].dtype == np.float64


def test_probe_linear_one_row_per_layer() -> None:
    acts = _make_activations(n_samples=20, n_layers=3)
    labels = torch.randint(0, 2, (20,))
    df = probe_linear(acts, labels)
    assert len(df) == 3


def test_probe_linear_accuracy_in_unit_interval() -> None:
    acts = _make_activations(n_samples=20, n_layers=2)
    labels = torch.randint(0, 2, (20,))
    df = probe_linear(acts, labels)
    assert ((df["train_accuracy"] >= 0) & (df["train_accuracy"] <= 1)).all()
    assert ((df["val_accuracy"] >= 0) & (df["val_accuracy"] <= 1)).all()


def test_probe_linear_deterministic_split() -> None:
    acts = _make_activations(n_samples=20, n_layers=1)
    labels = torch.randint(0, 2, (20,))
    df1 = probe_linear(acts, labels, random_state=7)
    df2 = probe_linear(acts, labels, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_probe_linear_different_seeds_may_differ() -> None:
    # With informative activations, different splits should give different results.
    torch.manual_seed(0)
    X = torch.randn(40, 8)
    labels = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    acts = {"fc": X}
    df0 = probe_linear(acts, labels, random_state=0)
    df1 = probe_linear(acts, labels, random_state=99)
    # At least one metric should differ when seeds differ
    assert (
        df0["val_accuracy"].values[0] != df1["val_accuracy"].values[0]
        or df0["train_accuracy"].values[0] != df1["train_accuracy"].values[0]
    )


def test_probe_linear_separable_data_high_accuracy() -> None:
    # Perfectly linearly separable activations → high train accuracy
    n = 40
    X = torch.cat([torch.ones(n // 2, 4), -torch.ones(n // 2, 4)], dim=0)
    X += 0.01 * torch.randn_like(X)
    labels = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
    df = probe_linear({"fc": X}, labels)
    assert df.loc[0, "train_accuracy"] > 0.9


# ---------------------------------------------------------------------------
# Integration: replay → projection_embed + probe_linear
# ---------------------------------------------------------------------------


def test_integration_replay_to_projection(
    multi_epoch_run: tuple[Path, Callable[[], nn.Module]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """Full flow: snapshot → replay → projection_embed."""
    run_dir, factory = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=2,
        model_factory=factory,
        dataloader=small_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations"],
    ).run()

    df = projection_embed(result.activations, n_components=2)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"layer", "sample_idx", "component_0", "component_1"}
    assert set(df["layer"]) == {"fc1", "fc2"}
    assert len(df) == 2 * 20  # 2 layers × 20 samples


def test_integration_replay_to_probe(
    multi_epoch_run: tuple[Path, Callable[[], nn.Module]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """Full flow: snapshot → replay → probe_linear."""
    run_dir, factory = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=2,
        model_factory=factory,
        dataloader=small_dataloader,
        modules=["fc1"],
        capture=["activations"],
    ).run()

    labels = torch.randint(0, 2, (20,))
    df = probe_linear(result.activations, labels)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"layer", "train_accuracy", "val_accuracy"}
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Integration: SnapshotCollection → trajectory_stats + spectrum_rank
# ---------------------------------------------------------------------------


def test_integration_collection_to_trajectory(
    multi_epoch_run: tuple[Path, Callable[[], nn.Module]],
) -> None:
    """Full flow: ni.load → by_layer → trajectory_stats."""
    from neuroinquisitor.loader import load as ni_load

    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    layer = col.layers[0]
    history = col.by_layer(layer)

    df = trajectory_stats(history)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(col.epochs)


def test_integration_collection_to_spectrum(
    multi_epoch_run: tuple[Path, Callable[[], nn.Module]],
) -> None:
    """Full flow: ni.load → by_epoch → spectrum_rank."""
    from neuroinquisitor.loader import load as ni_load

    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    weights = col.by_epoch(0)

    df = spectrum_rank(weights, epoch=0)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"layer", "epoch", "spectral_norm", "frobenius_norm",
                                "stable_rank", "effective_rank"}
    assert len(df) == len(weights)
