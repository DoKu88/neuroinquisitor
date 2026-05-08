"""Tests for replay-based activation and gradient capture (NI-BETA-001–004)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.replay import (
    BalancedNSlice,
    CheckpointSelector,
    ExplicitIndicesSlice,
    FirstNSlice,
    RandomNSlice,
    ReplayConfig,
    ReplayMetadata,
    ReplayResult,
    ReplaySession,
)


# ---------------------------------------------------------------------------
# Tiny models
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlp_run(tmp_path: Path) -> tuple[Path, Callable[[], nn.Module]]:
    torch.manual_seed(0)
    model = TinyMLP()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    return tmp_path, TinyMLP


@pytest.fixture()
def cnn_run(tmp_path: Path) -> tuple[Path, Callable[[], nn.Module]]:
    torch.manual_seed(0)
    model = TinyCNN()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    return tmp_path, TinyCNN


@pytest.fixture()
def mlp_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(1)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


@pytest.fixture()
def cnn_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(2)
    X = torch.randn(16, 1, 8, 8)
    y = torch.randint(0, 2, (16,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# NI-BETA-001: ReplaySession end-to-end (MLP and CNN)
# ---------------------------------------------------------------------------


def test_replay_session_mlp_end_to_end(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations", "logits"],
    )
    result = session.run()

    assert isinstance(result, ReplayResult)
    assert "fc1" in result.activations
    assert "fc2" in result.activations
    assert result.logits is not None
    assert result.metadata is not None
    assert result.metadata.checkpoint_epoch == 0
    assert result.metadata.n_samples == 16


def test_replay_session_cnn_end_to_end(
    cnn_run: tuple[Path, Callable[[], nn.Module]],
    cnn_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = cnn_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=cnn_dataloader,
        modules=["conv1", "fc"],
        capture=["activations"],
    )
    result = session.run()

    assert "conv1" in result.activations
    assert "fc" in result.activations
    assert result.metadata is not None


def test_replay_session_accepts_checkpoint_selector(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=CheckpointSelector(epoch=0),
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    )
    result = session.run()
    assert "fc1" in result.activations


# ---------------------------------------------------------------------------
# NI-BETA-001: invalid module names produce clear errors
# ---------------------------------------------------------------------------


def test_invalid_module_name_raises(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["nonexistent_layer"],
        capture=["activations"],
    )
    with pytest.raises(ValueError, match="Invalid module name"):
        session.run()


def test_invalid_module_name_lists_available_modules(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["bad_name"],
        capture=["activations"],
    )
    with pytest.raises(ValueError, match="Available modules"):
        session.run()


# ---------------------------------------------------------------------------
# NI-BETA-002: activation capture — shape correctness per reduction mode
# ---------------------------------------------------------------------------


def test_activation_raw_shape(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        activation_reduction="raw",
    )
    result = session.run()
    # fc1: Linear(4→8), 16 samples → (16, 8)
    assert result.activations["fc1"].shape == (16, 8)


def test_activation_mean_shape(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        activation_reduction="mean",
    )
    result = session.run()
    # mean over 16 samples → (8,)
    assert result.activations["fc1"].shape == (8,)


def test_activation_pool_shape_linear_is_identity(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        activation_reduction="pool",
    )
    result = session.run()
    # 2-D output — pool is identity → (16, 8)
    assert result.activations["fc1"].shape == (16, 8)


def test_activation_pool_shape_conv(
    cnn_run: tuple[Path, Callable[[], nn.Module]],
    cnn_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = cnn_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=cnn_dataloader,
        modules=["conv1"],
        capture=["activations"],
        activation_reduction="pool",
    )
    result = session.run()
    # conv1: Conv2d(1,4,3) on (B,1,8,8) → (B,4,8,8); pool → (B,4)
    assert result.activations["conv1"].shape == (16, 4)


def test_artifact_sizes_reported(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations", "logits"],
    )
    result = session.run()
    assert result.metadata is not None
    assert "activations/fc1" in result.metadata.artifact_sizes
    assert result.metadata.artifact_sizes["activations/fc1"] > 0
    assert "logits" in result.metadata.artifact_sizes
    assert result.metadata.artifact_sizes["logits"] > 0


# ---------------------------------------------------------------------------
# NI-BETA-002: logits capture
# ---------------------------------------------------------------------------


def test_logits_capture_shape(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["logits"],
    )
    result = session.run()
    assert result.logits is not None
    # TinyMLP output: (16, 2)
    assert result.logits.shape == (16, 2)


def test_logits_not_captured_when_not_requested(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    )
    result = session.run()
    assert result.logits is None


# ---------------------------------------------------------------------------
# NI-BETA-003: gradient capture — shape correctness
# ---------------------------------------------------------------------------


def test_gradient_capture_per_example_shape(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
        gradient_mode="per_example",
    )
    result = session.run()
    assert "fc1" in result.gradients
    # grad of loss w.r.t. fc1 output: same shape as activation → (16, 8)
    assert result.gradients["fc1"].shape == (16, 8)


def test_gradient_capture_aggregated_shape(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
        gradient_mode="aggregated",
    )
    result = session.run()
    # mean over 16 samples → (8,)
    assert result.gradients["fc1"].shape == (8,)


def test_gradient_and_activation_captured_together(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations", "gradients"],
    )
    result = session.run()
    assert "fc1" in result.activations
    assert "fc1" in result.gradients


# ---------------------------------------------------------------------------
# NI-BETA-004: dataset slice — correctness, metadata, determinism
# ---------------------------------------------------------------------------


def test_first_n_slice_n_samples(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        dataset_slice=FirstNSlice(n=4),
    )
    result = session.run()
    assert result.metadata is not None
    assert result.metadata.n_samples == 4
    assert result.activations["fc1"].shape == (4, 8)


def test_random_n_slice_deterministic(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run

    def make() -> ReplayResult:
        return ReplaySession(
            run=run_dir,
            checkpoint=0,
            model_factory=factory,
            dataloader=mlp_dataloader,
            modules=["fc1"],
            capture=["activations"],
            dataset_slice=RandomNSlice(n=8, seed=42),
        ).run()

    r1 = make()
    r2 = make()
    torch.testing.assert_close(r1.activations["fc1"], r2.activations["fc1"])


def test_random_n_different_seeds_differ(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run

    def make(seed: int) -> ReplayResult:
        return ReplaySession(
            run=run_dir,
            checkpoint=0,
            model_factory=factory,
            dataloader=mlp_dataloader,
            modules=["fc1"],
            capture=["activations"],
            dataset_slice=RandomNSlice(n=8, seed=seed),
        ).run()

    r1 = make(0)
    r2 = make(1)
    assert not torch.allclose(r1.activations["fc1"], r2.activations["fc1"])


def test_balanced_n_slice(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        dataset_slice=BalancedNSlice(n=8, seed=0),
    )
    result = session.run()
    assert result.metadata is not None
    assert result.metadata.n_samples <= 8
    assert result.metadata.dataset_slice is not None
    assert result.metadata.dataset_slice["kind"] == "balanced_n"


def test_explicit_indices_slice(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        dataset_slice=ExplicitIndicesSlice(indices=[0, 1, 2, 3]),
    )
    result = session.run()
    assert result.metadata is not None
    assert result.metadata.n_samples == 4
    assert result.activations["fc1"].shape == (4, 8)


def test_explicit_indices_out_of_range_raises(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        dataset_slice=ExplicitIndicesSlice(indices=[0, 999]),
    )
    with pytest.raises(ValueError, match="out of range"):
        session.run()


def test_slice_metadata_persisted_in_result(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
        dataset_slice=RandomNSlice(n=6, seed=7),
    )
    result = session.run()
    assert result.metadata is not None
    assert result.metadata.dataset_slice is not None
    assert result.metadata.dataset_slice["kind"] == "random_n"
    assert result.metadata.dataset_slice["n"] == 6
    assert result.metadata.dataset_slice["seed"] == 7


def test_no_slice_uses_all_samples(
    mlp_run: tuple[Path, Callable[[], nn.Module]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, factory = mlp_run
    session = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=factory,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    )
    result = session.run()
    assert result.metadata is not None
    assert result.metadata.n_samples == 16
    assert result.metadata.dataset_slice is None


# ---------------------------------------------------------------------------
# NI-BETA-001: pydantic validation on config and slice models
# ---------------------------------------------------------------------------


def test_replay_config_rejects_empty_modules() -> None:
    with pytest.raises(Exception):
        ReplayConfig(
            checkpoint=CheckpointSelector(epoch=0),
            modules=[],
            capture=["activations"],
        )


def test_replay_config_rejects_empty_capture() -> None:
    with pytest.raises(Exception):
        ReplayConfig(
            checkpoint=CheckpointSelector(epoch=0),
            modules=["fc1"],
            capture=[],
        )


def test_replay_config_rejects_invalid_capture_kind() -> None:
    with pytest.raises(Exception):
        ReplayConfig(
            checkpoint=CheckpointSelector(epoch=0),
            modules=["fc1"],
            capture=["invalid_kind"],  # type: ignore[list-item]
        )


def test_checkpoint_selector_rejects_negative_epoch() -> None:
    with pytest.raises(Exception):
        CheckpointSelector(epoch=-1)


def test_first_n_slice_rejects_zero() -> None:
    with pytest.raises(Exception):
        FirstNSlice(n=0)


def test_first_n_slice_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        FirstNSlice(n=5, unknown=True)  # type: ignore[call-arg]


def test_explicit_indices_slice_rejects_empty() -> None:
    with pytest.raises(Exception):
        ExplicitIndicesSlice(indices=[])


def test_replay_config_round_trip() -> None:
    config = ReplayConfig(
        checkpoint=CheckpointSelector(epoch=3),
        modules=["fc1", "fc2"],
        capture=["activations", "gradients"],
        activation_reduction="mean",
        gradient_mode="per_example",
        dataset_slice=RandomNSlice(n=32, seed=99),
    )
    restored = ReplayConfig.model_validate(config.model_dump())
    assert restored == config


def test_replay_metadata_round_trip() -> None:
    meta = ReplayMetadata(
        run="/tmp/run",
        checkpoint_epoch=5,
        modules=["fc1"],
        capture=["activations"],
        activation_reduction="raw",
        gradient_mode="aggregated",
        dataset_slice={"kind": "first_n", "n": 10},
        n_samples=10,
        artifact_sizes={"activations/fc1": 512},
    )
    restored = ReplayMetadata.model_validate(meta.model_dump())
    assert restored == meta
