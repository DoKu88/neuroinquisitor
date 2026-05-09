"""Captum compatibility tests (NI-DELTA-007).

Verifies that NI outputs feed into Captum attribution methods without
any type conversion.  All tests use the public NI API only.

Skip condition: captum not installed.
Install:        pip install captum

Run standalone:
    pytest tests/compat/test_captum.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.loader import load as ni_load
from neuroinquisitor.replay import ReplaySession

captum_attr = pytest.importorskip(
    "captum.attr",
    reason=(
        "captum is not installed — Captum compatibility tests skipped.\n"
        "To run these tests: pip install captum"
    ),
)


# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlp_run(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = TinyMLP()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    return tmp_path, {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture()
def mlp_batch() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(8, 4)


@pytest.fixture()
def mlp_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(3)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_integrated_gradients_accepts_ni_state_dict(
    mlp_run: tuple[Path, dict[str, torch.Tensor]],
    mlp_batch: torch.Tensor,
) -> None:
    """col.to_state_dict(epoch) is loadable and usable with IntegratedGradients.

    No type conversion between NI and Captum — to_state_dict returns
    Dict[str, Tensor] which model.load_state_dict accepts directly.
    """
    IntegratedGradients = captum_attr.IntegratedGradients

    run_dir, _ = mlp_run
    col = ni_load(run_dir)

    model = TinyMLP()
    model.load_state_dict(col.to_state_dict(epoch=0))
    model.eval()

    ig = IntegratedGradients(model)
    inputs = mlp_batch.requires_grad_(True)
    attributions = ig.attribute(inputs, target=0)

    assert attributions.shape == inputs.shape
    assert attributions.dtype == inputs.dtype


def test_layer_activation_accepts_ni_loaded_model(
    mlp_run: tuple[Path, dict[str, torch.Tensor]],
    mlp_batch: torch.Tensor,
) -> None:
    """Captum LayerActivation runs on an NI-loaded model without conversion."""
    LayerActivation = captum_attr.LayerActivation

    run_dir, _ = mlp_run
    col = ni_load(run_dir)

    model = TinyMLP()
    model.load_state_dict(col.to_state_dict(epoch=0))
    model.eval()

    la = LayerActivation(model, model.fc1)
    activations = la.attribute(mlp_batch)

    # fc1: Linear(4→8), batch=8 → (8, 8)
    assert activations.shape == (8, 8)


def test_replay_activation_tensors_are_captum_compatible(
    mlp_run: tuple[Path, dict[str, torch.Tensor]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
    mlp_batch: torch.Tensor,
) -> None:
    """ReplayResult.activations values are plain torch.Tensor.

    Captum accepts plain tensors natively — no wrapper, no conversion needed.
    """
    IntegratedGradients = captum_attr.IntegratedGradients

    run_dir, _ = mlp_run

    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    ).run()

    for name, tensor in result.activations.items():
        assert type(tensor) is torch.Tensor, (
            f"Activation {name!r} is {type(tensor).__name__}, "
            "expected torch.Tensor. Captum requires plain tensors."
        )

    # Confirm the loaded model also works end-to-end with Captum
    col = ni_load(run_dir)
    model = TinyMLP()
    model.load_state_dict(col.to_state_dict(epoch=0))
    model.eval()

    ig = IntegratedGradients(model)
    attrs = ig.attribute(mlp_batch.requires_grad_(True), target=0)
    assert attrs.shape == (8, 4)
