"""Integration test: train a small network on a toy dataset with NeuroInquisitor."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def make_toy_dataset(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary classification: label = 1 if sum(x) > 0 else 0."""
    torch.manual_seed(42)
    X = torch.randn(n, 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()
    return X, y


def test_full_training_loop_with_observer(tmp_path: Path) -> None:
    """Train for several epochs, snapshot each one, then verify storage."""
    model = TinyMLP()
    X, y = make_toy_dataset()

    observer = NeuroInquisitor(
        model,
        log_dir=tmp_path,
        filename="training.h5",
        compress=True,
        create_new=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    num_epochs = 5
    epoch_weights: dict[int, dict[str, np.ndarray]] = {}

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        observer.snapshot(epoch=epoch, metadata={"loss": loss.detach().item()})
        epoch_weights[epoch] = {
            name: param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
        }

    observer.close()

    # --- verify HDF5 structure ---
    with h5py.File(tmp_path / "training.h5", "r") as f:
        for epoch in range(num_epochs):
            key = f"epoch_{epoch:04d}"
            assert key in f, f"Missing group {key}"
            grp = f[key]
            assert grp.attrs["epoch"] == epoch
            assert "loss" in grp.attrs

    # --- verify load_snapshot round-trips values ---
    observer2 = NeuroInquisitor(
        model,
        log_dir=tmp_path,
        filename="training.h5",
        create_new=False,
    )
    for epoch in range(num_epochs):
        loaded = observer2.load_snapshot(epoch=epoch)
        for name, original_arr in epoch_weights[epoch].items():
            np.testing.assert_allclose(
                loaded[name], original_arr, rtol=1e-5,
                err_msg=f"Value mismatch at epoch {epoch}, param {name}",
            )
    observer2.close()


def test_weights_change_across_epochs(tmp_path: Path) -> None:
    """Weights stored at different epochs must actually differ (training is happening)."""
    model = TinyMLP()
    X, y = make_toy_dataset()

    observer = NeuroInquisitor(model, log_dir=tmp_path, filename="w.h5")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)

    snap0 = observer.load_snapshot(epoch=0)
    snap2 = observer.load_snapshot(epoch=2)
    observer.close()

    # At least one parameter must have changed
    any_changed = any(
        not np.allclose(snap0[name], snap2[name])
        for name in snap0
    )
    assert any_changed, "Weights did not change between epoch 0 and epoch 2"


def test_import_surface() -> None:
    """Public API is importable and complete."""
    from neuroinquisitor import NeuroInquisitor, __version__

    assert isinstance(__version__, str)
    assert hasattr(NeuroInquisitor, "snapshot")
    assert hasattr(NeuroInquisitor, "load_snapshot")
    assert hasattr(NeuroInquisitor, "close")
