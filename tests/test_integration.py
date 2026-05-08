"""Integration tests: full training loop with NeuroInquisitor."""

from __future__ import annotations

from pathlib import Path

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
    torch.manual_seed(42)
    X = torch.randn(n, 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()
    return X, y


def test_full_training_loop_with_observer(tmp_path: Path) -> None:
    """Train for several epochs, snapshot each one, verify storage and round-trip."""
    model = TinyMLP()
    X, y = make_toy_dataset()

    observer = NeuroInquisitor(model, log_dir=tmp_path, compress=True, create_new=True)
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

    # Verify snapshot files exist on disk.
    for epoch in range(num_epochs):
        assert (tmp_path / f"epoch_{epoch:04d}.safetensors").exists()

    # Verify index records all epochs.
    col = NeuroInquisitor.load(tmp_path)
    assert col.epochs == list(range(num_epochs))

    # End-to-end round-trip: values must match what the model had at that epoch.
    for epoch in range(num_epochs):
        loaded = col.by_epoch(epoch)
        for name, original_arr in epoch_weights[epoch].items():
            np.testing.assert_allclose(
                loaded[name], original_arr, rtol=1e-5,
                err_msg=f"Value mismatch at epoch {epoch}, param {name}",
            )


def test_weights_change_across_epochs(tmp_path: Path) -> None:
    """Weights stored at different epochs must actually differ."""
    model = TinyMLP()
    X, y = make_toy_dataset()

    observer = NeuroInquisitor(model, log_dir=tmp_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)

    col = observer.load_all_snapshots()
    observer.close()

    snap0 = col.by_epoch(0)
    snap2 = col.by_epoch(2)
    any_changed = any(not np.allclose(snap0[n], snap2[n]) for n in snap0)
    assert any_changed, "Weights did not change between epoch 0 and epoch 2"


def test_by_layer_parallel_read(tmp_path: Path) -> None:
    """by_layer should return correct values read in parallel."""
    model = TinyMLP()
    X, y = make_toy_dataset()
    observer = NeuroInquisitor(model, log_dir=tmp_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    saved: dict[int, np.ndarray] = {}
    for epoch in range(4):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        saved[epoch] = model.fc1.weight.detach().cpu().numpy().copy()
        observer.snapshot(epoch=epoch)

    col = observer.load_all_snapshots()
    observer.close()

    by_epoch = col.by_layer("fc1.weight", max_workers=4)
    for epoch, arr in saved.items():
        np.testing.assert_allclose(by_epoch[epoch], arr, rtol=1e-5)


def test_import_surface() -> None:
    """Public API is importable and complete."""
    from neuroinquisitor import (
        Backend,
        Format,
        Index,
        IndexEntry,
        JSONIndex,
        LocalBackend,
        NeuroInquisitor,
        SafeTensorsFormat,
        SnapshotCollection,
        __version__,
    )

    assert isinstance(__version__, str)
    assert hasattr(NeuroInquisitor, "snapshot")
    assert hasattr(NeuroInquisitor, "load_snapshot")
    assert hasattr(NeuroInquisitor, "load_all_snapshots")
    assert hasattr(NeuroInquisitor, "load")
    assert hasattr(NeuroInquisitor, "close")
