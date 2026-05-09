"""Basic NeuroInquisitor feature showcase with a tiny MLP.

Demonstrates every implemented capability:
  • CapturePolicy  — capture parameters, buffers, and optimizer state
  • RunMetadata    — attach training provenance to the run
  • Snapshots      — weight + buffer checkpoints with per-epoch metadata
  • SnapshotCollection — by_epoch, by_layer, select, to_state_dict, to_numpy
  • ReplaySession  — activations, gradients, and logits via forward/backward hooks
  • Visualization  — weight-evolution video, replay figure, and loss curves

Run:
    python examples/basic_usage.py

Requires:
    pip install petname matplotlib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    RunMetadata,
    SnapshotCollection,
)
from basic_usage_utils import generate_visualizations


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Two-layer MLP for binary classification (4 features → 1 logit)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(device: torch.device) -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    X = torch.randn(512, 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()

    n_train = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:], y[n_train:]

    pin = device.type == "cuda"
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64,  shuffle=True,  pin_memory=pin)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=128, shuffle=False, pin_memory=pin)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    observer: NeuroInquisitor,
    device: torch.device,
    num_epochs: int,
) -> tuple[list[float], list[float]]:
    train_loss_history: list[float] = []
    test_loss_history:  list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        step = 0
        for step, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                test_loss_acc += loss_fn(model(X_batch), y_batch).item()

        avg_test_loss = test_loss_acc / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"  epoch {epoch:2d}  train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}")

        observer.snapshot(
            epoch=epoch,
            step=step,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")
    return train_loss_history, test_loss_history


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    snapshots: SnapshotCollection,
    run_dir: Path,
    test_loader: DataLoader,
    replay_modules: list[str],
    num_epochs: int,
    device: torch.device,
) -> list[dict[str, dict[str, np.ndarray]]]:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")
    print(f"  by_epoch(0) keys : {list(snapshots.by_epoch(0).keys())}")
    print(f"  by_layer('fc1.weight') epochs : {list(snapshots.by_layer('fc1.weight').keys())}")

    arrays = snapshots.to_numpy(epoch=0, layers=["fc1.weight"])
    print(f"  to_numpy(epoch=0, layers=['fc1.weight']) shape : {arrays['fc1.weight'].shape}")

    restored = TinyMLP().to(device)
    restored.load_state_dict(snapshots.to_state_dict(epoch=0), strict=False)
    print(f"  to_state_dict(epoch=0) → model restored (strict=False)")

    print("\n── ReplaySession ──")
    replay_history: list[dict[str, dict[str, np.ndarray]]] = []
    final_replay = None

    for epoch in range(num_epochs):
        final_replay = ReplaySession(
            run=run_dir,
            checkpoint=epoch,
            model_factory=TinyMLP,
            dataloader=test_loader,
            modules=replay_modules,
            capture=["activations", "gradients", "logits"],
            activation_reduction="pool",
            gradient_mode="aggregated",
        ).run()
        replay_history.append({
            "activations": final_replay.activations.to_numpy(),
            "gradients":   final_replay.gradients.to_numpy(),
        })

    print(f"  Samples replayed : {final_replay.metadata.n_samples}")
    print(f"  Logits shape     : {final_replay.logits.shape}")
    for name in replay_modules:
        print(f"  {name:6s} — activations {tuple(final_replay.activations[name].shape)}"
              f"  gradients {tuple(final_replay.gradients[name].shape)}")

    return replay_history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent / "outputs" / "basic_usage" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}\n")

    num_epochs     = 30
    replay_modules = ["fc1", "fc2"]

    train_loader, test_loader = load_data(device)

    model     = TinyMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn   = nn.BCEWithLogitsLoss()

    policy = CapturePolicy(
        capture_parameters=True,
        capture_buffers=True,
        capture_optimizer=True,
        replay_activations=True,
        replay_gradients=True,
    )
    run_meta = RunMetadata(
        training_config={"batch_size": 64, "lr": 1e-2},
        optimizer_class="Adam",
        device=str(device),
        model_class="TinyMLP",
    )
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=policy,
        run_metadata=run_meta,
    )

    train_losses, test_losses = train(
        model, optimizer, loss_fn,
        train_loader, test_loader, observer,
        device, num_epochs,
    )

    snapshots      = NeuroInquisitor.load(run_dir)
    replay_history = analyze(snapshots, run_dir, test_loader, replay_modules, num_epochs, device)
    weight_history = [snapshots.by_epoch(e) for e in range(num_epochs)]

    generate_visualizations(
        weight_history, replay_history, replay_modules,
        train_losses, test_losses, run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
