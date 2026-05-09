"""MNIST classification — full NeuroInquisitor feature showcase.

Demonstrates every implemented capability:
  • CapturePolicy  — capture parameters, buffers, and optimizer state
  • RunMetadata    — attach training provenance to the run
  • Snapshots      — weight + buffer checkpoints with per-epoch metadata
  • SnapshotCollection — by_epoch, by_layer, select, to_state_dict, to_numpy
  • ReplaySession  — activations, gradients, and logits via forward/backward hooks
  • Visualization  — weight-evolution video, replay figure, and loss curves

Run:
    python examples/mnist_example.py

Requires:
    pip install tqdm petname torchvision matplotlib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import petname
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    RunMetadata,
    SnapshotCollection,
)
from mnist_example_utils import generate_visualizations


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MNISTNet(nn.Module):
    """Small CNN: two conv layers + two FC layers, ~200k parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)    # (8, 26, 26)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   # (16, 11, 11) after pool
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # (8, 13, 13)
        x = self.pool(self.relu(self.conv2(x)))  # (16, 5, 5)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(
    data_dir: Path,
    device: torch.device,
    train_batch_size: int,
    test_batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,  num_workers=2, pin_memory=pin, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size,  shuffle=False, num_workers=2, pin_memory=pin, persistent_workers=True)
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
) -> tuple[list[float], list[float], list[float]]:
    train_loss_history: list[float] = []
    test_loss_history:  list[float] = []
    accuracy_history:   list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{num_epochs}", unit="batch", leave=True) as pbar:
            for step, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        correct = total = 0
        test_loss_acc = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total   += labels.size(0)

        avg_test_loss = test_loss_acc / len(test_loader)
        acc = correct / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(f"  → train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}  acc={acc:.1%}")

        observer.snapshot(
            epoch=epoch,
            step=step,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss, "accuracy": acc},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")
    return train_loss_history, test_loss_history, accuracy_history


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
    print(f"  by_epoch(0) keys : {list(snapshots.by_epoch(0).keys())[:4]} …")
    print(f"  by_layer('conv1.weight') epochs : {list(snapshots.by_layer('conv1.weight').keys())}")

    arrays = snapshots.to_numpy(epoch=0, layers=["conv1.weight"])
    print(f"  to_numpy(epoch=0, layers=['conv1.weight']) shape : {arrays['conv1.weight'].shape}")

    restored = MNISTNet().to(device)
    restored.load_state_dict(snapshots.to_state_dict(epoch=0), strict=False)
    print(f"  to_state_dict(epoch=0) → model restored (strict=False)")

    print("\n── ReplaySession ──")
    replay_history: list[dict[str, dict[str, np.ndarray]]] = []
    final_replay = None

    for epoch in tqdm(range(num_epochs), desc="  Replaying checkpoints", unit="ckpt", leave=True):
        final_replay = ReplaySession(
            run=run_dir,
            checkpoint=epoch,
            model_factory=MNISTNet,
            dataloader=test_loader,
            modules=replay_modules,
            capture=["activations", "gradients", "logits"],
            activation_reduction="pool",
            gradient_mode="aggregated",
            dataset_slice=lambda samples: samples[:128],
            slice_metadata={"description": "first 128 test samples"},
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
    cfg_path = Path(__file__).parent / "configs" / "mnist_example.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent / "outputs" / "MNIST_example" / run_name
    data_dir = Path(__file__).parent.parent / "outputs" / "MNIST_example" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}\n")

    num_epochs     = cfg["num_epochs"]
    replay_modules = cfg["replay_modules"]
    lr             = cfg["lr"]

    train_loader, test_loader = load_data(
        data_dir, device, cfg["train_batch_size"], cfg["test_batch_size"]
    )

    model     = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    policy = CapturePolicy(
        capture_parameters=True,
        capture_buffers=True,
        capture_optimizer=True,
        replay_activations=True,
        replay_gradients=True,
    )
    run_meta = RunMetadata(
        training_config={"batch_size": cfg["train_batch_size"], "lr": lr},
        optimizer_class="Adam",
        device=str(device),
        model_class="MNISTNet",
    )
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=policy,
        run_metadata=run_meta,
    )

    train_losses, test_losses, accuracy_history = train(
        model, optimizer, loss_fn,
        train_loader, test_loader, observer,
        device, num_epochs,
    )

    snapshots      = NeuroInquisitor.load(run_dir)
    replay_history = analyze(snapshots, run_dir, test_loader, replay_modules, num_epochs, device)
    weight_history = [snapshots.by_epoch(e) for e in range(num_epochs)]

    generate_visualizations(
        weight_history, replay_history, replay_modules,
        accuracy_history, train_losses, test_losses, run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
