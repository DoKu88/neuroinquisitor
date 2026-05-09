"""CIFAR-10 + Captum integration with NeuroInquisitor.

Demonstrates NeuroInquisitor's out-of-the-box compatibility with Captum
via two integration paths:

  Path A  col.to_state_dict(epoch) → model.load_state_dict() → Captum
  Path B  ReplaySession.run() → result.activations (plain tensors) → Captum

Attribution methods shown:
  1. IntegratedGradients — per-pixel input attribution.
  2. LayerGradCam — spatial importance map on conv2.

Outputs (in outputs/CIFAR10_Captum/<run-name>/):
  attribution_evolution.mp4  — IG + GradCAM animated across training epochs.
  final_attributions.png     — per-class attribution panel at the final epoch.
  replay_activations.txt     — activation shapes captured via ReplaySession.
  loss_curves.png            — train/test loss + accuracy.

Run:
    python examples/captum_use_examples/cifar10_captum.py

Requires:
    pip install tqdm petname torchvision matplotlib captum
"""

from __future__ import annotations

import torch.utils.data
from pathlib import Path

import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    SnapshotCollection,
)

from cifar10_captum_utils import (
    compute_attributions_per_epoch,
    generate_captum_visualizations,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CIFAR10Net(nn.Module):
    """Three-conv-layer CNN for CIFAR-10 (32×32 RGB → 10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x.flatten(1))
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _one_per_class(
    dataset: torch.utils.data.Dataset,
    n_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first occurrence of each class as (images, labels)."""
    images, labels = [], []
    seen: set[int] = set()
    for img, label in dataset:
        if label not in seen:
            seen.add(label)
            images.append(img)
            labels.append(label)
        if len(seen) == n_classes:
            break
    imgs_t = torch.stack(images)
    order = torch.tensor(labels).argsort()
    return imgs_t[order], torch.tensor(labels)[order]


def load_data(
    data_dir: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=pin, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin, persistent_workers=True)

    # Fixed attribution batch: one normalised image per class, always kept on CPU.
    attr_images, attr_labels = _one_per_class(test_ds)
    return train_loader, test_loader, attr_images, attr_labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
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
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total = 0
        test_loss_acc = torch.zeros(1, device=device)
        correct_acc   = torch.zeros(1, device=device, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels)
                correct_acc   += (out.argmax(1) == labels).sum()
                total         += labels.size(0)

        avg_test_loss = test_loss_acc.item() / len(test_loader)
        acc = correct_acc.item() / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(f"  → train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}  acc={acc:.1%}")

        observer.snapshot(
            epoch=epoch,
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
    attr_images: torch.Tensor,
    attr_labels: torch.Tensor,
    num_epochs: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")

    print("\n── Path A  col.to_state_dict() → Captum ──")
    print("  Each NI checkpoint: load weights into fresh model → IntegratedGradients + LayerGradCam")
    ig_mag_grids, gc_grids, ig_signed_grids = compute_attributions_per_epoch(
        snapshots, CIFAR10Net, attr_images, attr_labels, n_steps=30,
    )
    print(f"  Attributions computed for {len(snapshots.epochs)} epoch(s).")

    print("\n── Path B  ReplaySession → plain tensors → Captum ──")
    final_epoch = num_epochs - 1
    attr_loader = DataLoader(
        torch.utils.data.TensorDataset(attr_images, attr_labels),
        batch_size=len(attr_images),
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )
    result = ReplaySession(
        run=run_dir,
        checkpoint=final_epoch,
        model_factory=CIFAR10Net,
        dataloader=attr_loader,
        modules=["conv2", "fc1"],
        capture=["activations"],
    ).run()

    print(f"  Samples replayed : {result.metadata.n_samples}")
    for name, tensor in result.activations.items():
        assert type(tensor) is torch.Tensor, (
            f"Activation {name!r} is {type(tensor).__name__}; expected torch.Tensor"
        )
        print(f"  {name:12s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print("  All activation values are plain torch.Tensor — Captum accepts them directly.")

    replay_path = run_dir / "replay_activations.txt"
    lines = [
        f"ReplaySession — epoch {final_epoch} activations",
        f"  n_samples : {result.metadata.n_samples}",
        "",
        *(
            f"  {name:12s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}"
            for name, tensor in result.activations.items()
        ),
        "",
        "All activation values are plain torch.Tensor — Captum accepts them directly.",
        "Example: feed conv2 activations into IntegratedGradients as additional_forward_args.",
    ]
    replay_path.write_text("\n".join(lines))
    print(f"  Replay log saved : {replay_path.name}")

    return ig_mag_grids, gc_grids, ig_signed_grids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    # Captum gradient-based methods are most reliable on CPU.
    # Training uses the fastest available device; attribution always runs on CPU.
    train_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_Captum" / run_name
    data_dir = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_Captum" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name      : {run_name}")
    print(f"Run dir       : {run_dir}/")
    print(f"Train device  : {train_device}")
    print(f"Captum device : cpu\n")

    num_epochs = 2

    train_loader, test_loader, attr_images, attr_labels = load_data(data_dir, train_device)

    model     = CIFAR10Net().to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    loss_fn   = nn.CrossEntropyLoss()

    # capture_buffers=True includes BatchNorm running stats so to_state_dict()
    # produces a complete state dict loadable with strict=True.
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=CapturePolicy(capture_buffers=True),
    )

    train_losses, test_losses, accuracy_history = train(
        model, optimizer, scheduler, loss_fn,
        train_loader, test_loader, observer,
        train_device, num_epochs,
    )

    snapshots = NeuroInquisitor.load(run_dir)
    ig_mag_grids, gc_grids, ig_signed_grids = analyze(
        snapshots, run_dir, attr_images, attr_labels, num_epochs,
    )

    generate_captum_visualizations(
        snapshots, CIFAR10Net, attr_images, attr_labels,
        ig_mag_grids, gc_grids, ig_signed_grids,
        accuracy_history, train_losses, test_losses,
        run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
