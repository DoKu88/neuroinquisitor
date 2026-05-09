"""CIFAR-10 + TorchLens: time-based neural network analysis with NeuroInquisitor.

Problem
-------
TorchLens is an outstanding tool for inspecting a model at a single moment in
time — it gives you per-operation activation tensors, gradient flow, and a full
computational graph.  But training is a *process*, not a moment.  How do
activations evolve?  When do neurons die?  How does gradient flow shift as the
model learns?  TorchLens alone can't answer these questions.

Solution
--------
NeuroInquisitor captures a full checkpoint every epoch.  For each checkpoint we
restore the weights into a fresh model and run TorchLens on a fixed probe batch.
The result is TorchLens-grade analysis at every training epoch — a film strip
instead of a single photo.

TorchLens features demonstrated
---------------------------------
  • log_forward_pass     — per-operation activation logging with gradient capture
  • model_log.log_backward   — gradient propagation captured inside ModelLog
  • visualization.show_model_graph   — computational graph saved as PDF
  • visualization.show_backward_graph — gradient-flow graph saved as PDF
  • model_log.to_pandas()     — layer metadata as a DataFrame (saved as CSV)
  • model_log.layer_list      — iterate every LayerPassLog for stats extraction
  • lyr.tensor / lyr.gradient — access activations and gradients per layer
  • lyr.module_address_normalized — map torchlens labels back to PyTorch modules

Outputs (in outputs/CIFAR10_TorchLens/<run-name>/)
----------------------------------------------------
  model_graph.pdf            — TorchLens forward computational graph
  backward_graph.pdf         — TorchLens gradient-flow graph (final epoch)
  layer_summary.csv          — per-operation metadata from to_pandas()
  torchlens_stats.png        — activation magnitude, gradient norm, and dead-
                               neuron fraction per module across all epochs
  activation_histograms.mp4  — animated per-layer activation distributions
  loss_curves.png            — train/test loss + test accuracy

Run
---
    python examples/torchlens_use_examples/torchlens_cifar10.py

Requires
--------
    pip install tqdm petname torchvision matplotlib torchlens
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
import torchlens as tl
import torchlens.visualization as tlv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import CapturePolicy, NeuroInquisitor, SnapshotCollection
from torchlens_cifar10_utils import generate_torchlens_visualizations


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_MODULES = ["conv1", "conv2", "conv3", "fc1", "fc2"]
PROBE_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CIFAR10Net(nn.Module):
    """Three-conv CNN for CIFAR-10 (32×32 RGB → 10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(3,   32, kernel_size=5, padding=2)
        self.conv2   = nn.Conv2d(32,  64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1     = nn.BatchNorm2d(32)
        self.bn2     = nn.BatchNorm2d(64)
        self.bn3     = nn.BatchNorm2d(128)
        self.fc1     = nn.Linear(128 * 4 * 4, 256)
        self.fc2     = nn.Linear(256, 10)

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


def load_data(
    data_dir: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Return train/test loaders plus a fixed probe batch for TorchLens analysis."""
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

    pin          = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin)

    # Fixed probe batch: same images every epoch so TorchLens comparisons are
    # apples-to-apples across checkpoints.
    probe_images, probe_labels = next(iter(
        DataLoader(test_ds, batch_size=PROBE_BATCH_SIZE, shuffle=False)
    ))

    return train_loader, test_loader, probe_images, probe_labels


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
        total         = 0
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
        acc           = correct_acc.item() / total
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
# TorchLens helpers
# ---------------------------------------------------------------------------


def save_architecture_graphs(
    model: nn.Module,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    loss_fn: nn.Module,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Save forward computation graph and backward gradient-flow graph as PDFs.

    The forward graph shows every tensor operation in the model.
    The backward graph shows how gradients propagate — which paths carry signal
    and which are dead ends.  Both use model weights from the final checkpoint.
    """
    images = probe_images.to(device)
    labels = probe_labels.to(device)
    model.eval()

    # Forward graph: no gradient capture needed.
    tlv.show_model_graph(
        model, images,
        vis_save_only=True,
        vis_outpath=str(run_dir / "model_graph"),
        vis_fileformat="pdf",
    )
    print(f"  Saved: model_graph.pdf")

    # Backward graph: use log_backward() so TorchLens can trace gradient flow.
    model_log = tl.log_forward_pass(model, images, save_gradients=True)
    loss = loss_fn(model_log.layer_list[-1].tensor, labels)
    model_log.log_backward(loss)

    tlv.show_backward_graph(
        model_log,
        vis_save_only=True,
        vis_outpath=str(run_dir / "backward_graph"),
        vis_fileformat="pdf",
    )
    print(f"  Saved: backward_graph.pdf")

    # Layer metadata table — useful for debugging architectures.
    df = model_log.to_pandas()
    useful_cols = [
        "layer_label", "layer_type", "tensor_shape", "func_name",
        "num_params_total", "modules_entered", "modules_exited",
        "min_distance_from_input", "max_distance_from_output",
        "is_input_layer", "is_output_layer",
    ]
    df[useful_cols].to_csv(run_dir / "layer_summary.csv", index=False)
    print(f"  Saved: layer_summary.csv  ({len(df)} operations logged)")


def extract_epoch_data(
    model: nn.Module,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    loss_fn: nn.Module,
    target_modules: list[str],
    device: torch.device,
) -> dict[str, dict]:
    """Run TorchLens on one checkpoint and return per-module stats + activations.

    Pattern
    -------
    log_forward_pass captures every operation.  log_backward() then propagates
    gradients back through the recorded graph, populating lyr.gradient on every
    layer that sits on the backward path.

    We filter to ``target_modules`` using lyr.module_address_normalized, which
    maps TorchLens internal labels (e.g. 'conv2d_1_1') back to the user-facing
    PyTorch module names (e.g. 'conv1').
    """
    images = probe_images.to(device)
    labels = probe_labels.to(device)
    model.eval()

    model_log = tl.log_forward_pass(
        model,
        images,
        layers_to_save=target_modules,   # only save activations for our modules
        save_gradients=True,
    )
    loss = loss_fn(model_log.layer_list[-1].tensor, labels)
    model_log.log_backward(loss)

    epoch_data: dict[str, dict] = {}
    for lyr in model_log.layer_list:
        if not lyr.has_saved_activations:
            continue
        mod = lyr.module_address_normalized
        if mod not in target_modules:
            continue

        act  = lyr.tensor.detach().cpu()
        flat = act.flatten()

        grad_stats: dict[str, float] = {"mean_grad": 0.0, "grad_norm": 0.0}
        grad_flat: np.ndarray | None = None
        if lyr.has_gradient:
            grad     = lyr.gradient.detach().cpu()
            grad_flat = grad.abs().flatten().numpy()
            grad_stats = {
                "mean_grad": float(grad.abs().mean()),
                "grad_norm": float(grad.norm()),
            }

        epoch_data[mod] = {
            # Scalar stats for line plots
            "mean_act":  float(flat.abs().mean()),
            "std_act":   float(flat.std()),
            "sparsity":  float((flat.abs() < 1e-4).float().mean()),  # dead-neuron proxy
            **grad_stats,
            # Raw arrays for animated histograms
            "act_flat":  flat.numpy(),
            "grad_flat": grad_flat,
        }

    return epoch_data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    snapshots: SnapshotCollection,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    loss_fn: nn.Module,
    run_dir: Path,
    num_epochs: int,
    device: torch.device,
) -> list[dict[str, dict]]:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")

    # Architecture graphs use the final trained weights.
    print("\n── TorchLens Architecture Graphs ──")
    final_model = CIFAR10Net().to(device)
    final_model.load_state_dict(snapshots.to_state_dict(num_epochs - 1))
    save_architecture_graphs(final_model, probe_images, probe_labels, loss_fn, run_dir, device)

    # Per-epoch TorchLens analysis: restore each checkpoint → extract stats.
    print("\n── TorchLens Per-Epoch Replay ──")
    epoch_history: list[dict[str, dict]] = []

    for epoch in tqdm(range(num_epochs), desc="  TorchLens replay", unit="ckpt"):
        model = CIFAR10Net().to(device)
        model.load_state_dict(snapshots.to_state_dict(epoch))
        epoch_data = extract_epoch_data(
            model, probe_images, probe_labels, loss_fn, TARGET_MODULES, device,
        )
        epoch_history.append(epoch_data)

    # Print a summary table for the final epoch.
    print(f"\n  Final epoch ({num_epochs - 1}) per-module summary:")
    print(f"  {'Module':8s}  {'Mean|act|':>10s}  {'Std(act)':>10s}  {'Sparsity':>10s}  {'Mean|grad|':>12s}  {'‖grad‖':>10s}")
    for mod in TARGET_MODULES:
        d = epoch_history[-1].get(mod, {})
        if not d:
            continue
        print(
            f"  {mod:8s}"
            f"  {d['mean_act']:10.4f}"
            f"  {d['std_act']:10.4f}"
            f"  {d['sparsity']:10.4f}"
            f"  {d['mean_grad']:12.6f}"
            f"  {d['grad_norm']:10.4f}"
        )

    return epoch_history


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
    run_dir  = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_TorchLens" / run_name
    data_dir = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_TorchLens" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 10

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}\n")

    train_loader, test_loader, probe_images, probe_labels = load_data(data_dir, device)

    model     = CIFAR10Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn   = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=CapturePolicy(capture_parameters=True, capture_buffers=True),
    )

    train_losses, test_losses, accuracy_history = train(
        model, optimizer, scheduler, loss_fn,
        train_loader, test_loader, observer,
        device, num_epochs,
    )

    snapshots     = NeuroInquisitor.load(run_dir)
    epoch_history = analyze(
        snapshots, probe_images, probe_labels, loss_fn,
        run_dir, num_epochs, device,
    )

    generate_torchlens_visualizations(
        epoch_history, TARGET_MODULES,
        accuracy_history, train_losses, test_losses,
        run_dir,
    )

    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
