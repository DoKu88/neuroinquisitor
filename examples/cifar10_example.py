"""CIFAR-10 classification with weight tracking via NeuroInquisitor.

Trains a CNN on CIFAR-10, snapshots weights each epoch, then generates a video
showing how the filters evolve. conv1 filters are rendered as RGB colour patches
so you can watch edge and colour detectors form from random noise.

Run:
    python examples/cifar10_example.py

Requires:
    pip install tqdm petname torchvision matplotlib
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import NeuroInquisitor

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CIFAR10Net(nn.Module):
    """Three-conv-layer CNN for CIFAR-10 (32×32 RGB → 10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # 32 → 16 → 8 → 4  (three max-pool ops)
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
# Layers to show in the video
# ---------------------------------------------------------------------------

LAYERS_TO_PLOT = [
    ("conv1.weight", "Conv1 filters  (32×3×3×3)  — RGB"),
    ("conv2.weight", "Conv2 filters  (64×32×3×3) — ch-mean"),
    ("fc1.weight",   "FC1 weights    (256×2048)"),
    ("fc2.weight",   "FC2 weights    (10×256)"),
]


# ---------------------------------------------------------------------------
# Weight rendering helpers
# ---------------------------------------------------------------------------

def _render_rgb_filters(weight: np.ndarray) -> np.ndarray:
    """Tile (out, 3, H, W) filters as a grid of small RGB patches, normalised per-filter."""
    out_ch, _, H, W = weight.shape
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), 0.5)
    for i in range(out_ch):
        r, c = divmod(i, cols)
        filt = weight[i].transpose(1, 2, 0).copy()  # (H, W, 3)
        lo, hi = filt.min(), filt.max()
        filt = (filt - lo) / (hi - lo + 1e-8)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def _render_gray_filters(weight: np.ndarray) -> np.ndarray:
    """Tile (out, in, H, W) filters as a 2-D grid, averaged over input channels."""
    out_ch, _in_ch, H, W = weight.shape
    w = weight.mean(axis=1)
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad), np.nan)
    for i, filt in enumerate(w):
        r, c = divmod(i, cols)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def _preprocess(key: str, weight: np.ndarray) -> np.ndarray:
    if weight.ndim == 4 and weight.shape[1] == 3:
        return _render_rgb_filters(weight)
    if weight.ndim == 4:
        return _render_gray_filters(weight)
    return weight


# ---------------------------------------------------------------------------
# Video + loss curve
# ---------------------------------------------------------------------------

def _make_video(
    weight_history: list[dict[str, np.ndarray]],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    n_frames = len(weight_history)
    n_panels = len(LAYERS_TO_PLOT)

    rendered: list[list[np.ndarray]] = [
        [_preprocess(k, snap[k]) for k, _ in LAYERS_TO_PLOT]
        for snap in weight_history
    ]

    # Colour limits — fixed per layer; RGB panels handled separately
    vlims: list[tuple[float, float] | None] = []
    for i, (key, _) in enumerate(LAYERS_TO_PLOT):
        arr0 = rendered[0][i]
        if arr0.ndim == 3:  # RGB
            vlims.append(None)
        else:
            flat = np.concatenate([f[i].ravel() for f in rendered])
            abs_max = float(np.nanmax(np.abs(flat))) or 1.0
            vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    images = []
    for ax, (key, label), arr, vlim in zip(axes, LAYERS_TO_PLOT, rendered[0], vlims):
        if vlim is None:
            im = ax.imshow(arr, interpolation="nearest")
        else:
            im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                           vmin=vlim[0], vmax=vlim[1], interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(label, fontsize=8)
        images.append(im)

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        for im, arr in zip(images, rendered[frame]):
            im.set_data(arr)
        return [*images, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


def _save_loss_curves(
    train_losses: list[float],
    test_losses: list[float],
    out_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs, test_losses,  marker="s", label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("CIFAR-10 training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = petname.generate(words=2, separator="-")
    run_dir = Path(__file__).parent.parent / "outputs" / "CIFAR10_example" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).parent.parent / "outputs" / "CIFAR10_example" / "data"

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}")

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
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = CIFAR10Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    loss_fn = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(model, log_dir=run_dir, compress=True, create_new=True)

    num_epochs = 30
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    accuracy_history: list[float] = []

    print()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:02d}/{num_epochs}",
            unit="batch",
            leave=True,
        ) as pbar:
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
        correct = total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss += loss_fn(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        acc = correct / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(f"  → train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}  acc={acc:.1%}")
        observer.snapshot(
            epoch=epoch,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss, "accuracy": acc},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")

    col = NeuroInquisitor.load(run_dir)
    weight_history = [col.by_epoch(e) for e in range(num_epochs)]

    video_path = run_dir / "weights_over_time.mp4"
    print(f"\nGenerating video → {video_path} ...")
    result = _make_video(weight_history, accuracy_history, video_path, fps=4)
    print(f"Video saved: {result}")

    curves_path = run_dir / "loss_curves.png"
    _save_loss_curves(train_loss_history, test_loss_history, curves_path)
    print(f"Loss curves: {curves_path}")
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
