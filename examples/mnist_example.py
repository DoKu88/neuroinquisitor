"""MNIST classification with weight tracking via NeuroInquisitor.

Trains a small CNN on MNIST, snapshots weights after each epoch,
then generates a video showing the weight evolution over training.

Run:
    python examples/mnist_example.py

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
# Layers shown in the video
# ---------------------------------------------------------------------------

LAYERS_TO_PLOT = [
    ("conv1.weight", "Conv1  (8×1×3×3)"),
    ("conv2.weight", "Conv2  (16×8×3×3)"),
    ("fc1.weight",   "FC1   (128×400)"),
    ("fc2.weight",   "FC2   (10×128)"),
]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _render_conv_filters(weight: np.ndarray) -> np.ndarray:
    """Tile conv filters into a 2-D grid. weight: (out, in, H, W)."""
    out_ch, _in_ch, H, W = weight.shape
    w = weight.mean(axis=1)  # average over input channels → (out, H, W)
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    canvas = np.full((rows * (H + 1) - 1, cols * (W + 1) - 1), np.nan)
    for i, filt in enumerate(w):
        r, c = divmod(i, cols)
        canvas[r * (H + 1): r * (H + 1) + H, c * (W + 1): c * (W + 1) + W] = filt
    return canvas


def _make_video(
    weight_history: list[dict[str, np.ndarray]],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate weight heatmaps over epochs; returns the path actually written."""
    n_frames = len(weight_history)
    n_panels = len(LAYERS_TO_PLOT)

    # Pre-process every frame so animation update is cheap
    rendered: list[list[np.ndarray]] = []
    for snap in weight_history:
        frame = []
        for key, _ in LAYERS_TO_PLOT:
            w = snap[key]
            frame.append(_render_conv_filters(w) if w.ndim == 4 else w)
        rendered.append(frame)

    # Fixed symmetric colour scale per layer so changes are clearly visible
    vlims: list[tuple[float, float]] = []
    for i in range(n_panels):
        flat = np.concatenate([f[i].ravel() for f in rendered])
        abs_max = float(np.nanmax(np.abs(flat))) or 1.0
        vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    images = []
    for ax, (_, label), arr, (vmin, vmax) in zip(axes, LAYERS_TO_PLOT, rendered[0], vlims):
        im = ax.imshow(
            arr, aspect="auto", cmap="RdBu_r",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(label, fontsize=9)
        images.append(im)

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        for im, arr in zip(images, rendered[frame]):
            im.set_data(arr)
        return [*images, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=False,
    )

    # Prefer MP4; fall back to GIF if ffmpeg is unavailable
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(str(out_path), writer=writer)
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
    """Save a train/test loss curve image."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs, test_losses,  marker="s", label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("MNIST training curves")
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
    run_dir = Path(__file__).parent.parent / "outputs" / "MNIST_example" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Shared MNIST download cache across runs
    data_dir = Path(__file__).parent.parent / "outputs" / "MNIST_example" / "data"

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(model, log_dir=run_dir, compress=True, create_new=True)

    num_epochs = 100
    accuracy_history: list[float] = []
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []

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
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
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

    # --- load all snapshots and generate video ---
    col = NeuroInquisitor.load(run_dir)
    weight_history = [col.by_epoch(e) for e in range(num_epochs)]

    video_path = run_dir / "weights_over_time.mp4"
    print(f"\nGenerating video → {video_path} ...")
    result = _make_video(weight_history, accuracy_history, video_path, fps=3)
    print(f"Video saved: {result}")

    curves_path = run_dir / "loss_curves.png"
    _save_loss_curves(train_loss_history, test_loss_history, curves_path)
    print(f"Loss curves: {curves_path}")
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
