"""CIFAR-10 classification with weight tracking via NeuroInquisitor.

Trains a CNN on CIFAR-10, snapshots weights each epoch, then generates a video
with four panels:
  1. Conv1 RGB filters — fixed per-filter bounds so weight growth is visible.
  2. Conv2 per-filter ‖W‖₂ heatmap — sign-cancellation-free.
  3. FC1 neuron row-norms over time — growing (256 × epochs) heatmap.
  4. Conv1 |W_t − W_0| delta — visibly grows each epoch.

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
        # 5×5 first conv: large enough for edge/colour detectors to be readable
        self.conv1 = nn.Conv2d(3,  32, kernel_size=5, padding=2)
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
# Weight rendering helpers
# ---------------------------------------------------------------------------

def _compute_conv1_filter_lims(
    weight_history: list[dict[str, np.ndarray]],
) -> list[tuple[float, float]]:
    """Fixed (lo, hi) per filter across ALL epochs so growth is visible."""
    stacked = np.stack([snap["conv1.weight"] for snap in weight_history])  # (T, 32, 3, H, W)
    return [
        (float(stacked[:, i].min()), float(stacked[:, i].max()))
        for i in range(stacked.shape[1])
    ]


def _render_rgb_filters(
    weight: np.ndarray,
    per_filter_lims: list[tuple[float, float]],
) -> np.ndarray:
    """Tile (out, 3, H, W) filters as an RGB grid with fixed per-filter bounds."""
    out_ch, _, H, W = weight.shape
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, 3), 0.5)
    for i in range(out_ch):
        r, c = divmod(i, cols)
        filt = weight[i].transpose(1, 2, 0).copy()  # (H, W, 3)
        lo, hi = per_filter_lims[i]
        filt = np.clip((filt - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def _render_delta_rgb(
    weight: np.ndarray,
    weight_0: np.ndarray,
    vmax: float,
) -> np.ndarray:
    """Tile |W_t − W_0| per filter as an RGB grid normalised by vmax."""
    delta = np.abs(weight - weight_0)
    out_ch, _, H, W = delta.shape
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad, 3))
    for i in range(out_ch):
        r, c = divmod(i, cols)
        filt = np.clip(delta[i].transpose(1, 2, 0) / (vmax + 1e-8), 0.0, 1.0)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def _render_conv2_norm_heatmap(weight: np.ndarray) -> np.ndarray:
    """Per-filter ‖W‖₂ for (64, in, H, W) as an 8×8 grid."""
    norms = np.sqrt((weight ** 2).sum(axis=(1, 2, 3)))  # (out_ch,)
    side = math.ceil(math.sqrt(len(norms)))
    padded = np.full(side * side, np.nan)
    padded[: len(norms)] = norms
    return padded.reshape(side, side)


def _build_fc1_timeline(weight_history: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Return (256, num_epochs) matrix of per-neuron row-norms."""
    row_norms = np.array([
        np.linalg.norm(snap["fc1.weight"], axis=1)
        for snap in weight_history
    ])  # (num_epochs, 256)
    return row_norms.T  # (256, num_epochs)


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
    weight_0 = weight_history[0]

    # Fixed colour limits computed once across all frames
    conv1_lims = _compute_conv1_filter_lims(weight_history)

    conv2_heatmaps = [_render_conv2_norm_heatmap(snap["conv2.weight"]) for snap in weight_history]
    conv2_vmax = float(max(float(h[~np.isnan(h)].max()) for h in conv2_heatmaps))

    fc1_timeline = _build_fc1_timeline(weight_history)  # (256, num_epochs)
    fc1_vmax = float(fc1_timeline.max())

    delta_vmax = float(max(
        np.abs(snap["conv1.weight"] - weight_0["conv1.weight"]).max()
        for snap in weight_history
    ))

    conv1_rendered = [_render_rgb_filters(snap["conv1.weight"], conv1_lims) for snap in weight_history]
    delta_rendered = [_render_delta_rgb(snap["conv1.weight"], weight_0["conv1.weight"], delta_vmax) for snap in weight_history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    # Panel 0: Conv1 RGB filters (fixed per-filter bounds)
    im_conv1 = axes[0].imshow(conv1_rendered[0], interpolation="nearest")
    axes[0].set_title("Conv1 — RGB filters (fixed bounds)", fontsize=8)
    axes[0].axis("off")

    # Panel 1: Conv2 per-filter ‖W‖₂ heatmap
    im_conv2 = axes[1].imshow(conv2_heatmaps[0], cmap="viridis", vmin=0, vmax=conv2_vmax, interpolation="nearest")
    fig.colorbar(im_conv2, ax=axes[1], shrink=0.8, label="‖W‖₂")
    axes[1].set_title("Conv2 — per-filter ‖W‖₂", fontsize=8)
    axes[1].axis("off")

    # Panel 2: FC1 row-norm timeline — grows one column per epoch
    fc1_init = np.full_like(fc1_timeline, np.nan)
    fc1_init[:, 0] = fc1_timeline[:, 0]
    im_fc1 = axes[2].imshow(
        fc1_init, aspect="auto", cmap="plasma",
        vmin=0, vmax=fc1_vmax, interpolation="nearest",
    )
    fig.colorbar(im_fc1, ax=axes[2], shrink=0.8, label="‖row‖₂")
    axes[2].set_title("FC1 — neuron row-norms over epochs", fontsize=8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Neuron index")

    # Panel 3: Conv1 delta from init
    im_delta = axes[3].imshow(delta_rendered[0], interpolation="nearest")
    axes[3].set_title("Conv1  |W_t − W_0|  (init delta)", fontsize=8)
    axes[3].axis("off")

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")

        im_conv1.set_data(conv1_rendered[frame])
        im_conv2.set_data(conv2_heatmaps[frame])

        fc1_data = np.full_like(fc1_timeline, np.nan)
        fc1_data[:, : frame + 1] = fc1_timeline[:, : frame + 1]
        im_fc1.set_data(fc1_data)

        im_delta.set_data(delta_rendered[frame])
        return [im_conv1, im_conv2, im_fc1, im_delta, title_text]

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
    accuracy_history: list[float],
    out_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax_loss.plot(epochs, train_losses, marker="o", label="Train loss")
    ax_loss.plot(epochs, test_losses,  marker="s", label="Test loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("CIFAR-10 — loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in accuracy_history], marker="o", color="tab:green")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_title("CIFAR-10 — test accuracy")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin)

    model = CIFAR10Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    loss_fn = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(model, log_dir=run_dir, compress=True, create_new=True)

    num_epochs = 60
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
        total = 0
        test_loss_acc = torch.zeros(1, device=device)
        correct_acc   = torch.zeros(1, device=device, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels)
                correct_acc   += (out.argmax(1) == labels).sum()
                total += labels.size(0)

        avg_test_loss = (test_loss_acc.item() / len(test_loader))
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

    col = NeuroInquisitor.load(run_dir)
    weight_history = [col.by_epoch(e) for e in range(num_epochs)]

    video_path = run_dir / "weights_over_time.mp4"
    print(f"\nGenerating video → {video_path} ...")
    result = _make_video(weight_history, accuracy_history, video_path, fps=4)
    print(f"Video saved: {result}")

    curves_path = run_dir / "loss_curves.png"
    _save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"Loss curves: {curves_path}")
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
