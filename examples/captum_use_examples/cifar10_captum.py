"""CIFAR-10 + Captum integration with NeuroInquisitor.

Trains a CNN on CIFAR-10, snapshots weights each epoch with NI, then uses
Captum to analyse attributions at every saved checkpoint.  Demonstrates
two NI→Captum integration paths:

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
    python examples/captum/cifar10_captum.py

Requires:
    pip install tqdm petname torchvision matplotlib captum
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
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from captum.attr import IntegratedGradients, LayerGradCam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.loader import load as ni_load
from neuroinquisitor.replay import ReplaySession
from neuroinquisitor.schema import CapturePolicy

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Model (identical to cifar10_example.py for comparability)
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
# Data helpers
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
    # Sort by class index so the grid is always in class order.
    order = torch.tensor(labels).argsort()
    return imgs_t[order], torch.tensor(labels)[order]


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) normalised tensor → (H, W, 3) float32 in [0, 1]."""
    return (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _tile_rgb(imgs: list[np.ndarray], cols: int, pad: int = 1) -> np.ndarray:
    """Tile a list of (H, W, 3) arrays into a (rows*H, cols*W, 3) canvas."""
    rows = math.ceil(len(imgs) / cols)
    H, W, C = imgs[0].shape
    canvas = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad, C), 0.5)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = img
    return canvas


def _tile_maps(maps: np.ndarray, cols: int, pad: int = 1) -> np.ndarray:
    """Tile (N, H, W) maps into a (rows*H, cols*W) canvas."""
    N, H, W = maps.shape
    rows = math.ceil(N / cols)
    canvas = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad))
    for i in range(N):
        r, c = divmod(i, cols)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = maps[i]
    return canvas


def _normalise_per_sample(maps: np.ndarray) -> np.ndarray:
    """Normalise each (H, W) map in (N, H, W) to [0, 1] independently."""
    out = np.empty_like(maps, dtype=np.float32)
    for i in range(maps.shape[0]):
        lo, hi = maps[i].min(), maps[i].max()
        out[i] = (maps[i] - lo) / (hi - lo + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Captum helpers — all use CPU; model must be in eval() before calling
# ---------------------------------------------------------------------------

def _ig_magnitude(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 30,
) -> np.ndarray:
    """IntegratedGradients → (N, H, W) attribution magnitude (|Δ| summed over channels)."""
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)
    attrs = ig.attribute(inputs, baselines=baseline, target=targets, n_steps=n_steps)
    return attrs.abs().sum(dim=1).detach().cpu().numpy()  # (N, H, W)


def _ig_signed(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 30,
) -> np.ndarray:
    """IntegratedGradients → (N, H, W) signed attribution (mean over channels)."""
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)
    attrs = ig.attribute(inputs, baselines=baseline, target=targets, n_steps=n_steps)
    return attrs.mean(dim=1).detach().cpu().numpy()  # (N, H, W)


def _gradcam(
    model: nn.Module,
    layer: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    out_size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """LayerGradCam → (N, H, W) map, clamped ≥ 0, upsampled to out_size."""
    lgc = LayerGradCam(model, layer)
    gc = lgc.attribute(inputs, target=targets)  # (N, C, h, w)
    if gc.shape[1] > 1:
        gc = gc.sum(dim=1, keepdim=True)
    gc_up = F.interpolate(gc.float(), size=out_size, mode="bilinear", align_corners=False)
    return gc_up.squeeze(1).clamp(min=0).detach().cpu().numpy()  # (N, H, W)


# ---------------------------------------------------------------------------
# Integration path A: col.to_state_dict(epoch) → Captum
# ---------------------------------------------------------------------------

def _compute_all_attributions(
    col: SnapshotCollection,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    cols_per_row: int,
    n_steps: int = 30,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Loop over NI checkpoints via col.to_state_dict() and run Captum.

    Returns three parallel lists of tiled grids — one per epoch:
      ig_mag_grids    (N, H, W) normalised magnitude maps, tiled
      gc_grids        (N, H, W) GradCAM maps, tiled
      ig_signed_grids (N, H, W) signed IG maps (raw, for diverging cmap)
    """
    ig_mag_grids, gc_grids, ig_signed_grids = [], [], []

    for epoch in tqdm(col.epochs, desc="Computing attributions (Path A)", unit="epoch"):
        # ── Integration point ──────────────────────────────────────────────
        # col.to_state_dict() returns Dict[str, torch.Tensor] — no NI types.
        # model.load_state_dict() accepts it directly; no conversion needed.
        model = CIFAR10Net()
        model.load_state_dict(col.to_state_dict(epoch))
        model.eval()
        # ───────────────────────────────────────────────────────────────────

        ig_mag  = _ig_magnitude(model, test_images, test_labels, n_steps)
        ig_sgn  = _ig_signed(model, test_images, test_labels, n_steps)
        gc      = _gradcam(model, model.conv2, test_images, test_labels)

        ig_mag_grids.append(_tile_maps(_normalise_per_sample(ig_mag), cols_per_row))
        gc_grids.append(_tile_maps(_normalise_per_sample(gc), cols_per_row))
        ig_signed_grids.append(_tile_maps(ig_sgn, cols_per_row))

    return ig_mag_grids, gc_grids, ig_signed_grids


# ---------------------------------------------------------------------------
# Integration path B: ReplaySession → plain tensors → Captum
# ---------------------------------------------------------------------------

def _demo_replay_captum(
    run_dir: Path,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    final_epoch: int,
    out_path: Path,
) -> None:
    """Demonstrate that ReplaySession activations are directly Captum-compatible.

    ReplayResult.activations values are plain torch.Tensor objects — no NI
    wrapper types appear, so Captum accepts them without any conversion.
    """
    loader = DataLoader(
        torch.utils.data.TensorDataset(test_images, test_labels),
        batch_size=len(test_images),
        shuffle=False,
    )

    # ── Integration point ──────────────────────────────────────────────────
    # ReplaySession loads the checkpoint internally via NI, runs the model,
    # and returns activations as plain torch.Tensor objects in a dict.
    result = ReplaySession(
        run=run_dir,
        checkpoint=final_epoch,
        model_factory=CIFAR10Net,
        dataloader=loader,
        modules=["conv2", "fc1"],
        capture=["activations"],
    ).run()
    # ──────────────────────────────────────────────────────────────────────

    lines = [
        f"ReplaySession — epoch {final_epoch} activations",
        f"  n_samples : {result.metadata.n_samples}",
        "",
    ]
    for name, tensor in result.activations.items():
        assert type(tensor) is torch.Tensor, (
            f"Activation {name!r} is {type(tensor).__name__}; expected torch.Tensor"
        )
        lines.append(f"  {name:12s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    lines += [
        "",
        "All activation values are plain torch.Tensor — Captum accepts them directly.",
        "",
        "Example: feed conv2 activations into IntegratedGradients as additional_forward_args.",
    ]

    out_path.write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _make_attribution_video(
    col: SnapshotCollection,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
    n_steps: int = 30,
) -> Path:
    """Four-panel animation: images | IG magnitude | GradCAM | signed IG."""
    cols_per_row = 5  # 10 classes → 2 rows × 5 cols

    display_imgs = [_denorm(test_images[i]) for i in range(len(test_images))]
    img_grid = _tile_rgb(display_imgs, cols=cols_per_row)

    ig_mag_grids, gc_grids, ig_signed_grids = _compute_all_attributions(
        col, test_images, test_labels, cols_per_row, n_steps=n_steps,
    )

    signed_stack = np.stack(ig_signed_grids)
    v_abs = float(max(abs(signed_stack.min()), abs(signed_stack.max())))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    im_img = axes[0].imshow(img_grid, interpolation="nearest")
    axes[0].set_title("Test images — one per class", fontsize=8)
    axes[0].axis("off")

    im_ig = axes[1].imshow(ig_mag_grids[0], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im_ig, ax=axes[1], shrink=0.8, label="IG magnitude")
    axes[1].set_title("IntegratedGradients — |attr| magnitude", fontsize=8)
    axes[1].axis("off")

    im_gc = axes[2].imshow(gc_grids[0], cmap="jet", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im_gc, ax=axes[2], shrink=0.8, label="GradCAM")
    axes[2].set_title("LayerGradCam — conv2 (upsampled)", fontsize=8)
    axes[2].axis("off")

    im_signed = axes[3].imshow(
        ig_signed_grids[0], cmap="RdBu_r", vmin=-v_abs, vmax=v_abs,
        interpolation="nearest",
    )
    fig.colorbar(im_signed, ax=axes[3], shrink=0.8, label="signed attr")
    axes[3].set_title("IntegratedGradients — signed attribution", fontsize=8)
    axes[3].axis("off")

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        im_ig.set_data(ig_mag_grids[frame])
        im_gc.set_data(gc_grids[frame])
        im_signed.set_data(ig_signed_grids[frame])
        return [im_img, im_ig, im_gc, im_signed, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(col.epochs), interval=1000 // fps, blit=False,
    )

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Final-epoch static panel
# ---------------------------------------------------------------------------

def _save_final_panel(
    col: SnapshotCollection,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    final_epoch: int,
    out_path: Path,
    n_steps: int = 50,
) -> None:
    """3-row × N-col figure: original | IG magnitude | GradCAM per class."""
    model = CIFAR10Net()
    model.load_state_dict(col.to_state_dict(final_epoch))
    model.eval()

    N = test_images.shape[0]
    ig_mag = _ig_magnitude(model, test_images, test_labels, n_steps)
    gc     = _gradcam(model, model.conv2, test_images, test_labels)

    fig, axes = plt.subplots(3, N, figsize=(N * 1.8, 6), constrained_layout=True)
    fig.suptitle(
        f"Final-epoch attributions (epoch {final_epoch}) — one sample per class",
        fontsize=11,
    )

    for i in range(N):
        cls = CIFAR10_CLASSES[test_labels[i].item()]

        axes[0, i].imshow(_denorm(test_images[i]), interpolation="nearest")
        axes[0, i].set_title(cls, fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(
            _normalise_per_sample(ig_mag[i : i + 1])[0],
            cmap="hot", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[1, i].axis("off")

        axes[2, i].imshow(
            _normalise_per_sample(gc[i : i + 1])[0],
            cmap="jet", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=8)
    axes[1, 0].set_ylabel("IG magnitude", fontsize=8)
    axes[2, 0].set_ylabel("GradCAM (conv2)", fontsize=8)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

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
    # Captum's gradient-based methods are most reliable on CPU.
    # Training can use the fastest available device; attribution always runs on CPU.
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
    print(f"Captum device : cpu")

    # --- Data ---
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
    pin = train_device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin)

    # Fixed attribution batch: one normalised image per class, kept on CPU.
    test_images, test_labels = _one_per_class(test_ds)

    # --- Model + NI observer ---
    model = CIFAR10Net().to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    loss_fn = nn.CrossEntropyLoss()

    # capture_buffers=True includes BatchNorm running_mean/running_var so that
    # col.to_state_dict() produces a complete state dict loadable with strict=True.
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=CapturePolicy(capture_buffers=True),
    )

    num_epochs = 2
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
                images, labels = images.to(train_device), labels.to(train_device)
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
        test_loss_acc = torch.zeros(1, device=train_device)
        correct_acc   = torch.zeros(1, device=train_device, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(train_device), labels.to(train_device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels)
                correct_acc   += (out.argmax(1) == labels).sum()
                total += labels.size(0)

        avg_test_loss = test_loss_acc.item() / len(test_loader)
        acc = correct_acc.item() / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(
            f"  → train loss={avg_train_loss:.4f}  "
            f"test loss={avg_test_loss:.4f}  acc={acc:.1%}"
        )
        observer.snapshot(
            epoch=epoch,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss, "accuracy": acc},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")

    # --- Post-training analysis ---
    # ni_load() returns a SnapshotCollection; no model needed to open it.
    col = ni_load(run_dir)
    print(f"\nLoaded collection: {col}")

    # Path A — attribution evolution video via col.to_state_dict()
    video_path = run_dir / "attribution_evolution.mp4"
    print(f"\n[Path A] Generating attribution evolution video → {video_path} …")
    result_path = _make_attribution_video(
        col, test_images, test_labels, accuracy_history, video_path, fps=4, n_steps=30,
    )
    print(f"Video saved: {result_path}")

    # Path A — high-quality static panel for the final epoch
    panel_path = run_dir / "final_attributions.png"
    print(f"\n[Path A] Generating final attribution panel → {panel_path} …")
    _save_final_panel(col, test_images, test_labels, num_epochs - 1, panel_path, n_steps=50)
    print(f"Panel saved: {panel_path}")

    # Path B — ReplaySession activations are plain tensors; show they go directly into Captum
    replay_path = run_dir / "replay_activations.txt"
    print(f"\n[Path B] ReplaySession → activations → Captum compatibility check …")
    _demo_replay_captum(run_dir, test_images, test_labels, num_epochs - 1, replay_path)
    print(f"Replay log saved: {replay_path}")

    # Loss curves
    curves_path = run_dir / "loss_curves.png"
    _save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"Loss curves: {curves_path}")

    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
