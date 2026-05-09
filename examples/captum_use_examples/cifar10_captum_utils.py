"""Visualization and Captum attribution utilities for the CIFAR-10 Captum example."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam
from tqdm import tqdm

from neuroinquisitor.collection import SnapshotCollection

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_captum_visualizations(
    snapshots: SnapshotCollection,
    model_factory: Callable[[], nn.Module],
    attr_images: torch.Tensor,
    attr_labels: torch.Tensor,
    ig_mag_grids: list[np.ndarray],
    gc_grids: list[np.ndarray],
    ig_signed_grids: list[np.ndarray],
    accuracy_history: list[float],
    train_loss_history: list[float],
    test_loss_history: list[float],
    run_dir: Path,
    fps: int = 4,
) -> None:
    print("\n── Visualizations ──")

    video_path = run_dir / "attribution_evolution.mp4"
    print(f"  Generating attribution video → {video_path.name} …")
    result = make_attribution_video(
        ig_mag_grids, gc_grids, ig_signed_grids,
        attr_images, accuracy_history, video_path, fps=fps,
    )
    print(f"  Saved: {result.name}")

    panel_path = run_dir / "final_attributions.png"
    print(f"  Generating final attribution panel → {panel_path.name} …")
    save_final_panel(
        snapshots, model_factory, attr_images, attr_labels,
        snapshots.epochs[-1], panel_path, n_steps=50,
    )
    print(f"  Saved: {panel_path.name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"  Saved: {curves_path.name}")


# ---------------------------------------------------------------------------
# Captum compute helpers
# ---------------------------------------------------------------------------


def ig_magnitude(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 30,
) -> np.ndarray:
    """IntegratedGradients → (N, H, W) attribution magnitude (|Δ| summed over channels)."""
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)
    attrs = ig.attribute(inputs, baselines=baseline, target=targets, n_steps=n_steps)
    return attrs.abs().sum(dim=1).detach().cpu().numpy()


def ig_signed(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 30,
) -> np.ndarray:
    """IntegratedGradients → (N, H, W) signed attribution (mean over channels)."""
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(inputs)
    attrs = ig.attribute(inputs, baselines=baseline, target=targets, n_steps=n_steps)
    return attrs.mean(dim=1).detach().cpu().numpy()


def gradcam(
    model: nn.Module,
    layer: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    out_size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """LayerGradCam → (N, H, W) map, clamped ≥ 0, upsampled to out_size."""
    lgc = LayerGradCam(model, layer)
    gc = lgc.attribute(inputs, target=targets)
    if gc.shape[1] > 1:
        gc = gc.sum(dim=1, keepdim=True)
    gc_up = F.interpolate(gc.float(), size=out_size, mode="bilinear", align_corners=False)
    return gc_up.squeeze(1).clamp(min=0).detach().cpu().numpy()


def compute_attributions_per_epoch(
    snapshots: SnapshotCollection,
    model_factory: Callable[[], nn.Module],
    attr_images: torch.Tensor,
    attr_labels: torch.Tensor,
    n_steps: int = 30,
    cols_per_row: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Loop over NI checkpoints via snapshots.to_state_dict() and run Captum attribution.

    Integration point: col.to_state_dict() returns Dict[str, torch.Tensor].
    model.load_state_dict() accepts it directly — no NI types cross the boundary.

    Returns three parallel lists — one tiled grid per epoch:
      ig_mag_grids    — |IG| magnitude summed over channels, normalised to [0, 1]
      gc_grids        — LayerGradCam on conv2, upsampled and normalised to [0, 1]
      ig_signed_grids — signed IG mean over channels (raw values, for diverging cmap)
    """
    ig_mag_grids, gc_grids, ig_signed_grids = [], [], []

    for epoch in tqdm(snapshots.epochs, desc="  Computing attributions (Path A)", unit="epoch"):
        model = model_factory()
        model.load_state_dict(snapshots.to_state_dict(epoch))
        model.eval()

        ig_mag_ = ig_magnitude(model, attr_images, attr_labels, n_steps)
        ig_sgn_ = ig_signed(model, attr_images, attr_labels, n_steps)
        gc_     = gradcam(model, model.conv2, attr_images, attr_labels)

        ig_mag_grids.append(_tile_maps(_normalise_per_sample(ig_mag_), cols_per_row))
        gc_grids.append(_tile_maps(_normalise_per_sample(gc_), cols_per_row))
        ig_signed_grids.append(_tile_maps(ig_sgn_, cols_per_row))

    return ig_mag_grids, gc_grids, ig_signed_grids


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) normalised tensor → (H, W, 3) float32 in [0, 1]."""
    return (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def _tile_rgb(imgs: list[np.ndarray], cols: int, pad: int = 1) -> np.ndarray:
    """Tile a list of (H, W, 3) arrays into a single (rows*H, cols*W, 3) canvas."""
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
    """Normalise each (H, W) slice in (N, H, W) to [0, 1] independently."""
    out = np.empty_like(maps, dtype=np.float32)
    for i in range(maps.shape[0]):
        lo, hi = maps[i].min(), maps[i].max()
        out[i] = (maps[i] - lo) / (hi - lo + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def make_attribution_video(
    ig_mag_grids: list[np.ndarray],
    gc_grids: list[np.ndarray],
    ig_signed_grids: list[np.ndarray],
    attr_images: torch.Tensor,
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
    cols_per_row: int = 5,
) -> Path:
    """Four-panel animation: images | IG magnitude | GradCAM | signed IG."""
    display_imgs = [_denorm(attr_images[i]) for i in range(len(attr_images))]
    img_grid = _tile_rgb(display_imgs, cols=cols_per_row)

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
        ig_signed_grids[0], cmap="RdBu_r", vmin=-v_abs, vmax=v_abs, interpolation="nearest",
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
        fig, update, frames=len(ig_mag_grids), interval=1000 // fps, blit=False,
    )

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


def save_final_panel(
    snapshots: SnapshotCollection,
    model_factory: Callable[[], nn.Module],
    attr_images: torch.Tensor,
    attr_labels: torch.Tensor,
    final_epoch: int,
    out_path: Path,
    n_steps: int = 50,
) -> None:
    """3-row × N-col figure: original | IG magnitude | GradCAM — one column per class."""
    model = model_factory()
    model.load_state_dict(snapshots.to_state_dict(final_epoch))
    model.eval()

    N = attr_images.shape[0]
    ig_mag_ = ig_magnitude(model, attr_images, attr_labels, n_steps)
    gc_     = gradcam(model, model.conv2, attr_images, attr_labels)

    fig, axes = plt.subplots(3, N, figsize=(N * 1.8, 6), constrained_layout=True)
    fig.suptitle(
        f"Final-epoch attributions (epoch {final_epoch}) — one sample per class",
        fontsize=11,
    )

    for i in range(N):
        cls = CIFAR10_CLASSES[attr_labels[i].item()]

        axes[0, i].imshow(_denorm(attr_images[i]), interpolation="nearest")
        axes[0, i].set_title(cls, fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(
            _normalise_per_sample(ig_mag_[i: i + 1])[0],
            cmap="hot", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[1, i].axis("off")

        axes[2, i].imshow(
            _normalise_per_sample(gc_[i: i + 1])[0],
            cmap="jet", vmin=0, vmax=1, interpolation="nearest",
        )
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=8)
    axes[1, 0].set_ylabel("IG magnitude", fontsize=8)
    axes[2, 0].set_ylabel("GradCAM (conv2)", fontsize=8)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_loss_curves(
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
