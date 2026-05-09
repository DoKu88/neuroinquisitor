"""Visualization utilities for the CIFAR-10 + TransformerLens example.

Produces five outputs for each run:
  attention_patterns.mp4   — animated (n_layers × n_heads) attention maps across epochs
  logit_lens.png           — heatmap: which layer first exposes class information
  activation_patching.png  — accuracy when epoch-0 residual stream is spliced in per layer
  weight_svd.png           — top singular value of W_Q/W_K per head over training
  loss_curves.png          — train/test loss + test accuracy
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_visualizations(
    attn_patterns:    list[dict[str, torch.Tensor]],
    logit_lens_probs: np.ndarray,
    patch_accs:       np.ndarray,
    svd_data:         dict[str, np.ndarray],
    accuracy_history: list[float],
    train_loss_history: list[float],
    test_loss_history:  list[float],
    n_layers:  int,
    n_heads:   int,
    run_dir:   Path,
    fps:       int = 4,
) -> None:
    print("\n── Visualizations ──")

    attn_path = run_dir / "attention_patterns.mp4"
    print(f"  Generating attention video  → {attn_path.name} …")
    saved = make_attention_video(attn_patterns, accuracy_history, n_layers, n_heads, attn_path, fps=fps)
    print(f"  Saved: {saved.name}")

    lens_path = run_dir / "logit_lens.png"
    plot_logit_lens(logit_lens_probs, n_layers, lens_path)
    print(f"  Saved: {lens_path.name}")

    patch_path = run_dir / "activation_patching.png"
    plot_patch_accuracy(patch_accs, n_layers, patch_path)
    print(f"  Saved: {patch_path.name}")

    svd_path = run_dir / "weight_svd.png"
    plot_weight_svd(svd_data, n_layers, n_heads, svd_path)
    print(f"  Saved: {svd_path.name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"  Saved: {curves_path.name}")


# ---------------------------------------------------------------------------
# Use Case 2 — Attention pattern animation
# ---------------------------------------------------------------------------


def _mean_attn_map(
    epoch_cache: dict[str, torch.Tensor],
    layer: int,
) -> np.ndarray:
    """Return mean attention map for the first head across the batch: (seq, seq)."""
    key = f"blocks.{layer}.attn.hook_pattern"
    if key not in epoch_cache:
        return np.zeros((1, 1))
    # shape: (batch, n_heads, seq, seq) → mean over batch → (n_heads, seq, seq)
    return epoch_cache[key].float().mean(dim=0).numpy()  # (n_heads, seq, seq)


def make_attention_video(
    attn_patterns:    list[dict[str, torch.Tensor]],
    accuracy_history: list[float],
    n_layers:         int,
    n_heads:          int,
    out_path:         Path,
    fps:              int = 4,
) -> Path:
    """Animate per-layer per-head attention maps across training epochs.

    Layout: n_layers rows × n_heads cols, each cell shows mean attention
    over the probe batch for that (layer, head) combination.
    Sequence position 0 is the CLS token.
    """
    n_epochs = len(attn_patterns)
    seq_len  = next(iter(attn_patterns[0].values())).shape[-1]

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3.5 * n_heads, 3.0 * n_layers),
        constrained_layout=True,
    )
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_heads == 1:
        axes = axes[:, np.newaxis]

    # Pre-compute global vmax so colour scale is fixed across frames
    all_maps: list[np.ndarray] = []
    for ep in attn_patterns:
        for li in range(n_layers):
            maps = _mean_attn_map(ep, li)
            if maps.shape[-1] > 1:
                all_maps.append(maps)
    vmax = float(np.stack(all_maps).max()) if all_maps else 1.0

    # Initialise imshow objects for each cell
    ims: list[list[Any]] = []
    for li in range(n_layers):
        row_ims: list[Any] = []
        maps0 = _mean_attn_map(attn_patterns[0], li)
        for hi in range(n_heads):
            ax = axes[li, hi]
            data = maps0[hi] if maps0.ndim == 3 else np.zeros((seq_len, seq_len))
            im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
            ax.set_title(f"L{li} H{hi}", fontsize=7)
            ax.set_xlabel("Key pos", fontsize=6)
            ax.set_ylabel("Query pos", fontsize=6)
            ax.tick_params(labelsize=5)
            row_ims.append(im)
        ims.append(row_ims)

    title = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title.set_text(f"Attention patterns — Epoch {frame + 1}  |  Test acc: {acc:.1%}")
        artists: list = [title]
        for li in range(n_layers):
            maps = _mean_attn_map(attn_patterns[frame], li)
            for hi in range(n_heads):
                data = maps[hi] if maps.ndim == 3 else np.zeros((seq_len, seq_len))
                ims[li][hi].set_data(data)
                artists.append(ims[li][hi])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_epochs, interval=1000 // fps, blit=False)

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Use Case 3 — Logit lens
# ---------------------------------------------------------------------------


def plot_logit_lens(
    logit_lens_probs: np.ndarray,
    n_layers:         int,
    out_path:         Path,
) -> None:
    """Three-panel logit lens figure.

    Panel 1 — class confidence at each layer (final epoch), mean over probe batch.
    Panel 2 — depth at which each class probability first exceeds 5% (final epoch).
    Panel 3 — evolution of the top class's confidence at every layer over epochs.
    """
    n_epochs, n_layers_, n_classes = logit_lens_probs.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Panel 1: (layer, class) heatmap at final epoch
    ax = axes[0]
    im = ax.imshow(
        logit_lens_probs[-1].T,
        aspect="auto", cmap="viridis", vmin=0, vmax=logit_lens_probs[-1].max(),
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean class probability")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=7)
    ax.set_xticks(range(n_layers_))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers_)], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Class")
    ax.set_title("Logit lens — final epoch\n(class probability at each layer)", fontsize=9)

    # Panel 2: first layer at which each class probability exceeds 0.05
    threshold     = 0.05
    emergence     = np.full(n_classes, n_layers_, dtype=float)
    final_probs   = logit_lens_probs[-1]  # (n_layers, n_classes)
    for c in range(n_classes):
        for li in range(n_layers_):
            if final_probs[li, c] >= threshold:
                emergence[c] = li
                break

    ax = axes[1]
    bars = ax.barh(range(n_classes), emergence, color="steelblue")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel(f"First layer with P(class) ≥ {threshold:.0%}")
    ax.set_title(f"Class emergence depth\n(final epoch)", fontsize=9)
    ax.set_xlim(0, n_layers_)
    ax.set_xticks(range(n_layers_))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers_)], fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: top class's probability at each layer over training epochs
    top_class = int(logit_lens_probs[-1, -1].argmax())
    data      = logit_lens_probs[:, :, top_class]  # (n_epochs, n_layers)

    ax = axes[2]
    for li in range(n_layers_):
        ax.plot(range(1, n_epochs + 1), data[:, li], label=f"Layer {li}", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"P({CIFAR10_CLASSES[top_class]})")
    ax.set_title(f"Logit lens over training\ntop class: {CIFAR10_CLASSES[top_class]}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Use Case 4 — Activation patching accuracy
# ---------------------------------------------------------------------------


def plot_patch_accuracy(
    patch_accs: np.ndarray,
    n_layers:   int,
    out_path:   Path,
) -> None:
    """Bar chart of accuracy when epoch-0 residual stream replaces the final model's, per layer.

    Bars below the dashed baseline (unpatched final model) show the accuracy cost
    of forcing the model to use early-training representations at that depth.
    """
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    colours = ["#e74c3c" if a < patch_accs.max() * 0.9 else "#2ecc71" for a in patch_accs]
    ax.bar(range(n_layers), patch_accs * 100, color=colours, edgecolor="white")
    ax.axhline(patch_accs.max() * 100, linestyle="--", color="grey", linewidth=1.2, label="Best patched layer")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_ylabel("Test accuracy after patching (%)")
    ax.set_title(
        "Activation patching: epoch-0 residual → final model\n"
        "Low bar = that layer's learned representation matters most",
        fontsize=9,
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for i, acc in enumerate(patch_accs):
        ax.text(i, acc * 100 + 1, f"{acc:.0%}", ha="center", va="bottom", fontsize=8)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Use Case 5 — Weight SVD
# ---------------------------------------------------------------------------


def plot_weight_svd(
    svd_data: dict[str, np.ndarray],
    n_layers: int,
    n_heads:  int,
    out_path: Path,
) -> None:
    """Grid of plots: top singular value of W_Q and W_K per head over training.

    Layout: n_layers rows × n_heads cols.
    Each cell shows σ₁(W_Q) (blue) and σ₁(W_K) (red) over epochs.
    Growing σ₁ signals that the head is specialising onto a dominant direction.
    """
    sig_Q = svd_data["sigma_Q"]  # (n_epochs, n_layers, n_heads, d_head)
    sig_K = svd_data["sigma_K"]
    n_epochs = sig_Q.shape[0]
    epochs   = range(1, n_epochs + 1)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(4.5 * n_heads, 3.0 * n_layers),
        constrained_layout=True,
        sharex=True,
    )
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_heads == 1:
        axes = axes[:, np.newaxis]

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li, hi]
            ax.plot(epochs, sig_Q[:, li, hi, 0], color="steelblue", label="σ₁(W_Q)", linewidth=1.5)
            ax.plot(epochs, sig_K[:, li, hi, 0], color="coral",     label="σ₁(W_K)", linewidth=1.5)
            ax.set_title(f"Layer {li}  Head {hi}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if li == n_layers - 1:
                ax.set_xlabel("Epoch", fontsize=7)
            if hi == 0:
                ax.set_ylabel("Top singular value", fontsize=7)
            ax.tick_params(labelsize=6)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Weight SVD — top singular value of W_Q and W_K per head over training", fontsize=10)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def save_loss_curves(
    train_losses:     list[float],
    test_losses:      list[float],
    accuracy_history: list[float],
    out_path:         Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax_loss.plot(epochs, train_losses, marker="o", markersize=3, label="Train loss")
    ax_loss.plot(epochs, test_losses,  marker="s", markersize=3, label="Test loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("CIFAR-10 PatchViT — loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in accuracy_history], marker="o", markersize=3, color="tab:green")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_title("CIFAR-10 PatchViT — test accuracy")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
