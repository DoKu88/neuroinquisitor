"""Visualization utilities for the TorchLens + NeuroInquisitor CIFAR-10 example."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_torchlens_visualizations(
    epoch_history: list[dict[str, dict]],
    target_modules: list[str],
    accuracy_history: list[float],
    train_loss_history: list[float],
    test_loss_history: list[float],
    run_dir: Path,
    fps: int = 4,
) -> None:
    print("\n── Visualizations ──")

    stats_path = run_dir / "torchlens_stats.png"
    save_stats_plot(epoch_history, target_modules, accuracy_history, stats_path)
    print(f"  Saved: {stats_path.name}")

    hist_path = run_dir / "activation_histograms.mp4"
    result = make_histogram_video(epoch_history, target_modules, accuracy_history, hist_path, fps=fps)
    print(f"  Saved: {result.name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"  Saved: {curves_path.name}")


# ---------------------------------------------------------------------------
# Stats plot — three panels, all modules on one figure
# ---------------------------------------------------------------------------


def save_stats_plot(
    epoch_history: list[dict[str, dict]],
    modules: list[str],
    accuracy_history: list[float],
    out_path: Path,
) -> None:
    """Three-row figure: activation magnitude, gradient norm, dead-neuron fraction.

    Each row shows one statistic as a line per module so you can see how the
    signal and gradient flow evolve together over training.
    """
    n_epochs = len(epoch_history)
    epochs   = np.arange(1, n_epochs + 1)

    mean_acts  = {m: [epoch_history[e][m]["mean_act"]  for e in range(n_epochs)] for m in modules}
    grad_norms = {m: [epoch_history[e][m]["grad_norm"] for e in range(n_epochs)] for m in modules}
    sparsities = {m: [epoch_history[e][m]["sparsity"]  for e in range(n_epochs)] for m in modules}

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True, sharex=True)
    colors    = plt.cm.tab10(np.linspace(0, 0.9, len(modules)))

    # Row 0 — mean absolute activation
    for color, mod in zip(colors, modules):
        axes[0].plot(epochs, mean_acts[mod], marker="o", markersize=4, label=mod, color=color)
    axes[0].set_ylabel("Mean |activation|")
    axes[0].set_title("Activation magnitude per module across training epochs", fontsize=10)
    axes[0].legend(ncol=len(modules), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Row 1 — gradient norm (log scale to show all layers together)
    for color, mod in zip(colors, modules):
        axes[1].plot(epochs, grad_norms[mod], marker="s", markersize=4, label=mod, color=color)
    axes[1].set_ylabel("‖gradient‖")
    axes[1].set_yscale("log")
    axes[1].set_title("Gradient norm per module (log scale) — how much each layer is being updated", fontsize=10)
    axes[1].legend(ncol=len(modules), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Row 2 — dead-neuron fraction + accuracy twin axis
    ax_dead = axes[2]
    ax_acc  = ax_dead.twinx()
    for color, mod in zip(colors, modules):
        ax_dead.plot(epochs, [s * 100 for s in sparsities[mod]], marker="^", markersize=4,
                     label=mod, color=color)
    ax_acc.plot(epochs, [a * 100 for a in accuracy_history], color="black",
                linestyle="--", linewidth=1.5, label="accuracy")
    ax_dead.set_ylabel("Near-zero activations (%)")
    ax_acc.set_ylabel("Test accuracy (%)", color="black")
    ax_dead.set_xlabel("Epoch")
    ax_dead.set_title(
        "Dead-neuron proxy (|act| < 1e-4) with test accuracy — "
        "spikes may correlate with learning phase transitions",
        fontsize=10,
    )
    ax_dead.legend(loc="upper left",  ncol=len(modules), fontsize=8)
    ax_acc.legend(loc="upper right", fontsize=8)
    ax_dead.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animated activation histogram video
# ---------------------------------------------------------------------------


def _make_histogram(
    values: np.ndarray,
    n_bins: int = 60,
    global_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, bin_centers) with fixed global range across epochs."""
    lo = float(values.min()) if global_range is None else global_range[0]
    hi = float(values.max()) if global_range is None else global_range[1]
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centers


def _precompute_histograms(
    epoch_history: list[dict[str, dict]],
    modules: list[str],
    n_bins: int = 60,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Pre-compute histogram (counts, centers) per module per epoch.

    Using a fixed global range per module ensures the x-axis is stable across
    frames — otherwise bins shift and the animation looks misleading.
    """
    # Find global value range per module across all epochs
    global_ranges: dict[str, tuple[float, float]] = {}
    for mod in modules:
        all_vals = np.concatenate([epoch_history[e][mod]["act_flat"] for e in range(len(epoch_history))])
        lo, hi   = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
        global_ranges[mod] = (lo, hi)

    histograms: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in modules}
    for e in range(len(epoch_history)):
        for mod in modules:
            vals             = epoch_history[e][mod]["act_flat"]
            counts, centers  = _make_histogram(vals, n_bins=n_bins, global_range=global_ranges[mod])
            histograms[mod].append((counts, centers))

    return histograms


def make_histogram_video(
    epoch_history: list[dict[str, dict]],
    modules: list[str],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate per-layer activation distributions across training epochs.

    Layout (2 rows × 3 cols):
      Row 0 — conv1, conv2, conv3 histograms
      Row 1 — fc1, fc2 histograms + accuracy/sparsity twin-axis line plot

    Watching this video reveals:
      - Early training: broad, low-amplitude distributions.
      - Later training: distributions sharpen and shift as the model specialises.
      - Dead neurons: large spike at zero that grows over time in some layers.
    """
    n_epochs   = len(epoch_history)
    n_cols     = 3
    histograms = _precompute_histograms(epoch_history, modules)
    colors     = plt.cm.tab10(np.linspace(0, 0.9, len(modules)))
    mod_color  = dict(zip(modules, colors))

    # Global y-max per module for stable y-axis across frames
    y_maxes = {
        mod: max(histograms[mod][e][0].max() for e in range(n_epochs))
        for mod in modules
    }

    fig, axes = plt.subplots(2, n_cols, figsize=(14, 8), constrained_layout=True)

    # Assign axes: first 5 positions to modules, last position to accuracy strip
    mod_axes: list[plt.Axes] = [
        axes[0, 0], axes[0, 1], axes[0, 2],
        axes[1, 0], axes[1, 1],
    ]
    summary_ax    = axes[1, 2]
    summary_ax_r  = summary_ax.twinx()

    # Initialise bar charts
    bar_containers: list[Any] = []
    for i, mod in enumerate(modules):
        ax           = mod_axes[i]
        counts0, ctrs = histograms[mod][0]
        width        = ctrs[1] - ctrs[0] if len(ctrs) > 1 else 1.0
        bars         = ax.bar(ctrs, counts0, width=width * 0.9,
                              color=mod_color[mod], alpha=0.75)
        ax.set_ylim(0, y_maxes[mod] * 1.15 + 1e-8)
        ax.set_title(f"{mod} — activation distribution", fontsize=8)
        ax.set_xlabel("Activation value", fontsize=7)
        ax.set_ylabel("Count",            fontsize=7)
        ax.tick_params(labelsize=6)
        bar_containers.append(bars)

    # Summary panel: sparsity per module + accuracy
    epoch_x = np.arange(1, n_epochs + 1)
    sparsity_lines: list[Any] = []
    for mod in modules:
        (line,) = summary_ax.plot(
            [], [], marker="o", markersize=3,
            color=mod_color[mod], label=mod,
        )
        sparsity_lines.append(line)
    (acc_line,) = summary_ax_r.plot([], [], "k--", linewidth=1.5, label="accuracy")

    summary_ax.set_xlim(0.5, n_epochs + 0.5)
    summary_ax.set_ylim(0, 100)
    summary_ax_r.set_ylim(0, 100)
    summary_ax.set_xlabel("Epoch",                  fontsize=7)
    summary_ax.set_ylabel("Near-zero act. (%)",     fontsize=7)
    summary_ax_r.set_ylabel("Test accuracy (%)",    fontsize=7)
    summary_ax.set_title("Dead neurons & accuracy", fontsize=8)
    summary_ax.legend(loc="upper left",  fontsize=6, ncol=2)
    summary_ax_r.legend(loc="upper right", fontsize=6)
    summary_ax.tick_params(labelsize=6)
    summary_ax_r.tick_params(labelsize=6)
    summary_ax.grid(True, alpha=0.25)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(
            f"TorchLens activation distributions — Epoch {frame + 1}  |  Test accuracy: {acc:.1%}"
        )
        artists: list = [title_text]

        # Update histograms
        for i, mod in enumerate(modules):
            counts, _ = histograms[mod][frame]
            for bar, h in zip(bar_containers[i], counts):
                bar.set_height(h)
                artists.append(bar)

        # Update summary panel up to current epoch
        xs = epoch_x[: frame + 1]
        for j, mod in enumerate(modules):
            ys = [epoch_history[e][mod]["sparsity"] * 100 for e in range(frame + 1)]
            sparsity_lines[j].set_data(xs, ys)
            artists.append(sparsity_lines[j])

        acc_ys = [accuracy_history[e] * 100 for e in range(min(frame + 1, len(accuracy_history)))]
        acc_line.set_data(xs[: len(acc_ys)], acc_ys)
        artists.append(acc_line)

        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_epochs, interval=1000 // fps, blit=False,
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
# Loss / accuracy curves
# ---------------------------------------------------------------------------


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
