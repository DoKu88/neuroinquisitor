"""Visualization and weight-rendering utilities for the basic usage example."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def generate_visualizations(
    weight_history: list[dict[str, np.ndarray]],
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    replay_modules: list[str],
    train_loss_history: list[float],
    test_loss_history: list[float],
    run_dir: Path,
    fps: int = 4,
) -> None:
    print("\n── Visualizations ──")

    video_path = run_dir / "weights_over_time.mp4"
    print(f"  Generating weight video    → {video_path.name} …")
    print(f"  Saved: {make_weight_video(weight_history, video_path, fps=fps).name}")

    replay_vid_path = run_dir / "activations_gradients.mp4"
    print(f"  Generating replay video    → {replay_vid_path.name} …")
    print(f"  Saved: {make_replay_video(replay_history, replay_modules, replay_vid_path, fps=fps).name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, curves_path)
    print(f"  Saved: {curves_path.name}")


# ---------------------------------------------------------------------------
# Weight rendering helpers
# ---------------------------------------------------------------------------


def make_weight_video(
    weight_history: list[dict[str, np.ndarray]],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate FC1 and FC2 weight heatmaps over epochs."""
    n_frames = len(weight_history)
    layers = [("fc1.weight", "FC1  (16×4)"), ("fc2.weight", "FC2  (1×16)")]

    vlims: list[tuple[float, float]] = []
    for key, _ in layers:
        all_vals = np.concatenate([snap[key].ravel() for snap in weight_history])
        abs_max = float(np.abs(all_vals).max()) or 1.0
        vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(1, len(layers), figsize=(10, 4), constrained_layout=True)

    images = []
    for ax, (key, label), (vmin, vmax) in zip(axes, layers, vlims):
        im = ax.imshow(
            weight_history[0][key], aspect="auto", cmap="RdBu_r",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Input dim")
        ax.set_ylabel("Output dim")
        images.append(im)

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        title_text.set_text(f"Epoch {frame + 1}")
        for im, (key, _) in zip(images, layers):
            im.set_data(weight_history[frame][key])
        return [*images, title_text]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


def save_loss_curves(
    train_losses: list[float],
    test_losses: list[float],
    out_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs, test_losses,  marker="s", label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("TinyMLP — training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Replay visualization helpers
# ---------------------------------------------------------------------------


def precompute_replay_data(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    """Precompute bar-chart magnitudes and gradient heatmaps for all epochs."""
    act_mags:  list[dict[str, np.ndarray]] = []
    grad_mags: list[dict[str, np.ndarray]] = []
    grad_heat: list[dict[str, np.ndarray]] = []

    for frame in replay_history:
        e_act, e_grad, e_heat = {}, {}, {}
        for name in modules:
            act = frame["activations"][name]
            e_act[name] = np.abs(act).mean(axis=0) if act.ndim > 1 else np.abs(act)

            if name not in frame["gradients"]:
                continue
            grad = frame["gradients"][name]
            e_grad[name] = np.abs(grad).reshape(grad.shape[0], -1).mean(axis=1)

            if grad.ndim == 3:
                e_heat[name] = np.abs(grad).mean(axis=0)
            else:
                side = math.ceil(math.sqrt(grad.shape[0]))
                buf = np.zeros(side * side)
                buf[: grad.shape[0]] = np.abs(grad)
                e_heat[name] = buf.reshape(side, side)

        act_mags.append(e_act)
        grad_mags.append(e_grad)
        grad_heat.append(e_heat)

    return act_mags, grad_mags, grad_heat


def make_replay_video(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate activations, gradient bar charts, and gradient heatmaps across epochs."""
    n_epochs = len(replay_history)
    n_mods   = len(modules)

    act_mags, grad_mags, grad_heat = precompute_replay_data(replay_history, modules)

    act_ylims   = {m: max(act_mags[e][m].max()  for e in range(n_epochs)) for m in modules}
    grad_ylims  = {m: max(grad_mags[e][m].max() for e in range(n_epochs)) for m in modules if m in grad_mags[0]}
    heat_vmaxes = {m: max(grad_heat[e][m].max() for e in range(n_epochs)) for m in modules if m in grad_heat[0]}

    fig, axes = plt.subplots(3, n_mods, figsize=(5 * n_mods, 9), constrained_layout=True)
    if n_mods == 1:
        axes = axes.reshape(3, 1)

    act_bars:  list[Any] = []
    grad_bars: list[Any] = []
    heat_ims:  list[Any] = []

    for c, name in enumerate(modules):
        b_a = axes[0, c].bar(np.arange(len(act_mags[0][name])), act_mags[0][name], width=1.0, color="steelblue")
        axes[0, c].set_ylim(0, act_ylims[name] * 1.1 + 1e-8)
        axes[0, c].set_title(f"{name} — activations", fontsize=8)
        axes[0, c].set_xlabel("Channel")
        axes[0, c].set_ylabel("Mean |activation|")
        act_bars.append(b_a)

        if name in grad_ylims:
            b_g = axes[1, c].bar(np.arange(len(grad_mags[0][name])), grad_mags[0][name], width=1.0, color="coral")
            axes[1, c].set_ylim(0, grad_ylims[name] * 1.1 + 1e-8)
            axes[1, c].set_title(f"{name} — gradients", fontsize=8)
            axes[1, c].set_xlabel("Channel")
            axes[1, c].set_ylabel("Mean |gradient|")
            grad_bars.append(b_g)
        else:
            axes[1, c].set_visible(False)
            grad_bars.append(None)

        if name in heat_vmaxes:
            raw_grad = replay_history[0]["gradients"][name]
            im = axes[2, c].imshow(
                grad_heat[0][name], cmap="inferno",
                vmin=0, vmax=heat_vmaxes[name] + 1e-8,
                interpolation="nearest", aspect="auto",
            )
            fig.colorbar(im, ax=axes[2, c], shrink=0.8, label="|∇|")
            if raw_grad.ndim == 3:
                axes[2, c].set_title(f"{name} — gradient heatmap\n(mean |∇| over channels)", fontsize=7)
                axes[2, c].set_xlabel("W")
                axes[2, c].set_ylabel("H")
            else:
                side = math.ceil(math.sqrt(raw_grad.shape[0]))
                axes[2, c].set_title(f"{name} — gradient heatmap\n(reshaped {side}×{side})", fontsize=7)
                axes[2, c].set_xlabel("Feature (col)")
                axes[2, c].set_ylabel("Feature (row)")
            heat_ims.append(im)
        else:
            axes[2, c].set_visible(False)
            heat_ims.append(None)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        title_text.set_text(f"Replay — Epoch {frame + 1}")
        artists: list = [title_text]
        for c, name in enumerate(modules):
            for bar, h in zip(act_bars[c], act_mags[frame][name]):
                bar.set_height(h)
                artists.append(bar)
            if grad_bars[c] is not None:
                for bar, h in zip(grad_bars[c], grad_mags[frame][name]):
                    bar.set_height(h)
                    artists.append(bar)
            if heat_ims[c] is not None and name in grad_heat[frame]:
                heat_ims[c].set_data(grad_heat[frame][name])
                artists.append(heat_ims[c])
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
