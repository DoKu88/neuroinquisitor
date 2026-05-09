"""Visualization and weight-rendering utilities for the CIFAR-10 example."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Weight rendering helpers
# ---------------------------------------------------------------------------


def compute_conv1_filter_lims(
    weight_history: list[dict[str, np.ndarray]],
) -> list[tuple[float, float]]:
    """Fixed (lo, hi) per filter across ALL epochs so weight growth is visible."""
    stacked = np.stack([snap["conv1.weight"] for snap in weight_history])
    return [
        (float(stacked[:, i].min()), float(stacked[:, i].max()))
        for i in range(stacked.shape[1])
    ]


def render_rgb_filters(
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
        filt = weight[i].transpose(1, 2, 0).copy()
        lo, hi = per_filter_lims[i]
        filt = np.clip((filt - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def render_delta_rgb(
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


def render_conv2_norm_heatmap(weight: np.ndarray) -> np.ndarray:
    """Per-filter ‖W‖₂ for (64, in, H, W) as an 8×8 grid."""
    norms = np.sqrt((weight ** 2).sum(axis=(1, 2, 3)))
    side = math.ceil(math.sqrt(len(norms)))
    padded = np.full(side * side, np.nan)
    padded[: len(norms)] = norms
    return padded.reshape(side, side)


def build_fc1_timeline(weight_history: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Return (256, num_epochs) matrix of per-neuron row-norms."""
    row_norms = np.array([
        np.linalg.norm(snap["fc1.weight"], axis=1)
        for snap in weight_history
    ])
    return row_norms.T  # (256, num_epochs)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def make_video(
    weight_history: list[dict[str, np.ndarray]],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    n_frames = len(weight_history)
    weight_0 = weight_history[0]

    conv1_lims = compute_conv1_filter_lims(weight_history)
    conv2_heatmaps = [render_conv2_norm_heatmap(snap["conv2.weight"]) for snap in weight_history]
    conv2_vmax = float(max(float(h[~np.isnan(h)].max()) for h in conv2_heatmaps))
    fc1_timeline = build_fc1_timeline(weight_history)
    fc1_vmax = float(fc1_timeline.max())
    delta_vmax = float(max(
        np.abs(snap["conv1.weight"] - weight_0["conv1.weight"]).max()
        for snap in weight_history
    ))
    conv1_rendered = [render_rgb_filters(snap["conv1.weight"], conv1_lims) for snap in weight_history]
    delta_rendered = [render_delta_rgb(snap["conv1.weight"], weight_0["conv1.weight"], delta_vmax) for snap in weight_history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)

    im_conv1 = axes[0].imshow(conv1_rendered[0], interpolation="nearest")
    axes[0].set_title("Conv1 — RGB filters (fixed bounds)", fontsize=8)
    axes[0].axis("off")

    im_conv2 = axes[1].imshow(conv2_heatmaps[0], cmap="viridis", vmin=0, vmax=conv2_vmax, interpolation="nearest")
    fig.colorbar(im_conv2, ax=axes[1], shrink=0.8, label="‖W‖₂")
    axes[1].set_title("Conv2 — per-filter ‖W‖₂", fontsize=8)
    axes[1].axis("off")

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


def precompute_replay_data(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
) -> tuple[
    list[dict[str, np.ndarray]],   # act_mags:  per-channel mean |activation|
    list[dict[str, np.ndarray]],   # grad_mags: per-channel mean |gradient|
    list[dict[str, np.ndarray]],   # grad_heat: 2-D gradient heatmaps
]:
    """Precompute bar-chart magnitudes and gradient heatmaps for all epochs.

    Gradient heatmaps:
      Conv layers  — mean |∇| across channels → (H, W) spatial map
      FC layers    — |∇| reshaped to the nearest square 2-D grid
    """
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
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate activations, gradient bar charts, and gradient heatmaps across epochs.

    Three rows per frame:
      Row 0 — Activation magnitude bar charts (mean |activation| per channel)
      Row 1 — Gradient magnitude bar charts   (mean |gradient| per channel)
      Row 2 — Gradient heatmaps: mean |∇| over channels for conv layers;
               near-square reshape for fully-connected layers
    """
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
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Replay — Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
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


def make_combined_video(
    weight_history: list[dict[str, np.ndarray]],
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    accuracy_history: list[float],
    replay_modules: list[str],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Single video combining weight evolution, activations, gradient bars, and gradient heatmaps.

    Layout (GridSpec 4 rows × 12 cols):
      Row 0 — 4 weight panels       (each 3 cols wide)
      Row 1 — Activation bar charts (each 4 cols wide, one per module)
      Row 2 — Gradient bar charts   (each 4 cols wide, one per module)
      Row 3 — Gradient heatmaps     (each 4 cols wide, one per module)
    """
    n_frames = len(weight_history)
    weight_0 = weight_history[0]
    n_mods   = len(replay_modules)

    conv1_lims     = compute_conv1_filter_lims(weight_history)
    conv2_hmaps    = [render_conv2_norm_heatmap(s["conv2.weight"]) for s in weight_history]
    conv2_vmax     = float(max(float(h[~np.isnan(h)].max()) for h in conv2_hmaps))
    fc1_timeline   = build_fc1_timeline(weight_history)
    fc1_vmax       = float(fc1_timeline.max())
    delta_vmax     = float(max(
        np.abs(s["conv1.weight"] - weight_0["conv1.weight"]).max() for s in weight_history
    ))
    conv1_rendered = [render_rgb_filters(s["conv1.weight"], conv1_lims) for s in weight_history]
    delta_rendered = [render_delta_rgb(s["conv1.weight"], weight_0["conv1.weight"], delta_vmax) for s in weight_history]

    act_mags, grad_mags, grad_heat = precompute_replay_data(replay_history, replay_modules)

    act_ylims   = {m: max(act_mags[e][m].max()  for e in range(n_frames)) for m in replay_modules}
    grad_ylims  = {m: max(grad_mags[e][m].max() for e in range(n_frames)) for m in replay_modules if m in grad_mags[0]}
    heat_vmaxes = {m: max(grad_heat[e][m].max() for e in range(n_frames)) for m in replay_modules if m in grad_heat[0]}

    fig = plt.figure(figsize=(22, 22))
    gs  = fig.add_gridspec(4, 12, hspace=0.55, wspace=0.45)

    ax_w    = [fig.add_subplot(gs[0, i * 3: (i + 1) * 3]) for i in range(4)]
    ax_act  = [fig.add_subplot(gs[1, i * 4: (i + 1) * 4]) for i in range(n_mods)]
    ax_grad = [fig.add_subplot(gs[2, i * 4: (i + 1) * 4]) for i in range(n_mods)]
    ax_heat = [fig.add_subplot(gs[3, i * 4: (i + 1) * 4]) for i in range(n_mods)]

    im_conv1_filt = ax_w[0].imshow(conv1_rendered[0], interpolation="nearest")
    ax_w[0].set_title("Conv1 — RGB filters", fontsize=7)
    ax_w[0].axis("off")

    im_conv2_norm = ax_w[1].imshow(conv2_hmaps[0], cmap="viridis", vmin=0, vmax=conv2_vmax, interpolation="nearest")
    fig.colorbar(im_conv2_norm, ax=ax_w[1], shrink=0.7, label="‖W‖₂")
    ax_w[1].set_title("Conv2 — per-filter ‖W‖₂", fontsize=7)
    ax_w[1].axis("off")

    fc1_init = np.full_like(fc1_timeline, np.nan)
    fc1_init[:, 0] = fc1_timeline[:, 0]
    im_fc1_norms = ax_w[2].imshow(fc1_init, aspect="auto", cmap="plasma", vmin=0, vmax=fc1_vmax, interpolation="nearest")
    fig.colorbar(im_fc1_norms, ax=ax_w[2], shrink=0.7, label="‖row‖₂")
    ax_w[2].set_title("FC1 — neuron row-norms", fontsize=7)
    ax_w[2].set_xlabel("Epoch", fontsize=7)
    ax_w[2].set_ylabel("Neuron", fontsize=7)

    im_delta = ax_w[3].imshow(delta_rendered[0], interpolation="nearest")
    ax_w[3].set_title("Conv1  |W_t − W_0|", fontsize=7)
    ax_w[3].axis("off")

    act_bars:  list[Any] = []
    grad_bars: list[Any] = []
    heat_ims:  list[Any] = []

    for c, name in enumerate(replay_modules):
        b_a = ax_act[c].bar(np.arange(len(act_mags[0][name])), act_mags[0][name], width=1.0, color="steelblue")
        ax_act[c].set_ylim(0, act_ylims[name] * 1.1 + 1e-8)
        ax_act[c].set_title(f"{name} — activations", fontsize=7)
        ax_act[c].set_xlabel("Channel", fontsize=7)
        ax_act[c].set_ylabel("Mean |act|", fontsize=7)
        ax_act[c].tick_params(labelsize=6)
        act_bars.append(b_a)

        if name in grad_ylims:
            b_g = ax_grad[c].bar(np.arange(len(grad_mags[0][name])), grad_mags[0][name], width=1.0, color="coral")
            ax_grad[c].set_ylim(0, grad_ylims[name] * 1.1 + 1e-8)
            ax_grad[c].set_title(f"{name} — gradients", fontsize=7)
            ax_grad[c].set_xlabel("Channel", fontsize=7)
            ax_grad[c].set_ylabel("Mean |grad|", fontsize=7)
            ax_grad[c].tick_params(labelsize=6)
            grad_bars.append(b_g)
        else:
            ax_grad[c].set_visible(False)
            grad_bars.append(None)

        if name in heat_vmaxes:
            raw_grad = replay_history[0]["gradients"][name]
            im_h = ax_heat[c].imshow(
                grad_heat[0][name], cmap="inferno",
                vmin=0, vmax=heat_vmaxes[name] + 1e-8,
                interpolation="nearest", aspect="auto",
            )
            fig.colorbar(im_h, ax=ax_heat[c], shrink=0.7, label="|∇|")
            if raw_grad.ndim == 3:
                ax_heat[c].set_title(f"{name} — gradient heatmap\n(mean |∇| over channels)", fontsize=7)
                ax_heat[c].set_xlabel("W", fontsize=7)
                ax_heat[c].set_ylabel("H", fontsize=7)
            else:
                side = math.ceil(math.sqrt(raw_grad.shape[0]))
                ax_heat[c].set_title(f"{name} — gradient heatmap\n(reshaped {side}×{side})", fontsize=7)
                ax_heat[c].set_xlabel("Feature (col)", fontsize=7)
                ax_heat[c].set_ylabel("Feature (row)", fontsize=7)
            ax_heat[c].tick_params(labelsize=6)
            heat_ims.append(im_h)
        else:
            ax_heat[c].set_visible(False)
            heat_ims.append(None)

    title_text = fig.suptitle("", fontsize=12)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")

        im_conv1_filt.set_data(conv1_rendered[frame])
        im_conv2_norm.set_data(conv2_hmaps[frame])
        fc1_data = np.full_like(fc1_timeline, np.nan)
        fc1_data[:, : frame + 1] = fc1_timeline[:, : frame + 1]
        im_fc1_norms.set_data(fc1_data)
        im_delta.set_data(delta_rendered[frame])

        for c, name in enumerate(replay_modules):
            for bar, h in zip(act_bars[c], act_mags[frame][name]):
                bar.set_height(h)
            if grad_bars[c] is not None:
                for bar, h in zip(grad_bars[c], grad_mags[frame][name]):
                    bar.set_height(h)
            if heat_ims[c] is not None and name in grad_heat[frame]:
                heat_ims[c].set_data(grad_heat[frame][name])

        return [
            im_conv1_filt, im_conv2_norm, im_fc1_norms, im_delta, title_text,
            *(b for bars in act_bars  for b in bars),
            *(b for bars in grad_bars if bars is not None for b in bars),
            *(im for im in heat_ims   if im is not None),
        ]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result
