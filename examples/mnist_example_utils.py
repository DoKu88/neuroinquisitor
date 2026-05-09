"""Visualization and weight-rendering utilities for the MNIST example."""

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
    accuracy_history: list[float],
    train_loss_history: list[float],
    test_loss_history: list[float],
    run_dir: Path,
    fps: int = 4,
) -> None:
    print("\n── Visualizations ──")

    for name, fn, args in [
        ("overview.mp4",      make_combined_video,    (weight_history, replay_history, replay_modules, accuracy_history)),
        ("weights.mp4",       make_weights_video,     (weight_history, accuracy_history)),
        ("activations.mp4",   make_activations_video, (replay_history, replay_modules, accuracy_history)),
        ("gradients.mp4",     make_gradients_video,   (replay_history, replay_modules, accuracy_history)),
    ]:
        print(f"  Generating {name:<22} …", end=" ", flush=True)
        result = fn(*args, out_path=run_dir / name, fps=fps)
        print(f"saved → {result.name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"  Saved: {curves_path.name}")


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


def render_gray_filters(
    weight: np.ndarray,
    per_filter_lims: list[tuple[float, float]],
) -> np.ndarray:
    """Tile (out, 1, H, W) grayscale filters as a grid with fixed per-filter bounds."""
    out_ch, _, H, W = weight.shape
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.full((rows * (H + pad) - pad, cols * (W + pad) - pad), 0.5)
    for i in range(out_ch):
        r, c = divmod(i, cols)
        filt = weight[i, 0].copy()
        lo, hi = per_filter_lims[i]
        filt = np.clip((filt - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def render_delta_gray(
    weight: np.ndarray,
    weight_0: np.ndarray,
    vmax: float,
) -> np.ndarray:
    """Tile |W_t − W_0| per filter as a grayscale grid normalised by vmax."""
    delta = np.abs(weight - weight_0)
    out_ch, _, H, W = delta.shape
    cols = math.ceil(math.sqrt(out_ch))
    rows = math.ceil(out_ch / cols)
    pad = 1
    canvas = np.zeros((rows * (H + pad) - pad, cols * (W + pad) - pad))
    for i in range(out_ch):
        r, c = divmod(i, cols)
        filt = np.clip(delta[i, 0] / (vmax + 1e-8), 0.0, 1.0)
        canvas[r * (H + pad): r * (H + pad) + H, c * (W + pad): c * (W + pad) + W] = filt
    return canvas


def render_conv2_norm_heatmap(weight: np.ndarray) -> np.ndarray:
    """Per-filter ‖W‖₂ for (16, in, H, W) as a near-square grid."""
    norms = np.sqrt((weight ** 2).sum(axis=(1, 2, 3)))
    side = math.ceil(math.sqrt(len(norms)))
    padded = np.full(side * side, np.nan)
    padded[: len(norms)] = norms
    return padded.reshape(side, side)


def build_fc1_timeline(weight_history: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Return (128, num_epochs) matrix of per-neuron row-norms."""
    row_norms = np.array([
        np.linalg.norm(snap["fc1.weight"], axis=1)
        for snap in weight_history
    ])
    return row_norms.T  # (128, num_epochs)


# ---------------------------------------------------------------------------
# Replay preprocessing
# ---------------------------------------------------------------------------


def precompute_replay_data(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
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


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------


def _animate(fig: plt.Figure, update, n_frames: int, fps: int, out_path: Path) -> Path:
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Weights video (2×2)
# ---------------------------------------------------------------------------


def make_weights_video(
    weight_history: list[dict[str, np.ndarray]],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """2×2 grid: conv1 filters | conv2 norms / fc1 timeline | conv1 delta."""
    n_frames = len(weight_history)
    weight_0 = weight_history[0]

    conv1_lims     = compute_conv1_filter_lims(weight_history)
    conv2_heatmaps = [render_conv2_norm_heatmap(snap["conv2.weight"]) for snap in weight_history]
    conv2_vmax     = float(max(float(h[~np.isnan(h)].max()) for h in conv2_heatmaps))
    fc1_timeline   = build_fc1_timeline(weight_history)
    fc1_vmax       = float(fc1_timeline.max())
    delta_vmax     = float(max(
        np.abs(snap["conv1.weight"] - weight_0["conv1.weight"]).max()
        for snap in weight_history
    ))
    conv1_rendered = [render_gray_filters(snap["conv1.weight"], conv1_lims) for snap in weight_history]
    delta_rendered = [render_delta_gray(snap["conv1.weight"], weight_0["conv1.weight"], delta_vmax)
                      for snap in weight_history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    im_conv1 = axes[0, 0].imshow(conv1_rendered[0], cmap="gray", interpolation="nearest")
    axes[0, 0].set_title("Conv1 — grayscale filters (fixed bounds)", fontsize=8)
    axes[0, 0].axis("off")

    im_conv2 = axes[0, 1].imshow(conv2_heatmaps[0], cmap="viridis", vmin=0, vmax=conv2_vmax,
                                  interpolation="nearest")
    fig.colorbar(im_conv2, ax=axes[0, 1], shrink=0.8, label="‖W‖₂")
    axes[0, 1].set_title("Conv2 — per-filter ‖W‖₂", fontsize=8)
    axes[0, 1].axis("off")

    fc1_init = np.full_like(fc1_timeline, np.nan)
    fc1_init[:, 0] = fc1_timeline[:, 0]
    im_fc1 = axes[1, 0].imshow(fc1_init, aspect="auto", cmap="plasma",
                                vmin=0, vmax=fc1_vmax, interpolation="nearest")
    fig.colorbar(im_fc1, ax=axes[1, 0], shrink=0.8, label="‖row‖₂")
    axes[1, 0].set_title("FC1 — neuron row-norms over epochs", fontsize=8)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Neuron index")

    im_delta = axes[1, 1].imshow(delta_rendered[0], cmap="hot", interpolation="nearest")
    axes[1, 1].set_title("Conv1  |W_t − W_0|  (init delta)", fontsize=8)
    axes[1, 1].axis("off")

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        im_conv1.set_data(conv1_rendered[frame])
        im_conv2.set_data(conv2_heatmaps[frame])
        fc1_data = np.full_like(fc1_timeline, np.nan)
        fc1_data[:, : frame + 1] = fc1_timeline[:, : frame + 1]
        im_fc1.set_data(fc1_data)
        im_delta.set_data(delta_rendered[frame])
        return [title_text, im_conv1, im_conv2, im_fc1, im_delta]

    return _animate(fig, update, n_frames, fps, out_path)


# ---------------------------------------------------------------------------
# Activations video (1×n_mods)
# ---------------------------------------------------------------------------


def make_activations_video(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """One row of activation bar charts, one column per module."""
    n_frames = len(replay_history)
    n_mods   = len(modules)

    act_mags, _, _ = precompute_replay_data(replay_history, modules)
    act_ylims = {m: max(act_mags[e][m].max() for e in range(n_frames)) for m in modules}

    fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 4), constrained_layout=True)
    if n_mods == 1:
        axes = [axes]

    bars: list[Any] = []
    for c, name in enumerate(modules):
        b = axes[c].bar(np.arange(len(act_mags[0][name])), act_mags[0][name], width=1.0, color="steelblue")
        axes[c].set_ylim(0, act_ylims[name] * 1.1 + 1e-8)
        axes[c].set_title(f"{name} — activations", fontsize=9)
        axes[c].set_xlabel("Channel")
        axes[c].set_ylabel("Mean |activation|")
        bars.append(b)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        artists: list = [title_text]
        for c, name in enumerate(modules):
            for bar, h in zip(bars[c], act_mags[frame][name]):
                bar.set_height(h)
                artists.append(bar)
        return artists

    return _animate(fig, update, n_frames, fps, out_path)


# ---------------------------------------------------------------------------
# Gradients video (2×n_mods)
# ---------------------------------------------------------------------------


def make_gradients_video(
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Two rows per module: gradient bar charts on top, gradient heatmaps below."""
    n_frames = len(replay_history)
    n_mods   = len(modules)

    _, grad_mags, grad_heat = precompute_replay_data(replay_history, modules)
    grad_ylims  = {m: max(grad_mags[e][m].max() for e in range(n_frames)) for m in modules if m in grad_mags[0]}
    heat_vmaxes = {m: max(grad_heat[e][m].max() for e in range(n_frames)) for m in modules if m in grad_heat[0]}

    fig, axes = plt.subplots(2, n_mods, figsize=(5 * n_mods, 7), constrained_layout=True)
    if n_mods == 1:
        axes = axes.reshape(2, 1)

    grad_bars: list[Any] = []
    heat_ims:  list[Any] = []

    for c, name in enumerate(modules):
        if name in grad_ylims:
            b_g = axes[0, c].bar(np.arange(len(grad_mags[0][name])), grad_mags[0][name],
                                  width=1.0, color="coral")
            axes[0, c].set_ylim(0, grad_ylims[name] * 1.1 + 1e-8)
            axes[0, c].set_title(f"{name} — gradients", fontsize=9)
            axes[0, c].set_xlabel("Channel")
            axes[0, c].set_ylabel("Mean |gradient|")
            grad_bars.append(b_g)
        else:
            axes[0, c].set_visible(False)
            grad_bars.append(None)

        if name in heat_vmaxes:
            raw_grad = replay_history[0]["gradients"][name]
            im = axes[1, c].imshow(
                grad_heat[0][name], cmap="inferno",
                vmin=0, vmax=heat_vmaxes[name] + 1e-8,
                interpolation="nearest", aspect="auto",
            )
            fig.colorbar(im, ax=axes[1, c], shrink=0.8, label="|∇|")
            if raw_grad.ndim == 3:
                axes[1, c].set_title(f"{name} — gradient heatmap\n(mean |∇| over channels)", fontsize=7)
                axes[1, c].set_xlabel("W")
                axes[1, c].set_ylabel("H")
            else:
                side = math.ceil(math.sqrt(raw_grad.shape[0]))
                axes[1, c].set_title(f"{name} — gradient heatmap\n(reshaped {side}×{side})", fontsize=7)
                axes[1, c].set_xlabel("Feature (col)")
                axes[1, c].set_ylabel("Feature (row)")
            heat_ims.append(im)
        else:
            axes[1, c].set_visible(False)
            heat_ims.append(None)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")
        artists: list = [title_text]
        for c, name in enumerate(modules):
            if grad_bars[c] is not None:
                for bar, h in zip(grad_bars[c], grad_mags[frame][name]):
                    bar.set_height(h)
                    artists.append(bar)
            if heat_ims[c] is not None and name in grad_heat[frame]:
                heat_ims[c].set_data(grad_heat[frame][name])
                artists.append(heat_ims[c])
        return artists

    return _animate(fig, update, n_frames, fps, out_path)


# ---------------------------------------------------------------------------
# Combined video
# ---------------------------------------------------------------------------


def make_combined_video(
    weight_history: list[dict[str, np.ndarray]],
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
    accuracy_history: list[float],
    out_path: Path,
    fps: int = 4,
) -> Path:
    """Animate weights, activations, and gradients together in one video.

    Layout:
      Top (2×2)       — weights: conv1 filters | conv2 norms
                                 fc1 timeline  | conv1 delta
      Bottom (3×mods) — per module: activation bars | gradient bars | gradient heatmap
    """
    n_frames = len(weight_history)
    n_mods   = len(modules)
    weight_0 = weight_history[0]

    # ── precompute weight data ───────────────────────────────────────────
    conv1_lims     = compute_conv1_filter_lims(weight_history)
    conv2_heatmaps = [render_conv2_norm_heatmap(snap["conv2.weight"]) for snap in weight_history]
    conv2_vmax     = float(max(float(h[~np.isnan(h)].max()) for h in conv2_heatmaps))
    fc1_timeline   = build_fc1_timeline(weight_history)
    fc1_vmax       = float(fc1_timeline.max())
    delta_vmax     = float(max(
        np.abs(snap["conv1.weight"] - weight_0["conv1.weight"]).max()
        for snap in weight_history
    ))
    conv1_rendered = [render_gray_filters(snap["conv1.weight"], conv1_lims) for snap in weight_history]
    delta_rendered = [render_delta_gray(snap["conv1.weight"], weight_0["conv1.weight"], delta_vmax)
                      for snap in weight_history]

    # ── precompute replay data ───────────────────────────────────────────
    act_mags, grad_mags, grad_heat = precompute_replay_data(replay_history, modules)
    act_ylims   = {m: max(act_mags[e][m].max()  for e in range(n_frames)) for m in modules}
    grad_ylims  = {m: max(grad_mags[e][m].max() for e in range(n_frames)) for m in modules if m in grad_mags[0]}
    heat_vmaxes = {m: max(grad_heat[e][m].max() for e in range(n_frames)) for m in modules if m in grad_heat[0]}

    # ── figure with two stacked GridSpecs ────────────────────────────────
    fig_w  = max(10, 5 * n_mods)
    fig_h  = 18
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Weight block: 2×2, occupies top 38% of figure (below suptitle)
    gs_w = fig.add_gridspec(2, 2, left=0.08, right=0.92, top=0.92, bottom=0.56,
                             hspace=0.45, wspace=0.35)
    # Replay block: 3×n_mods, occupies bottom 52%
    gs_r = fig.add_gridspec(3, n_mods, left=0.05, right=0.95, top=0.50, bottom=0.04,
                             hspace=0.45, wspace=0.35)

    # Row labels on the left margin
    _label_kw = dict(rotation=90, va="center", ha="center", fontsize=11,
                     fontweight="bold", color="#333333")
    fig.text(0.015, (0.92 + 0.56) / 2,      "Weights",     **_label_kw)
    fig.text(0.015, 0.50 - (0.50 - 0.04) / 6, "Activations", **_label_kw)
    fig.text(0.015, 0.50 - (0.50 - 0.04) * 2 / 3, "Gradients",   **_label_kw)

    ax_conv1 = fig.add_subplot(gs_w[0, 0])
    ax_conv2 = fig.add_subplot(gs_w[0, 1])
    ax_fc1   = fig.add_subplot(gs_w[1, 0])
    ax_delta = fig.add_subplot(gs_w[1, 1])

    im_conv1 = ax_conv1.imshow(conv1_rendered[0], cmap="gray", interpolation="nearest")
    ax_conv1.set_title("Conv1 — grayscale filters (fixed bounds)", fontsize=8)
    ax_conv1.axis("off")

    im_conv2 = ax_conv2.imshow(conv2_heatmaps[0], cmap="viridis", vmin=0, vmax=conv2_vmax,
                                interpolation="nearest")
    fig.colorbar(im_conv2, ax=ax_conv2, shrink=0.8, label="‖W‖₂")
    ax_conv2.set_title("Conv2 — per-filter ‖W‖₂", fontsize=8)
    ax_conv2.axis("off")

    fc1_init = np.full_like(fc1_timeline, np.nan)
    fc1_init[:, 0] = fc1_timeline[:, 0]
    im_fc1 = ax_fc1.imshow(fc1_init, aspect="auto", cmap="plasma",
                            vmin=0, vmax=fc1_vmax, interpolation="nearest")
    fig.colorbar(im_fc1, ax=ax_fc1, shrink=0.8, label="‖row‖₂")
    ax_fc1.set_title("FC1 — neuron row-norms over epochs", fontsize=8)
    ax_fc1.set_xlabel("Epoch")
    ax_fc1.set_ylabel("Neuron index")

    im_delta = ax_delta.imshow(delta_rendered[0], cmap="hot", interpolation="nearest")
    ax_delta.set_title("Conv1  |W_t − W_0|  (init delta)", fontsize=8)
    ax_delta.axis("off")

    # Replay axes
    act_bars:  list[Any] = []
    grad_bars: list[Any] = []
    heat_ims:  list[Any] = []

    for c, name in enumerate(modules):
        ax_act  = fig.add_subplot(gs_r[0, c])
        ax_grad = fig.add_subplot(gs_r[1, c])
        ax_heat = fig.add_subplot(gs_r[2, c])

        b_a = ax_act.bar(np.arange(len(act_mags[0][name])), act_mags[0][name], width=1.0, color="steelblue")
        ax_act.set_ylim(0, act_ylims[name] * 1.1 + 1e-8)
        ax_act.set_title(f"{name} — activations", fontsize=8)
        ax_act.set_xlabel("Channel")
        ax_act.set_ylabel("Mean |activation|")
        act_bars.append(b_a)

        if name in grad_ylims:
            b_g = ax_grad.bar(np.arange(len(grad_mags[0][name])), grad_mags[0][name], width=1.0, color="coral")
            ax_grad.set_ylim(0, grad_ylims[name] * 1.1 + 1e-8)
            ax_grad.set_title(f"{name} — gradients", fontsize=8)
            ax_grad.set_xlabel("Channel")
            ax_grad.set_ylabel("Mean |gradient|")
            grad_bars.append(b_g)
        else:
            ax_grad.set_visible(False)
            grad_bars.append(None)

        if name in heat_vmaxes:
            raw_grad = replay_history[0]["gradients"][name]
            im = ax_heat.imshow(
                grad_heat[0][name], cmap="inferno",
                vmin=0, vmax=heat_vmaxes[name] + 1e-8,
                interpolation="nearest", aspect="auto",
            )
            fig.colorbar(im, ax=ax_heat, shrink=0.8, label="|∇|")
            if raw_grad.ndim == 3:
                ax_heat.set_title(f"{name} — gradient heatmap\n(mean |∇| over channels)", fontsize=7)
                ax_heat.set_xlabel("W")
                ax_heat.set_ylabel("H")
            else:
                side = math.ceil(math.sqrt(raw_grad.shape[0]))
                ax_heat.set_title(f"{name} — gradient heatmap\n(reshaped {side}×{side})", fontsize=7)
                ax_heat.set_xlabel("Feature (col)")
                ax_heat.set_ylabel("Feature (row)")
            heat_ims.append(im)
        else:
            ax_heat.set_visible(False)
            heat_ims.append(None)

    title_text = fig.suptitle("", fontsize=12, y=0.97)

    def update(frame: int) -> list:
        acc = accuracy_history[frame] if frame < len(accuracy_history) else 0.0
        title_text.set_text(f"Epoch {frame + 1}  |  Test accuracy: {acc:.1%}")

        im_conv1.set_data(conv1_rendered[frame])
        im_conv2.set_data(conv2_heatmaps[frame])
        fc1_data = np.full_like(fc1_timeline, np.nan)
        fc1_data[:, : frame + 1] = fc1_timeline[:, : frame + 1]
        im_fc1.set_data(fc1_data)
        im_delta.set_data(delta_rendered[frame])

        artists: list = [title_text, im_conv1, im_conv2, im_fc1, im_delta]

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

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Static loss/accuracy curves
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
    ax_loss.set_title("MNIST — loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in accuracy_history], marker="o", color="tab:green")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_title("MNIST — test accuracy")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
