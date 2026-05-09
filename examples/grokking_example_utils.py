"""Visualization and weight-rendering utilities for the grokking example."""

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
    train_acc_history: list[float],
    test_acc_history: list[float],
    snapshot_every: int,
    component_keys: list[tuple[str, str]],
    run_dir: Path,
    fps: int = 5,
) -> None:
    print("\n── Visualizations ──")

    video_path = run_dir / "overview.mp4"
    print(f"  Generating combined video  → {video_path.name} …")
    result = make_combined_video(
        weight_history, replay_history, replay_modules,
        train_acc_history, test_acc_history,
        snapshot_every, component_keys, video_path, fps=fps,
    )
    print(f"  Saved: {result.name}")

    curves_path = run_dir / "accuracy_curves.png"
    save_accuracy_curves(train_acc_history, test_acc_history, snapshot_every, curves_path)
    print(f"  Saved: {curves_path.name}")


# ---------------------------------------------------------------------------
# Weight rendering helpers
# ---------------------------------------------------------------------------


def _token_cosine_sim(emb_weight: np.ndarray, p: int) -> np.ndarray:
    """(P×P) cosine similarity between the P token embeddings (drops EQ_TOKEN row)."""
    emb = emb_weight[:p]
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    normed = emb / norms
    return normed @ normed.T


def _emb_fourier_power(emb_weight: np.ndarray, p: int) -> np.ndarray:
    """Mean Fourier power over embedding dims, DC component dropped."""
    emb = emb_weight[:p]
    fft = np.fft.rfft(emb, axis=0)
    power = (np.abs(fft) ** 2).mean(axis=1)
    return power[1:]


def _build_fourier_timeline(
    weight_history: list[dict[str, np.ndarray]], p: int,
) -> np.ndarray:
    """(P//2, num_snapshots) Fourier power matrix — grows left to right in video."""
    spectra = np.array([
        _emb_fourier_power(snap["token_emb.weight"], p)
        for snap in weight_history
    ])
    return spectra.T


def _build_norm_timeline(
    weight_history: list[dict[str, np.ndarray]],
    component_keys: list[tuple[str, str]],
) -> np.ndarray:
    """(num_components, num_snapshots) Frobenius norm matrix — grows left to right."""
    norms = np.array([
        [np.linalg.norm(snap[k]) for k, _ in component_keys]
        for snap in weight_history
    ])
    return norms.T


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
# Combined video
# ---------------------------------------------------------------------------


def make_combined_video(
    weight_history: list[dict[str, np.ndarray]],
    replay_history: list[dict[str, dict[str, np.ndarray]]],
    modules: list[str],
    train_acc_history: list[float],
    test_acc_history: list[float],
    snapshot_every: int,
    component_keys: list[tuple[str, str]],
    out_path: Path,
    fps: int = 5,
) -> Path:
    """Animate weights, activations, and gradients together in one video.

    Layout (rows × 4 cols):
      Row 0  — weights: cosine-sim | Fourier timeline | norm timeline | output-proj
      Row 1+ — per module: activation bars | gradient bars | gradient heatmap | accuracy curve
    """
    n_frames = len(weight_history)
    n_mods   = len(modules)
    p = weight_history[0]["token_emb.weight"].shape[0] - 1

    # ── precompute weight data ───────────────────────────────────────────
    cosine_sims      = [_token_cosine_sim(snap["token_emb.weight"], p) for snap in weight_history]
    fourier_timeline = _build_fourier_timeline(weight_history, p)
    fourier_vmax     = float(fourier_timeline.max()) or 1.0
    norm_timeline    = _build_norm_timeline(weight_history, component_keys)
    norm_vmax        = float(norm_timeline.max()) or 1.0
    output_projs     = [snap["output_proj.weight"] for snap in weight_history]
    out_abs_max      = float(np.abs(np.stack(output_projs)).max()) or 1.0

    # ── precompute replay data ───────────────────────────────────────────
    act_mags, grad_mags, grad_heat = precompute_replay_data(replay_history, modules)
    act_ylims   = {m: max(act_mags[e][m].max()  for e in range(n_frames)) for m in modules}
    grad_ylims  = {m: max(grad_mags[e][m].max() for e in range(n_frames)) for m in modules if m in grad_mags[0]}
    heat_vmaxes = {m: max(grad_heat[e][m].max() for e in range(n_frames)) for m in modules if m in grad_heat[0]}
    steps = [i * snapshot_every for i in range(n_frames)]

    # ── figure layout ────────────────────────────────────────────────────
    n_rows = 1 + n_mods
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, 4)

    # Row 0 — weight panels
    im_cos = axes[0, 0].imshow(
        cosine_sims[0], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
    )
    fig.colorbar(im_cos, ax=axes[0, 0], shrink=0.8, label="cosine sim")
    axes[0, 0].set_title(f"Token emb cosine similarity ({p}×{p})", fontsize=8)
    axes[0, 0].set_xlabel("Token b")
    axes[0, 0].set_ylabel("Token a")

    fourier_init = np.full_like(fourier_timeline, np.nan)
    fourier_init[:, 0] = fourier_timeline[:, 0]
    im_fourier = axes[0, 1].imshow(
        fourier_init, aspect="auto", cmap="plasma",
        vmin=0, vmax=fourier_vmax, interpolation="nearest",
    )
    fig.colorbar(im_fourier, ax=axes[0, 1], shrink=0.8, label="mean |FFT|²")
    axes[0, 1].set_title("Embedding Fourier power over training", fontsize=8)
    axes[0, 1].set_xlabel("Snapshot")
    axes[0, 1].set_ylabel("Frequency (DC dropped)")

    norm_init = np.full_like(norm_timeline, np.nan)
    norm_init[:, 0] = norm_timeline[:, 0]
    im_norm = axes[0, 2].imshow(
        norm_init, aspect="auto", cmap="viridis",
        vmin=0, vmax=norm_vmax, interpolation="nearest",
    )
    fig.colorbar(im_norm, ax=axes[0, 2], shrink=0.8, label="‖W‖_F")
    axes[0, 2].set_title("Component Frobenius norms", fontsize=8)
    axes[0, 2].set_xlabel("Snapshot")
    axes[0, 2].set_yticks(range(len(component_keys)))
    axes[0, 2].set_yticklabels([label for _, label in component_keys], fontsize=7)

    out_h, out_w = output_projs[0].shape
    im_out = axes[0, 3].imshow(
        output_projs[0], aspect="auto", cmap="RdBu_r",
        vmin=-out_abs_max, vmax=out_abs_max, interpolation="nearest",
    )
    fig.colorbar(im_out, ax=axes[0, 3], shrink=0.8, label="weight value")
    axes[0, 3].set_title(f"Output proj weights ({out_h}×{out_w})", fontsize=8)

    # Rows 1+ — replay panels (one row per module)
    act_bars_list: list[Any]  = []
    grad_bars_list: list[Any] = []
    heat_ims_list:  list[Any] = []
    acc_vline = None

    for r, name in enumerate(modules):
        row = 1 + r

        b_a = axes[row, 0].bar(
            np.arange(len(act_mags[0][name])), act_mags[0][name], width=1.0, color="steelblue",
        )
        axes[row, 0].set_ylim(0, act_ylims[name] * 1.1 + 1e-8)
        axes[row, 0].set_title(f"{name} — activations", fontsize=8)
        axes[row, 0].set_xlabel("Feature")
        axes[row, 0].set_ylabel("Mean |activation|")
        act_bars_list.append(b_a)

        if name in grad_ylims:
            b_g = axes[row, 1].bar(
                np.arange(len(grad_mags[0][name])), grad_mags[0][name], width=1.0, color="coral",
            )
            axes[row, 1].set_ylim(0, grad_ylims[name] * 1.1 + 1e-8)
            axes[row, 1].set_title(f"{name} — gradients", fontsize=8)
            axes[row, 1].set_xlabel("Feature")
            axes[row, 1].set_ylabel("Mean |gradient|")
            grad_bars_list.append(b_g)
        else:
            axes[row, 1].set_visible(False)
            grad_bars_list.append(None)

        if name in heat_vmaxes:
            raw_grad = replay_history[0]["gradients"][name]
            im = axes[row, 2].imshow(
                grad_heat[0][name], cmap="inferno",
                vmin=0, vmax=heat_vmaxes[name] + 1e-8,
                interpolation="nearest", aspect="auto",
            )
            fig.colorbar(im, ax=axes[row, 2], shrink=0.8, label="|∇|")
            if raw_grad.ndim == 3:
                axes[row, 2].set_title(f"{name} — gradient heatmap\n(mean |∇| over channels)", fontsize=7)
            else:
                side = math.ceil(math.sqrt(raw_grad.shape[0]))
                axes[row, 2].set_title(f"{name} — gradient heatmap\n(reshaped {side}×{side})", fontsize=7)
            heat_ims_list.append(im)
        else:
            axes[row, 2].set_visible(False)
            heat_ims_list.append(None)

        # Accuracy curve with moving marker — only on the first replay row
        if r == 0:
            axes[row, 3].plot(steps, train_acc_history, color="steelblue", label="Train acc", linewidth=1.5)
            axes[row, 3].plot(steps, test_acc_history,  color="coral",     label="Test acc",  linewidth=1.5)
            axes[row, 3].set_xlabel("Step")
            axes[row, 3].set_ylabel("Accuracy")
            axes[row, 3].set_title("Accuracy over training", fontsize=8)
            axes[row, 3].legend(fontsize=7)
            axes[row, 3].grid(True, alpha=0.3)
            axes[row, 3].set_xlim(steps[0], steps[-1])
            acc_vline = axes[row, 3].axvline(steps[0], color="black", linewidth=1.5, linestyle="--")
        else:
            axes[row, 3].set_visible(False)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        step = frame * snapshot_every
        tr = train_acc_history[frame] if frame < len(train_acc_history) else 0.0
        te = test_acc_history[frame]  if frame < len(test_acc_history)  else 0.0
        title_text.set_text(f"Step {step}  |  train acc={tr:.1%}  test acc={te:.1%}")

        im_cos.set_data(cosine_sims[frame])

        fourier_data = np.full_like(fourier_timeline, np.nan)
        fourier_data[:, : frame + 1] = fourier_timeline[:, : frame + 1]
        im_fourier.set_data(fourier_data)

        norm_data = np.full_like(norm_timeline, np.nan)
        norm_data[:, : frame + 1] = norm_timeline[:, : frame + 1]
        im_norm.set_data(norm_data)

        im_out.set_data(output_projs[frame])

        artists: list = [title_text, im_cos, im_fourier, im_norm, im_out]

        for r, name in enumerate(modules):
            for bar, h in zip(act_bars_list[r], act_mags[frame][name]):
                bar.set_height(h)
                artists.append(bar)
            if grad_bars_list[r] is not None:
                for bar, h in zip(grad_bars_list[r], grad_mags[frame][name]):
                    bar.set_height(h)
                    artists.append(bar)
            if heat_ims_list[r] is not None and name in grad_heat[frame]:
                heat_ims_list[r].set_data(grad_heat[frame][name])
                artists.append(heat_ims_list[r])

        if acc_vline is not None:
            acc_vline.set_xdata([steps[frame], steps[frame]])
            artists.append(acc_vline)

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
# Static accuracy curves
# ---------------------------------------------------------------------------


def save_accuracy_curves(
    train_accs: list[float],
    test_accs: list[float],
    snapshot_every: int,
    out_path: Path,
) -> None:
    steps = [i * snapshot_every for i in range(len(train_accs))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(steps, train_accs, label="Train acc")
    axes[0].plot(steps, test_accs,  label="Test acc")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Grokking: accuracy over training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, train_accs, label="Train acc")
    axes[1].plot(steps, test_accs,  label="Test acc")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Step (log scale)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Grokking: accuracy (log x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
