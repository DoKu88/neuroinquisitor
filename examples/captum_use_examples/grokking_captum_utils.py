"""Visualization utilities for the Grokking Captum example."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


TOKEN_LABELS = ["operand a\n(first input)", "operand b\n(second input)", "equals =\n(output position)"]
TOKEN_LABELS_SHORT = ["a", "b", "="]
TOKEN_COLORS = ["tab:blue", "tab:orange", "tab:green"]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate_grokking_captum_visualizations(
    lig_evolution: np.ndarray,       # (n_snapshots, 3)
    conductance: np.ndarray,          # (n_checkpoints, 3)
    checkpoint_indices: list[int],
    train_accs: list[float],
    test_accs: list[float],
    snapshot_every: int,
    run_dir: Path,
    fps: int = 10,
) -> None:
    print("\n── Visualizations ──")

    p1 = run_dir / "token_attribution_evolution.png"
    _save_attribution_evolution(lig_evolution, test_accs, snapshot_every, p1)
    print(f"  Saved: {p1.name}")

    p2 = save_attribution_evolution_video(lig_evolution, snapshot_every, run_dir / "attribution_evolution", fps=fps)
    print(f"  Saved: {p2.name}")

    p3 = run_dir / "conductance_at_checkpoints.png"
    _save_conductance_chart(conductance, checkpoint_indices, snapshot_every, p3)
    print(f"  Saved: {p3.name}")

    p4 = run_dir / "accuracy_curves.png"
    _save_accuracy_curves(train_accs, test_accs, snapshot_every, p4)
    print(f"  Saved: {p4.name}")

    p5 = save_symmetry_scatter_video(lig_evolution, snapshot_every, run_dir / "ab_symmetry", fps=fps)
    print(f"  Saved: {p5.name}")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def save_attribution_evolution_video(
    lig_evolution: np.ndarray,
    snapshot_every: int,
    out_stem: Path,
    fps: int = 10,
) -> Path:
    """Animated line chart: token attributions grow over training, frame by frame.

    Left panel: lines for operand a, operand b, and equals = extend rightward
    with each frame.  Both panels share the same y-axis scale so the bar chart
    directly corresponds to the current point on the lines.

    The key signal: do a and b converge to the same value?  If so, the model
    has learned the symmetric addition algorithm (grokking generalisation).
    """
    n_snaps = lig_evolution.shape[0]
    steps   = np.array([(i + 1) * snapshot_every for i in range(n_snaps)])
    y_max   = lig_evolution.max() * 1.1

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle("Which input token drives predictions? (LayerIntegratedGradients)", fontsize=11)

    # ── left: growing line chart ──
    line_a,  = ax_line.plot([], [], color="tab:blue",   linewidth=2,   label="operand a")
    line_b,  = ax_line.plot([], [], color="tab:orange", linewidth=2,   label="operand b")
    line_eq, = ax_line.plot([], [], color="tab:green",  linewidth=1.2, linestyle="--",
                             alpha=0.7, label="equals = (output position)")
    vline = ax_line.axvline(x=steps[0], color="gray", linestyle=":", linewidth=1, alpha=0.6)

    ax_line.set_xlim(steps[0], steps[-1])
    ax_line.set_ylim(0, y_max)
    ax_line.set_xlabel("Training step")
    ax_line.set_ylabel("Mean token attribution (L2 norm)")
    ax_line.legend(loc="upper right", fontsize=9)
    ax_line.grid(True, alpha=0.3)
    ax_line.set_title(
        "a and b converging → model learned the algorithm\n"
        "a and b diverging → memorisation only",
        fontsize=9,
    )

    # ── right: bar chart (same y scale) ──
    bars = ax_bar.bar([0, 1, 2], [0, 0, 0], color=TOKEN_COLORS, alpha=0.85)
    ax_bar.set_xticks([0, 1, 2])
    ax_bar.set_xticklabels(TOKEN_LABELS_SHORT, fontsize=9)
    ax_bar.set_ylim(0, y_max)
    ax_bar.set_ylabel("Attribution (L2 norm)")
    ax_bar.grid(True, axis="y", alpha=0.3)

    def update(frame: int) -> list:
        line_a.set_data(steps[: frame + 1], lig_evolution[: frame + 1, 0])
        line_b.set_data(steps[: frame + 1], lig_evolution[: frame + 1, 1])
        line_eq.set_data(steps[: frame + 1], lig_evolution[: frame + 1, 2])
        vline.set_xdata([steps[frame], steps[frame]])
        for bar, val in zip(bars, lig_evolution[frame]):
            bar.set_height(val)
        ax_bar.set_title(f"step {steps[frame]:,}", fontsize=10)
        return [line_a, line_b, line_eq, vline, *bars]

    ani = animation.FuncAnimation(fig, update, frames=n_snaps, interval=1000 // fps, blit=False)
    out_path = _save_animation(ani, out_stem, fps)
    plt.close(fig)
    return out_path


def save_symmetry_scatter_video(
    lig_evolution: np.ndarray,
    snapshot_every: int,
    out_stem: Path,
    fps: int = 10,
) -> Path:
    """Animated scatter: a-attribution vs b-attribution, dots appear one per frame.

    The trail reveals the model's trajectory through attribution space.
    Points near the diagonal mean the model treats both operands equally —
    the hallmark of the grokking generalisation algorithm.
    """
    a_vals = lig_evolution[:, 0]
    b_vals = lig_evolution[:, 1]
    n_snaps = len(a_vals)
    steps   = np.array([(i + 1) * snapshot_every for i in range(n_snaps)])
    lim     = max(a_vals.max(), b_vals.max()) * 1.08

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, linewidth=1.2,
            label="a = b  (perfect symmetry)")

    # Ghost of full trajectory
    ax.scatter(a_vals, b_vals, c="lightgray", s=15, alpha=0.25, zorder=2)

    # Growing trail (colored by training step)
    scat = ax.scatter([], [], c=[], cmap="viridis", s=35, alpha=0.9, zorder=3,
                      vmin=steps[0], vmax=steps[-1])
    current_dot, = ax.plot([], [], "ro", ms=9, zorder=5, label="current step")

    fig.colorbar(scat, ax=ax, label="Training step",
                 format=plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    step_label = ax.text(0.05, 0.95, "", transform=ax.transAxes,
                         va="top", fontsize=10, fontweight="bold")

    ax.set_xlabel("LIG attribution — operand a (first input)", fontsize=10)
    ax.set_ylabel("LIG attribution — operand b (second input)", fontsize=10)
    ax.set_title(
        "a vs b attribution: trajectory through training\n"
        "Trail converging to diagonal = model learning symmetric algorithm",
        fontsize=10,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    def update(frame: int) -> list:
        xy = np.column_stack([a_vals[: frame + 1], b_vals[: frame + 1]])
        scat.set_offsets(xy)
        scat.set_array(steps[: frame + 1])
        current_dot.set_data([a_vals[frame]], [b_vals[frame]])
        step_label.set_text(f"step {steps[frame]:,}")
        return [scat, current_dot, step_label]

    ani = animation.FuncAnimation(fig, update, frames=n_snaps, interval=1000 // fps, blit=False)
    out_path = _save_animation(ani, out_stem, fps)
    plt.close(fig)
    return out_path


def _save_animation(ani: animation.FuncAnimation, out_stem: Path, fps: int) -> Path:
    mp4 = out_stem.with_suffix(".mp4")
    try:
        ani.save(str(mp4), writer=animation.FFMpegWriter(fps=fps))
        return mp4
    except Exception:
        gif = out_stem.with_suffix(".gif")
        ani.save(str(gif), writer=animation.PillowWriter(fps=fps))
        return gif


# ---------------------------------------------------------------------------
# Static plot helpers
# ---------------------------------------------------------------------------


def _save_attribution_evolution(
    lig_evolution: np.ndarray,
    test_accs: list[float],
    snapshot_every: int,
    out_path: Path,
) -> None:
    steps = [(i + 1) * snapshot_every for i in range(lig_evolution.shape[0])]

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = ["operand a  (pos 0)", "operand b  (pos 1)", "equals =  (pos 2)"]
    for i, (label, color) in enumerate(zip(labels, TOKEN_COLORS)):
        ax1.plot(steps, lig_evolution[:, i], label=label, color=color, linewidth=1.5)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean token attribution (L2 norm)")
    ax1.set_title(
        "LayerIntegratedGradients — token attribution evolution\n"
        "Symmetric a/b lines signal algorithmic generalisation",
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if test_accs:
        ax2 = ax1.twinx()
        acc_steps = [(i + 1) * snapshot_every for i in range(len(test_accs))]
        ax2.plot(acc_steps, [a * 100 for a in test_accs], "--",
                 color="gray", linewidth=1.0, alpha=0.7, label="test acc %")
        ax2.set_ylabel("Test accuracy (%)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="upper right")

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_conductance_chart(
    conductance: np.ndarray,
    checkpoint_indices: list[int],
    snapshot_every: int,
    out_path: Path,
) -> None:
    """Two-panel chart: = output token (top) and operand tokens (bottom, own scale)."""
    steps       = [(idx + 1) * snapshot_every for idx in checkpoint_indices]
    step_labels = [f"step {s:,}" for s in steps]
    x           = np.arange(len(checkpoint_indices))
    width       = 0.35

    fig, (ax_eq, ax_ab) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    fig.suptitle(
        "LayerConductance on transformer encoder\n"
        "How much does the transformer block contribute at each token position?",
        fontsize=10,
    )

    # Top: = token
    ax_eq.bar(x, conductance[:, 2], color="tab:green", alpha=0.85,
              label="equals = (output position)")
    ax_eq.set_xticks(x)
    ax_eq.set_xticklabels(step_labels)
    ax_eq.set_ylabel("Conductance (L2 norm)")
    ax_eq.set_title(
        "Equals token — dominates because output_proj reads h[:, −1] (this position only)",
        fontsize=9,
    )
    ax_eq.legend(fontsize=9)
    ax_eq.grid(True, axis="y", alpha=0.3)

    # Bottom: a and b — near-zero by architecture, but shown on their own scale
    ax_ab.bar(x - width / 2, conductance[:, 0], width, color="tab:blue",  alpha=0.85,
              label="operand a (first input)")
    ax_ab.bar(x + width / 2, conductance[:, 1], width, color="tab:orange", alpha=0.85,
              label="operand b (second input)")
    ax_ab.set_xticks(x)
    ax_ab.set_xticklabels(step_labels)
    ax_ab.set_xlabel("Training step")
    ax_ab.set_ylabel("Conductance (L2 norm)")
    ax_ab.set_ylim(bottom=0)
    ax_ab.set_title(
        "Operand tokens — near-zero: transformer routes all output-relevant signal through =",
        fontsize=9,
    )
    ax_ab.legend(fontsize=9)
    ax_ab.grid(True, axis="y", alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_accuracy_curves(
    train_accs: list[float],
    test_accs: list[float],
    snapshot_every: int,
    out_path: Path,
) -> None:
    steps = [(i + 1) * snapshot_every for i in range(len(train_accs))]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(steps, [a * 100 for a in train_accs], label="Train accuracy", linewidth=1.5)
    ax.plot(steps, [a * 100 for a in test_accs],  label="Test accuracy",  linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Grokking — accuracy dynamics\n"
        "(train→100% = memorisation phase;  test jump = generalisation / grokking)",
    )
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
