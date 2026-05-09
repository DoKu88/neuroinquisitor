"""Visualization helpers for multi_arch_showcase.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCH_COLORS = {
    "mlp":         "tab:blue",
    "cnn":         "tab:orange",
    "transformer": "tab:green",
}
ARCH_LABELS = {
    "mlp":         "TinyMLP (FC)",
    "cnn":         "SmallCNN (1-D conv)",
    "transformer": "TinyTransformer (1-layer encoder)",
}


def generate_visualizations(arch_summaries: dict, base_dir: Path) -> None:
    """Write all three visualization files to base_dir."""
    print("\n── Visualizations ──")

    p1 = base_dir / "loss_curves.png"
    _save_loss_curves(arch_summaries, p1)
    print(f"  Saved: {p1.name}")

    p2 = base_dir / "weight_evolution.png"
    _save_weight_evolution(arch_summaries, p2)
    print(f"  Saved: {p2.name}")

    p3 = base_dir / "activation_drift.png"
    _save_activation_drift(arch_summaries, p3)
    print(f"  Saved: {p3.name}")


# ---------------------------------------------------------------------------
# Loss curves — all three architectures on one plot
# ---------------------------------------------------------------------------


def _save_loss_curves(arch_summaries: dict, out_path: Path) -> None:
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    fig.suptitle("Loss curves — all three architectures", fontsize=11)

    for arch, s in arch_summaries.items():
        color = ARCH_COLORS[arch]
        label = ARCH_LABELS[arch]
        epochs = range(len(s["train_losses"]))
        ax_train.plot(epochs, s["train_losses"], color=color, linewidth=1.8, label=label)
        ax_test.plot(epochs,  s["test_losses"],  color=color, linewidth=1.8, label=label)

    for ax, title in [(ax_train, "Train loss"), (ax_test, "Test loss")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Weight evolution — one column per architecture, two rows (first / last layer)
# ---------------------------------------------------------------------------


def _weight_norms_over_epochs(
    weight_history: dict[int, dict[str, np.ndarray]],
) -> dict[str, dict[int, float]]:
    """Compute L2 norm per layer per epoch from weight_history."""
    layer_epoch_norm: dict[str, dict[int, float]] = {}
    for epoch, tensors in sorted(weight_history.items()):
        for layer, arr in tensors.items():
            layer_epoch_norm.setdefault(layer, {})[epoch] = float(np.linalg.norm(arr))
    return layer_epoch_norm


def _save_weight_evolution(arch_summaries: dict, out_path: Path) -> None:
    archs = list(arch_summaries.keys())
    n_cols = len(archs)

    # Choose two representative layers per architecture (first weight, last weight)
    def _weight_layers(s: dict) -> list[str]:
        layers = [k for k in s["weight_history"][0].keys() if k.endswith(".weight")]
        return [layers[0], layers[-1]] if len(layers) >= 2 else layers

    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), constrained_layout=True,
    )
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle("Weight L2 norm over epochs — first and last weight layer", fontsize=11)

    for col_idx, arch in enumerate(archs):
        s = arch_summaries[arch]
        color = ARCH_COLORS[arch]
        layer_norms = _weight_norms_over_epochs(s["weight_history"])
        w_layers = _weight_layers(s)

        for row_idx, layer in enumerate(w_layers[:n_rows]):
            ax = axes[row_idx][col_idx]
            norms = layer_norms.get(layer, {})
            if norms:
                xs = sorted(norms.keys())
                ys = [norms[e] for e in xs]
                ax.plot(xs, ys, color=color, linewidth=1.8)

            row_label = "First weight layer" if row_idx == 0 else "Last weight layer"
            ax.set_title(
                f"{ARCH_LABELS[arch]}\n{layer}",
                fontsize=8,
            )
            ax.set_xlabel("Epoch" if row_idx == n_rows - 1 else "")
            ax.set_ylabel("L2 norm" if col_idx == 0 else "")
            ax.grid(True, alpha=0.3)

        # If fewer than n_rows layers, hide spare axes
        for row_idx in range(len(w_layers), n_rows):
            axes[row_idx][col_idx].set_visible(False)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Activation drift — mean activation magnitude at replay checkpoints
# ---------------------------------------------------------------------------


def _save_activation_drift(arch_summaries: dict, out_path: Path) -> None:
    archs = list(arch_summaries.keys())
    fig, axes = plt.subplots(
        1, len(archs), figsize=(4.5 * len(archs), 4), constrained_layout=True,
    )
    if len(archs) == 1:
        axes = [axes]
    fig.suptitle("Activation drift — mean |activation| at replay checkpoints", fontsize=11)

    for ax, arch in zip(axes, archs):
        s = arch_summaries[arch]
        replay_epochs = s["replay_epochs"]
        act_history   = s["act_history"]
        modules       = s["replay_modules"]

        for mod in modules:
            ys = []
            for act_snap in act_history:
                if mod in act_snap:
                    ys.append(float(np.abs(act_snap[mod]).mean()))
                else:
                    ys.append(float("nan"))
            xs = replay_epochs[: len(ys)]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, label=mod)

        ax.set_title(ARCH_LABELS[arch], fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean |activation|" if arch == archs[0] else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
