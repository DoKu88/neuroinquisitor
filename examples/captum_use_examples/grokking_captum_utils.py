"""Captum attribution utilities for the Grokking Captum example."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerConductance, LayerIntegratedGradients
from tqdm import tqdm

from neuroinquisitor.collection import SnapshotCollection

TOKEN_LABELS = ["a  (pos 0)", "b  (pos 1)", "=  (pos 2)"]
TOKEN_COLORS = ["tab:blue", "tab:orange", "tab:green"]


# ---------------------------------------------------------------------------
# Captum compute helpers
# ---------------------------------------------------------------------------


def compute_lig_per_snapshot(
    snapshots: SnapshotCollection,
    model_factory: Callable[[], nn.Module],
    attr_inputs: torch.Tensor,   # (N, 3) LongTensor, CPU
    attr_targets: torch.Tensor,  # (N,) LongTensor, CPU
    baseline: torch.Tensor,      # (N, 3) LongTensor, CPU
    n_steps: int = 30,
) -> np.ndarray:
    """LayerIntegratedGradients on token_emb across all snapshots.

    Integration point: col.to_state_dict(epoch) → model.load_state_dict()
    No NI types cross the Captum boundary — just a plain nn.Module and tensors.

    Returns (n_snapshots, 3): mean L2-norm of embedding attribution per token position.
    """
    results = []
    for epoch in tqdm(snapshots.epochs, desc="  LIG per snapshot", unit="snap"):
        model = model_factory()
        model.load_state_dict(snapshots.to_state_dict(epoch), strict=False)
        model.eval()

        lig = LayerIntegratedGradients(model, model.token_emb)
        attrs = lig.attribute(
            attr_inputs,
            baselines=baseline,
            target=attr_targets,
            n_steps=n_steps,
            return_convergence_delta=False,
        )
        # attrs: (N, 3, d_model) — attribution in embedding space
        # L2-norm over d_model → (N, 3), then mean over N → (3,)
        token_attr = attrs.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
        results.append(token_attr)

    return np.array(results)  # (n_snapshots, 3)


class _EmbeddingForwardModel(nn.Module):
    """Wrapper that takes float embeddings as input, bypassing nn.Embedding lookup.

    LayerConductance interpolates its input tensor between baseline and actual,
    producing floats. nn.Embedding rejects floats, so we pre-compute embeddings
    and feed this wrapper instead.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._m = model

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (N, seq_len, d_model) — already in embedding space
        pos = torch.arange(embeddings.size(1), device=embeddings.device)
        h = embeddings + self._m.pos_emb(pos)
        h = self._m.transformer(h)
        return self._m.output_proj(h[:, -1])


def compute_conductance_at_checkpoints(
    snapshots: SnapshotCollection,
    model_factory: Callable[[], nn.Module],
    attr_inputs: torch.Tensor,
    attr_targets: torch.Tensor,
    baseline: torch.Tensor,
    checkpoint_indices: list[int],
    n_steps: int = 30,
) -> np.ndarray:
    """LayerConductance on the transformer encoder at selected snapshot indices.

    Measures how much the full transformer encoder block contributes to the
    prediction at each token position, vs. the learned embeddings alone.

    Uses _EmbeddingForwardModel so Captum interpolates in continuous embedding
    space rather than passing floats to nn.Embedding.

    Returns (len(checkpoint_indices), 3).
    """
    epochs = snapshots.epochs
    results = []
    for idx in tqdm(checkpoint_indices, desc="  Conductance at checkpoints", unit="ckpt"):
        model = model_factory()
        model.load_state_dict(snapshots.to_state_dict(epochs[idx]), strict=False)
        model.eval()

        wrapper = _EmbeddingForwardModel(model)
        actual_embeds   = model.token_emb(attr_inputs).detach()   # (N, 3, d_model)
        baseline_embeds = model.token_emb(baseline).detach()      # (N, 3, d_model)

        lc = LayerConductance(wrapper, wrapper._m.transformer)
        cond = lc.attribute(
            actual_embeds,
            baselines=baseline_embeds,
            target=attr_targets,
            n_steps=n_steps,
        )
        # cond: (N, 3, d_model) — conductance at transformer output
        token_cond = cond.norm(dim=-1).mean(dim=0).detach().cpu().numpy()  # (3,)
        results.append(token_cond)

    return np.array(results)  # (n_checkpoints, 3)


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
) -> None:
    print("\n── Visualizations ──")

    p1 = run_dir / "token_attribution_evolution.png"
    _save_attribution_evolution(lig_evolution, test_accs, snapshot_every, p1)
    print(f"  Saved: {p1.name}")

    p2 = run_dir / "attribution_heatmap.png"
    _save_attribution_heatmap(lig_evolution, p2)
    print(f"  Saved: {p2.name}")

    p3 = run_dir / "conductance_at_checkpoints.png"
    _save_conductance_chart(conductance, checkpoint_indices, p3)
    print(f"  Saved: {p3.name}")

    p4 = run_dir / "accuracy_curves.png"
    _save_accuracy_curves(train_accs, test_accs, snapshot_every, p4)
    print(f"  Saved: {p4.name}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_attribution_evolution(
    lig_evolution: np.ndarray,
    test_accs: list[float],
    snapshot_every: int,
    out_path: Path,
) -> None:
    steps = [(i + 1) * snapshot_every for i in range(lig_evolution.shape[0])]

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, (label, color) in enumerate(zip(TOKEN_LABELS, TOKEN_COLORS)):
        ax1.plot(steps, lig_evolution[:, i], label=label, color=color, linewidth=1.5)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean token attribution (L2 norm)")
    ax1.set_title("LayerIntegratedGradients — token attribution evolution\n"
                  "Symmetric a/b attribution signals algorithmic generalisation")
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


def _save_attribution_heatmap(lig_evolution: np.ndarray, out_path: Path) -> None:
    row_max = lig_evolution.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)
    normed = lig_evolution / row_max  # normalise per snapshot for relative comparison

    fig, ax = plt.subplots(figsize=(4, 8), constrained_layout=True)
    im = ax.imshow(normed, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["a", "b", "="])
    ax.set_ylabel("Snapshot index")
    ax.set_title("Token attribution heatmap\n(row-normalised LIG magnitude)")
    fig.colorbar(im, ax=ax, shrink=0.5, label="relative attribution")
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_conductance_chart(
    conductance: np.ndarray,
    checkpoint_indices: list[int],
    out_path: Path,
) -> None:
    n_ckpts = len(checkpoint_indices)
    x = np.arange(n_ckpts)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, (label, color) in enumerate(zip(TOKEN_LABELS, TOKEN_COLORS)):
        ax.bar(x + i * width, conductance[:, i], width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"snap {idx}" for idx in checkpoint_indices])
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel("Mean token conductance (L2 norm)")
    ax.set_title("LayerConductance on transformer encoder — selected checkpoints")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
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
    ax.set_title("Grokking — accuracy dynamics\n"
                 "(train→100% = memorisation; test jump = generalisation)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
