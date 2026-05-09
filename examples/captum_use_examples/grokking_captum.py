"""Grokking (modular addition) + Captum integration with NeuroInquisitor.

Demonstrates Captum attribution on a transformer trained on the grokking task
(Power et al. 2022 — learn (a + b) mod p, two-phase memorisation → generalisation).

Integration paths shown:
  Path A  col.to_state_dict(epoch) → model.load_state_dict() → Captum
  Path B  ReplaySession.run() → result.activations (plain tensors) → Captum

Attribution methods:
  1. LayerIntegratedGradients on token_emb — which input position (a, b, or =)
     drives predictions across all training snapshots?
  2. LayerConductance on the transformer encoder — how much does the full
     transformer block contribute per token position at selected checkpoints?

Key observation: during memorisation, token attributions may be asymmetric.
During generalisation, attribution to positions 0 (a) and 1 (b) becomes
symmetric — matching the algebraic symmetry of addition.

Outputs (in outputs/Grokking_Captum/<run-name>/):
  token_attribution_evolution.png  — mean |LIG| per position across snapshots
  attribution_heatmap.png          — full (snapshots × position) attribution map
  conductance_at_checkpoints.png   — LayerConductance per position at checkpoints
  accuracy_curves.png              — train/test accuracy over training
  replay_activations.txt           — activation shapes from ReplaySession

Run:
    python examples/captum_use_examples/grokking_captum.py

Requires:
    pip install tqdm petname matplotlib captum

Note: full grokking generalisation typically requires ~15 000+ steps with
weight_decay=1.0. The default config uses 5 000 steps for a faster demo;
increase num_steps in the config to observe the complete phase transition.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import petname
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    SnapshotCollection,
)

from grokking_captum_utils import (
    compute_conductance_at_checkpoints,
    compute_lig_per_snapshot,
    generate_grokking_captum_visualizations,
)

_cfg = yaml.safe_load(
    (Path(__file__).parent.parent / "configs" / "captum_use_examples_grokking_captum.yaml").read_text()
)

P             = _cfg["p"]
D_MODEL       = _cfg["d_model"]
N_HEADS       = _cfg["n_heads"]
FFN_DIM       = _cfg["ffn_dim"]
EQ_TOKEN      = P
NUM_STEPS     = _cfg["num_steps"]
SNAPSHOT_EVERY = _cfg["snapshot_every"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GrokkingTransformer(nn.Module):
    """Minimal 1-layer transformer for modular addition.

    Input:  [a, b, =]  (sequence length 3, tokens in {0..P})
    Output: logits over P classes, read from the '=' position.
    """

    def __init__(self, p: int, d_model: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(p + 1, d_model)
        self.pos_emb   = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.transformer(h)
        return self.output_proj(h[:, -1])


def model_factory() -> GrokkingTransformer:
    return GrokkingTransformer(P, D_MODEL, N_HEADS, FFN_DIM)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def make_dataset(
    p: int, train_frac: float, seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    split = int(len(pairs) * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]

    def build(indices):
        rows = [pairs[i] for i in indices]
        x = torch.tensor([[a, b, EQ_TOKEN] for a, b in rows], dtype=torch.long)
        y = torch.tensor([(a + b) % p for a, b in rows], dtype=torch.long)
        return x, y

    return (*build(train_idx), *build(test_idx))


def make_test_loader(test_x: torch.Tensor, test_y: torch.Tensor, batch_size: int = 512) -> DataLoader:
    return DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size,
        shuffle=False, num_workers=2, persistent_workers=True,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    observer: NeuroInquisitor,
    num_steps: int,
    snapshot_every: int,
) -> tuple[list[float], list[float]]:
    train_acc_history: list[float] = []
    test_acc_history:  list[float] = []
    snapshot_idx = 0

    with tqdm(range(1, num_steps + 1), desc="Training", unit="step") as pbar:
        for step in pbar:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(train_x), train_y)
            loss.backward()
            optimizer.step()

            if step % snapshot_every == 0:
                model.eval()
                with torch.no_grad():
                    train_logits = model(train_x)
                    test_logits  = model(test_x)
                    tr_loss = loss_fn(train_logits, train_y).item()
                    te_loss = loss_fn(test_logits,  test_y).item()
                    tr_acc  = (train_logits.argmax(1) == train_y).float().mean().item()
                    te_acc  = (test_logits.argmax(1)  == test_y).float().mean().item()

                train_acc_history.append(tr_acc)
                test_acc_history.append(te_acc)
                pbar.set_postfix(tr_acc=f"{tr_acc:.1%}", te_acc=f"{te_acc:.1%}")

                observer.snapshot(
                    epoch=snapshot_idx,
                    step=step,
                    metadata={
                        "train_step": step, "loss": tr_loss, "test_loss": te_loss,
                        "accuracy": tr_acc, "test_accuracy": te_acc,
                    },
                )
                snapshot_idx += 1

    observer.close()
    print(f"\nTraining done. {num_steps // snapshot_every} snapshots saved.")
    return train_acc_history, test_acc_history


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    snapshots: SnapshotCollection,
    run_dir: Path,
    attr_inputs: torch.Tensor,
    attr_targets: torch.Tensor,
    baseline: torch.Tensor,
    test_loader: DataLoader,
    n_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")

    print("\n── Path A  col.to_state_dict() → LayerIntegratedGradients ──")
    print("  Tracking token attribution across all snapshots …")
    lig_evolution = compute_lig_per_snapshot(
        snapshots, model_factory, attr_inputs, attr_targets, baseline,
        n_steps=_cfg["ig_n_steps"],
    )
    print(f"  Attribution shape  : {lig_evolution.shape}  (snapshots × token_positions)")
    a, b, eq = lig_evolution[-1]
    print(f"  Final snapshot  →  a={a:.4f}  b={b:.4f}  ={eq:.4f}")
    sym = abs(a - b) / (max(a, b) + 1e-8)
    print(f"  |a−b| / max(a,b) = {sym:.3f}  {'(symmetric ✓)' if sym < 0.1 else '(asymmetric — more training may show grokking)'}")

    print("\n── Path A  col.to_state_dict() → LayerConductance ──")
    n_ckpts = _cfg["n_conductance_checkpoints"]
    ckpt_indices = [int(round(i * (n_snapshots - 1) / (n_ckpts - 1))) for i in range(n_ckpts)]
    conductance = compute_conductance_at_checkpoints(
        snapshots, model_factory, attr_inputs, attr_targets, baseline,
        ckpt_indices, n_steps=_cfg["ig_n_steps"],
    )
    print(f"  Conductance shape : {conductance.shape}  (checkpoints × token_positions)")

    print("\n── Path B  ReplaySession → plain tensors → Captum ──")
    final_epoch = n_snapshots - 1
    result = ReplaySession(
        run=run_dir,
        checkpoint=final_epoch,
        model_factory=model_factory,
        dataloader=test_loader,
        modules=_cfg["replay_modules"],
        capture=["activations"],
    ).run()
    print(f"  Samples replayed : {result.metadata.n_samples}")
    for name, tensor in result.activations.items():
        assert type(tensor) is torch.Tensor, (
            f"Activation {name!r} is {type(tensor).__name__}; expected torch.Tensor"
        )
        print(f"  {name:16s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print("  All activation values are plain torch.Tensor — Captum accepts them directly.")

    replay_path = run_dir / "replay_activations.txt"
    lines = [
        f"ReplaySession — epoch {final_epoch} activations",
        f"  n_samples : {result.metadata.n_samples}",
        "",
        *(
            f"  {name:16s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}"
            for name, tensor in result.activations.items()
        ),
        "",
        "All activation values are plain torch.Tensor — Captum accepts them directly.",
        "Example: feed transformer output activations into LayerConductance as additional evidence.",
    ]
    replay_path.write_text("\n".join(lines))
    print(f"  Replay log saved : {replay_path.name}")

    return lig_evolution, conductance, ckpt_indices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    # Captum gradient-based methods are most reliable on CPU.
    # Training uses the fastest available device; attribution always runs on CPU.
    train_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent.parent / "outputs" / "Grokking_Captum" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name      : {run_name}")
    print(f"Run dir       : {run_dir}/")
    print(f"Train device  : {train_device}")
    print(f"Captum device : cpu")
    print(f"Task          : (a + b) mod {P},  train_frac={_cfg['train_frac']},  steps={NUM_STEPS}\n")

    n_snapshots = NUM_STEPS // SNAPSHOT_EVERY

    train_x, train_y, test_x, test_y = make_dataset(P, _cfg["train_frac"])
    test_loader = make_test_loader(test_x, test_y)
    train_x_dev, train_y_dev = train_x.to(train_device), train_y.to(train_device)
    test_x_dev,  test_y_dev  = test_x.to(train_device),  test_y.to(train_device)

    # Attribution samples stay on CPU (Captum requirement for gradient-based methods).
    # Baseline: [0, 0, EQ_TOKEN] — null operands, structural token preserved.
    attr_n       = _cfg["attr_samples"]
    attr_inputs  = test_x[:attr_n]   # (N, 3) LongTensor, CPU
    attr_targets = test_y[:attr_n]   # (N,) LongTensor, CPU
    baseline     = attr_inputs.clone()
    baseline[:, 0] = 0
    baseline[:, 1] = 0  # zero out operands; keep EQ_TOKEN at position 2

    model     = model_factory().to(train_device)
    optimizer = optim.AdamW(model.parameters(), lr=_cfg["lr"], weight_decay=_cfg["weight_decay"])
    loss_fn   = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=CapturePolicy(capture_parameters=True),
    )

    train_accs, test_accs = train(
        model, optimizer, loss_fn,
        train_x_dev, train_y_dev, test_x_dev, test_y_dev,
        observer, NUM_STEPS, SNAPSHOT_EVERY,
    )

    snapshots = NeuroInquisitor.load(run_dir)
    lig_evolution, conductance, ckpt_indices = analyze(
        snapshots, run_dir, attr_inputs, attr_targets, baseline,
        test_loader, n_snapshots,
    )

    generate_grokking_captum_visualizations(
        lig_evolution, conductance, ckpt_indices,
        train_accs, test_accs, SNAPSHOT_EVERY, run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
