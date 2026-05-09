"""Grokking: modular addition — full NeuroInquisitor feature showcase.

Demonstrates every implemented capability:
  • CapturePolicy  — capture parameters, buffers, and optimizer state
  • RunMetadata    — attach training provenance to the run
  • Snapshots      — weight + buffer checkpoints with per-step metadata
  • SnapshotCollection — by_epoch, by_layer, select, to_state_dict, to_numpy
  • ReplaySession  — activations, gradients, and logits via forward/backward hooks
  • Visualization  — weight-evolution video, replay figure, and accuracy curves

"Grokking" (Power et al. 2022) is a two-phase training phenomenon:
  Phase 1 — memorisation: train loss → 0, test loss stays high.
  Phase 2 — generalisation: after much more training, test loss suddenly collapses too.

The weight transition between phases is dramatic. The token embeddings develop
clean Fourier structure in phase 2 that is completely absent in phase 1.

Task: learn (a + b) mod p for a prime p, treating it as classification over p classes.
      All p² input pairs form the dataset; ~50% is held out as the test split.

Run:
    python examples/grokking_example.py

Requires:
    pip install tqdm petname matplotlib
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    RunMetadata,
    SnapshotCollection,
)
from grokking_example_utils import generate_visualizations


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P = 97
D_MODEL = 128
N_HEADS = 4
FFN_DIM = 512
TRAIN_FRAC = 0.5
NUM_STEPS = 15_000
SNAPSHOT_EVERY = 150

EQ_TOKEN = P

COMPONENT_KEYS = [
    ("token_emb.weight",                               "token_emb"),
    ("transformer.layers.0.self_attn.in_proj_weight",  "attn_in_proj"),
    ("transformer.layers.0.linear1.weight",            "ffn_linear1"),
    ("output_proj.weight",                             "output_proj"),
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def make_dataset(
    p: int, train_frac: float, seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_inputs, train_labels, test_inputs, test_labels) as LongTensors.

    Each input row is [a, b, EQ_TOKEN]; label is (a + b) % p.
    """
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
    return DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)


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
) -> tuple[list[float], list[float], list[float], list[float]]:
    train_acc_history:  list[float] = []
    test_acc_history:   list[float] = []
    train_loss_history: list[float] = []
    test_loss_history:  list[float] = []
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

                train_loss_history.append(tr_loss)
                test_loss_history.append(te_loss)
                train_acc_history.append(tr_acc)
                test_acc_history.append(te_acc)

                pbar.set_postfix(tr_acc=f"{tr_acc:.1%}", te_acc=f"{te_acc:.1%}", te_loss=f"{te_loss:.3f}")

                observer.snapshot(
                    epoch=snapshot_idx,
                    step=step,
                    metadata={
                        "train_step": step,
                        "loss": tr_loss,
                        "test_loss": te_loss,
                        "accuracy": tr_acc,
                        "test_accuracy": te_acc,
                    },
                )
                snapshot_idx += 1

    observer.close()
    print(f"\nTraining done. {NUM_STEPS // snapshot_every} snapshots saved.")
    return train_acc_history, test_acc_history, train_loss_history, test_loss_history


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    snapshots: SnapshotCollection,
    run_dir: Path,
    test_loader: DataLoader,
    replay_modules: list[str],
    num_snapshots: int,
    device: torch.device,
) -> list[dict[str, dict[str, np.ndarray]]]:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")
    print(f"  by_epoch(0) keys : {list(snapshots.by_epoch(0).keys())[:4]} …")
    print(f"  by_layer('token_emb.weight') epochs : {list(snapshots.by_layer('token_emb.weight').keys())}")

    arrays = snapshots.to_numpy(epoch=0, layers=["token_emb.weight"])
    print(f"  to_numpy(epoch=0, layers=['token_emb.weight']) shape : {arrays['token_emb.weight'].shape}")

    restored = model_factory().to(device)
    restored.load_state_dict(snapshots.to_state_dict(epoch=0), strict=False)
    print(f"  to_state_dict(epoch=0) → model restored (strict=False)")

    print("\n── ReplaySession ──")
    replay_history: list[dict[str, dict[str, np.ndarray]]] = []
    final_replay = None

    for snapshot_idx in tqdm(range(num_snapshots), desc="  Replaying checkpoints", unit="ckpt", leave=True):
        final_replay = ReplaySession(
            run=run_dir,
            checkpoint=snapshot_idx,
            model_factory=model_factory,
            dataloader=test_loader,
            modules=replay_modules,
            capture=["activations", "gradients", "logits"],
            activation_reduction="pool",
            gradient_mode="aggregated",
            dataset_slice=lambda samples: samples[:256],
            slice_metadata={"description": "first 256 test samples"},
        ).run()
        replay_history.append({
            "activations": final_replay.activations.to_numpy(),
            "gradients":   final_replay.gradients.to_numpy(),
        })

    print(f"  Samples replayed : {final_replay.metadata.n_samples}")
    print(f"  Logits shape     : {final_replay.logits.shape}")
    for name in replay_modules:
        print(f"  {name:12s} — activations {tuple(final_replay.activations[name].shape)}"
              f"  gradients {tuple(final_replay.gradients[name].shape)}")

    return replay_history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent / "outputs" / "grokking_example" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}")
    print(f"Task     : (a + b) mod {P},  train frac={TRAIN_FRAC},  steps={NUM_STEPS}")
    print()

    num_snapshots  = NUM_STEPS // SNAPSHOT_EVERY
    replay_modules = ["output_proj"]

    train_x, train_y, test_x, test_y = make_dataset(P, TRAIN_FRAC)
    test_loader = make_test_loader(test_x, test_y)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x,  test_y  = test_x.to(device),  test_y.to(device)

    model     = model_factory().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    loss_fn   = nn.CrossEntropyLoss()

    policy = CapturePolicy(
        capture_parameters=True,
        capture_buffers=True,
        capture_optimizer=True,
        replay_activations=True,
        replay_gradients=True,
    )
    run_meta = RunMetadata(
        training_config={
            "lr": 1e-3, "weight_decay": 1.0,
            "num_steps": NUM_STEPS, "snapshot_every": SNAPSHOT_EVERY,
        },
        optimizer_class="AdamW",
        device=str(device),
        model_class="GrokkingTransformer",
    )
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=policy,
        run_metadata=run_meta,
    )

    train_accs, test_accs, train_losses, test_losses = train(
        model, optimizer, loss_fn,
        train_x, train_y, test_x, test_y,
        observer, NUM_STEPS, SNAPSHOT_EVERY,
    )

    snapshots      = NeuroInquisitor.load(run_dir)
    replay_history = analyze(snapshots, run_dir, test_loader, replay_modules, num_snapshots, device)
    weight_history = [snapshots.by_epoch(e) for e in range(num_snapshots)]

    generate_visualizations(
        weight_history, replay_history, replay_modules,
        train_accs, test_accs, SNAPSHOT_EVERY, COMPONENT_KEYS, run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
