"""Grokking: modular addition with a small transformer, tracked via NeuroInquisitor.

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

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from neuroinquisitor import NeuroInquisitor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P = 97              # prime modulus
D_MODEL = 128       # transformer embedding dimension
N_HEADS = 4
FFN_DIM = 512
TRAIN_FRAC = 0.5
NUM_STEPS = 15_000  # full-batch gradient steps
SNAPSHOT_EVERY = 150  # → 100 snapshots total

EQ_TOKEN = P        # index for the '=' token; vocabulary size = P + 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(p: int, train_frac: float, seed: int = 42):
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


# ---------------------------------------------------------------------------
# Model — 1-layer transformer
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
        return self.output_proj(h[:, -1])  # read from '=' position


# ---------------------------------------------------------------------------
# Layers to show in the video
# ---------------------------------------------------------------------------

LAYERS_TO_PLOT = [
    ("token_emb.weight",                                     "Token embeddings  (98×128)"),
    ("transformer.layers.0.self_attn.in_proj_weight",        "Attn in-proj      (384×128)"),
    ("transformer.layers.0.linear1.weight",                  "FFN linear1        (512×128)"),
    ("output_proj.weight",                                   "Output proj        (97×128)"),
]


# ---------------------------------------------------------------------------
# Video + loss curve
# ---------------------------------------------------------------------------

def _make_video(
    weight_history: list[dict[str, np.ndarray]],
    train_acc_history: list[float],
    test_acc_history: list[float],
    snapshot_every: int,
    out_path: Path,
    fps: int = 5,
) -> Path:
    n_frames = len(weight_history)
    n_panels = len(LAYERS_TO_PLOT)

    rendered = [[snap[k] for k, _ in LAYERS_TO_PLOT] for snap in weight_history]

    vlims: list[tuple[float, float]] = []
    for i in range(n_panels):
        flat = np.concatenate([f[i].ravel() for f in rendered])
        abs_max = float(np.abs(flat).max()) or 1.0
        vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), constrained_layout=True)

    images = []
    for ax, (_, label), arr, (vmin, vmax) in zip(axes, LAYERS_TO_PLOT, rendered[0], vlims):
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(label, fontsize=8)
        images.append(im)

    title_text = fig.suptitle("", fontsize=11)

    def update(frame: int) -> list:
        step = frame * snapshot_every
        tr = train_acc_history[frame] if frame < len(train_acc_history) else 0.0
        te = test_acc_history[frame]  if frame < len(test_acc_history)  else 0.0
        title_text.set_text(f"Step {step}  |  train acc={tr:.1%}  test acc={te:.1%}")
        for im, arr in zip(images, rendered[frame]):
            im.set_data(arr)
        return [*images, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    try:
        ani.save(str(out_path), writer=animation.FFMpegWriter(fps=fps))
        result = out_path
    except Exception:
        result = out_path.with_suffix(".gif")
        ani.save(str(result), writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return result


def _save_loss_curves(
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

    # Log-scale x-axis emphasises the sudden generalisation jump
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = petname.generate(words=2, separator="-")
    run_dir = Path(__file__).parent.parent / "outputs" / "grokking_example" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}")
    print(f"Task     : (a + b) mod {P},  train frac={TRAIN_FRAC},  steps={NUM_STEPS}")
    print()

    train_x, train_y, test_x, test_y = make_dataset(P, TRAIN_FRAC)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x,  test_y  = test_x.to(device),  test_y.to(device)

    model = GrokkingTransformer(P, D_MODEL, N_HEADS, FFN_DIM).to(device)
    # High weight decay is the key ingredient that drives grokking
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    loss_fn = nn.CrossEntropyLoss()

    num_snapshots = NUM_STEPS // SNAPSHOT_EVERY
    observer = NeuroInquisitor(model, log_dir=run_dir, compress=True, create_new=True)

    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    snapshot_idx = 0

    with tqdm(range(1, NUM_STEPS + 1), desc="Training", unit="step") as pbar:
        for step in pbar:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(train_x), train_y)
            loss.backward()
            optimizer.step()

            if step % SNAPSHOT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    train_logits = model(train_x)
                    test_logits  = model(test_x)
                    tr_loss = loss_fn(train_logits, train_y).item()
                    te_loss = loss_fn(test_logits,  test_y).item()
                    tr_acc = (train_logits.argmax(1) == train_y).float().mean().item()
                    te_acc = (test_logits.argmax(1)  == test_y).float().mean().item()

                train_loss_history.append(tr_loss)
                test_loss_history.append(te_loss)
                train_acc_history.append(tr_acc)
                test_acc_history.append(te_acc)

                pbar.set_postfix(
                    tr_acc=f"{tr_acc:.1%}",
                    te_acc=f"{te_acc:.1%}",
                    te_loss=f"{te_loss:.3f}",
                )

                observer.snapshot(
                    epoch=snapshot_idx,
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
    print(f"\nTraining done. {num_snapshots} snapshots saved.")

    col = NeuroInquisitor.load(run_dir)
    weight_history = [col.by_epoch(e) for e in range(num_snapshots)]

    video_path = run_dir / "weights_over_time.mp4"
    print(f"\nGenerating video → {video_path} ...")
    result = _make_video(
        weight_history, train_acc_history, test_acc_history,
        SNAPSHOT_EVERY, video_path, fps=5,
    )
    print(f"Video saved: {result}")

    curves_path = run_dir / "accuracy_curves.png"
    _save_loss_curves(train_acc_history, test_acc_history, SNAPSHOT_EVERY, curves_path)
    print(f"Accuracy curves: {curves_path}")
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
