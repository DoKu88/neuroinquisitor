"""Grokking: modular addition with a small transformer, tracked via NeuroInquisitor.

"Grokking" (Power et al. 2022) is a two-phase training phenomenon:
  Phase 1 — memorisation: train loss → 0, test loss stays high.
  Phase 2 — generalisation: after much more training, test loss suddenly collapses too.

The weight transition between phases is dramatic. The token embeddings develop
clean Fourier structure in phase 2 that is completely absent in phase 1.

Task: learn (a + b) mod p for a prime p, treating it as classification over p classes.
      All p² input pairs form the dataset; ~50% is held out as the test split.

The video shows four panels that capture the grokking dynamics:
  1. Token embedding cosine similarity (P×P) — noisy during memorisation, develops
     block modular structure when the model generalises.
  2. Embedding Fourier power spectrum over training (freq × snapshots, growing) —
     specific frequency modes spike exactly when grokking occurs.
  3. Component Frobenius norms over training (4 components × snapshots, growing) —
     weight decay drives grokking; watch norms compress then re-grow.
  4. Output projection (97×128, fixed bounds) — structure emerges in phase 2.

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

# Components tracked for norm timeline
COMPONENT_KEYS = [
    ("token_emb.weight",                               "token_emb"),
    ("transformer.layers.0.self_attn.in_proj_weight",  "attn_in_proj"),
    ("transformer.layers.0.linear1.weight",            "ffn_linear1"),
    ("output_proj.weight",                             "output_proj"),
]


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
# Weight rendering helpers
# ---------------------------------------------------------------------------

def _token_cosine_sim(emb_weight: np.ndarray) -> np.ndarray:
    """(P×P) cosine similarity between the P token embeddings (drops EQ_TOKEN row)."""
    emb = emb_weight[:P]  # (P, D)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    normed = emb / norms
    return normed @ normed.T  # (P, P)


def _emb_fourier_power(emb_weight: np.ndarray) -> np.ndarray:
    """Mean Fourier power over embedding dims, indexed by frequency over token axis.

    Returns shape (P//2,) — DC component dropped so the colourmap isn't dominated
    by the mean offset.
    """
    emb = emb_weight[:P]           # (P, D)
    fft = np.fft.rfft(emb, axis=0) # (P//2 + 1, D) complex
    power = (np.abs(fft) ** 2).mean(axis=1)  # (P//2 + 1,)
    return power[1:]  # drop DC; (P//2,)


def _build_fourier_timeline(
    weight_history: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """(P//2, num_snapshots) Fourier power matrix — grows left to right in video."""
    spectra = np.array([
        _emb_fourier_power(snap["token_emb.weight"])
        for snap in weight_history
    ])  # (num_snapshots, P//2)
    return spectra.T  # (P//2, num_snapshots)


def _build_norm_timeline(
    weight_history: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """(num_components, num_snapshots) Frobenius norm matrix — grows left to right."""
    norms = np.array([
        [np.linalg.norm(snap[k]) for k, _ in COMPONENT_KEYS]
        for snap in weight_history
    ])  # (num_snapshots, num_components)
    return norms.T  # (num_components, num_snapshots)


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

    # --- Pre-compute all data ---
    cosine_sims = [_token_cosine_sim(snap["token_emb.weight"]) for snap in weight_history]

    fourier_timeline = _build_fourier_timeline(weight_history)  # (P//2, num_snapshots)
    fourier_vmax = float(fourier_timeline.max()) or 1.0

    norm_timeline = _build_norm_timeline(weight_history)  # (4, num_snapshots)
    norm_vmax = float(norm_timeline.max()) or 1.0

    output_projs = [snap["output_proj.weight"] for snap in weight_history]
    out_abs_max = float(np.abs(np.stack(output_projs)).max()) or 1.0

    # --- Build figure ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    # Panel 0: token embedding cosine similarity
    im_cos = axes[0].imshow(
        cosine_sims[0], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
    )
    fig.colorbar(im_cos, ax=axes[0], shrink=0.8, label="cosine sim")
    axes[0].set_title(f"Token emb cosine similarity ({P}×{P})", fontsize=8)
    axes[0].set_xlabel("Token b")
    axes[0].set_ylabel("Token a")

    # Panel 1: Fourier power spectrum over training (growing)
    freq_bins = fourier_timeline.shape[0]
    fourier_init = np.full_like(fourier_timeline, np.nan)
    fourier_init[:, 0] = fourier_timeline[:, 0]
    im_fourier = axes[1].imshow(
        fourier_init, aspect="auto", cmap="plasma",
        vmin=0, vmax=fourier_vmax, interpolation="nearest",
    )
    fig.colorbar(im_fourier, ax=axes[1], shrink=0.8, label="mean |FFT|²")
    axes[1].set_title("Embedding Fourier power over training", fontsize=8)
    axes[1].set_xlabel("Snapshot")
    axes[1].set_ylabel("Frequency (token axis, DC dropped)")

    # Panel 2: Component Frobenius norm timeline (growing)
    norm_init = np.full_like(norm_timeline, np.nan)
    norm_init[:, 0] = norm_timeline[:, 0]
    im_norm = axes[2].imshow(
        norm_init, aspect="auto", cmap="viridis",
        vmin=0, vmax=norm_vmax, interpolation="nearest",
    )
    fig.colorbar(im_norm, ax=axes[2], shrink=0.8, label="‖W‖_F")
    axes[2].set_title("Component Frobenius norms over training", fontsize=8)
    axes[2].set_xlabel("Snapshot")
    axes[2].set_yticks(range(len(COMPONENT_KEYS)))
    axes[2].set_yticklabels([label for _, label in COMPONENT_KEYS], fontsize=7)

    # Panel 3: Output projection (fixed bounds across all frames)
    im_out = axes[3].imshow(
        output_projs[0], aspect="auto", cmap="RdBu_r",
        vmin=-out_abs_max, vmax=out_abs_max, interpolation="nearest",
    )
    fig.colorbar(im_out, ax=axes[3], shrink=0.8, label="weight value")
    axes[3].set_title("Output proj (97×128) — fixed bounds", fontsize=8)

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
        return [im_cos, im_fourier, im_norm, im_out, title_text]

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
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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
                    tr_loss_t = loss_fn(train_logits, train_y)
                    te_loss_t = loss_fn(test_logits,  test_y)
                    tr_acc_t  = (train_logits.argmax(1) == train_y).float().mean()
                    te_acc_t  = (test_logits.argmax(1)  == test_y).float().mean()
                    # single sync point: batch all .item() calls together
                    tr_loss, te_loss, tr_acc, te_acc = (
                        tr_loss_t.item(), te_loss_t.item(), tr_acc_t.item(), te_acc_t.item()
                    )

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
