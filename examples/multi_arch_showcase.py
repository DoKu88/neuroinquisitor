"""Multi-Architecture NeuroInquisitor Showcase.

Trains a fully connected network (TinyMLP), a 1-D convolutional network (SmallCNN),
and a single-layer Transformer (TinyTransformer) on the same synthetic sequence
classification task — no dataset downloads required.

All three are instrumented with an identical NeuroInquisitor setup (same CapturePolicy,
same RunMetadata fields, same snapshot() call signature).  The analysis section calls
SnapshotCollection and ReplaySession identically on each architecture, demonstrating
that the NI API is architecture-agnostic.

NI features demonstrated for every architecture:
  • CapturePolicy  — parameters, buffers, optimizer state
  • RunMetadata    — training provenance
  • NeuroInquisitor.snapshot() — per-epoch checkpoints with metadata
  • SnapshotCollection — by_epoch, by_layer, select, to_state_dict, to_numpy
  • ReplaySession  — activations, gradients, and logits via forward/backward hooks
  • Visualization  — side-by-side weight evolution and activation drift per architecture

Task: binary sequence classification on synthetic float data.
  Input   (N, seq_len)  float, values drawn from N(0, 1)
  Label   0 if sequence sum < 0, 1 if sequence sum ≥ 0

Architecture asymmetry:
  TinyTransformer — attention can aggregate any set of positions trivially.
  TinyMLP         — solvable; learns a weighted sum over all positions.
  SmallCNN        — solvable; AdaptiveAvgPool aggregates local conv features.

Run:
    python examples/multi_arch_showcase.py

Requires:
    pip install petname matplotlib
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import petname
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    RunMetadata,
    SnapshotCollection,
)
from multi_arch_showcase_utils import generate_visualizations


# ---------------------------------------------------------------------------
# ─── Shared data ───
# ---------------------------------------------------------------------------


def make_dataset(
    n_samples: int,
    seq_len: int,
    train_frac: float,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_train, y_train, X_test, y_test).

    X: (N, seq_len) float.  y: (N,) long, y=1 if X.sum(dim=1) >= 0.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, seq_len)
    y = (X.sum(dim=1) >= 0).long()
    n_train = int(n_samples * train_frac)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def make_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
    )
    return train_dl, test_dl


# ---------------------------------------------------------------------------
# ─── TinyMLP ───
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Two-layer fully-connected network for sequence classification.

    Input shape: (N, seq_len).  Passes through two linear layers with ReLU.
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(seq_len, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# ─── SmallCNN ───
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """1-D convolutional network for sequence classification.

    Input shape: (N, seq_len).  Treats the sequence as a single-channel signal.
    Conv1d(kernel=3) → ReLU → AdaptiveAvgPool → Linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x.unsqueeze(1)))  # (N, 16, seq_len)
        h = self.pool(h).squeeze(-1)               # (N, 16)
        return self.fc(h)


# ---------------------------------------------------------------------------
# ─── TinyTransformer ───
# ---------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """Single-layer Transformer encoder for sequence classification.

    Input shape: (N, seq_len).  Projects each scalar to d_model, runs one
    TransformerEncoderLayer, then classifies from the mean-pooled output.
    """

    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.encoder    = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1))    # (N, seq_len, d_model)
        h = self.encoder(h)               # (N, seq_len, d_model)
        return self.classifier(h.mean(dim=1))


# ---------------------------------------------------------------------------
# ─── Shared training ───
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    observer: NeuroInquisitor,
    device: torch.device,
    num_epochs: int,
) -> tuple[list[float], list[float]]:
    loss_fn = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    test_losses:  list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = sum(
                loss_fn(model(X.to(device)), y.to(device)).item()
                for X, y in test_loader
            ) / len(test_loader)
        test_losses.append(test_loss)

        print(f"    epoch {epoch:2d}  train={train_loss:.4f}  test={test_loss:.4f}")

        observer.snapshot(
            epoch=epoch,
            metadata={"loss": train_loss, "test_loss": test_loss},
        )

    observer.close()
    return train_losses, test_losses


# ---------------------------------------------------------------------------
# ─── Shared analysis ───
# ---------------------------------------------------------------------------


def analyze(
    snapshots: SnapshotCollection,
    run_dir: Path,
    test_loader: DataLoader,
    model_factory: Callable[[], nn.Module],
    replay_modules: list[str],
    replay_epochs: list[int],
) -> dict:
    """Demonstrate SnapshotCollection and ReplaySession — identical call pattern per arch."""
    first_layer = snapshots.layers[0]

    print(f"  epochs            : {snapshots.epochs}")
    print(f"  layers            : {len(snapshots.layers)} tensors")
    print(f"  by_epoch(0)       : {len(snapshots.by_epoch(0))} tensors")
    print(f"  by_layer          : {len(snapshots.by_layer(first_layer))} epochs")

    sliced = snapshots.select(epochs=range(0, 5))
    print(f"  select(0..4)      : {len(sliced)} snapshots")

    arr = snapshots.to_numpy(epoch=0, layers=[first_layer])
    print(f"  to_numpy shape    : {arr[first_layer].shape}")

    state = snapshots.to_state_dict(epoch=0)
    restored = model_factory()
    restored.load_state_dict(state, strict=False)
    print(f"  to_state_dict     : model restored (strict=False)")

    # Weight norms across all epochs
    weight_history = {epoch: snapshots.by_epoch(epoch) for epoch in snapshots.epochs}

    # Activation / gradient / logit capture at selected checkpoints
    act_history: list[dict[str, np.ndarray]] = []
    for ckpt in replay_epochs:
        result = ReplaySession(
            run=run_dir,
            checkpoint=ckpt,
            model_factory=model_factory,
            dataloader=test_loader,
            modules=replay_modules,
            capture=["activations", "gradients", "logits"],
            activation_reduction="mean",
            gradient_mode="aggregated",
        ).run()
        act_history.append(result.activations.to_numpy())

    # Final-checkpoint activation magnitudes
    final_acts = act_history[-1]
    act_mags = {m: float(np.abs(final_acts[m]).mean()) for m in replay_modules if m in final_acts}

    # Final-epoch weight norms
    final_weights = weight_history[snapshots.epochs[-1]]
    w_norms = {k: float(np.linalg.norm(v)) for k, v in final_weights.items()}

    print(f"  replay epochs     : {replay_epochs}")
    for m, mag in act_mags.items():
        print(f"    {m:16s}  mean |activation| = {mag:.4f}")

    return {
        "weight_history": weight_history,
        "act_history":    act_history,
        "act_mags":       act_mags,
        "w_norms":        w_norms,
        "replay_epochs":  replay_epochs,
        "replay_modules": replay_modules,
    }


# ---------------------------------------------------------------------------
# ─── Entry point ───
# ---------------------------------------------------------------------------


def main() -> None:
    cfg_path = Path(__file__).parent / "configs" / "multi_arch_showcase.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    base_dir = Path(__file__).parent.parent / "outputs" / "multi_arch_showcase" / run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Base dir : {base_dir}/")
    print(f"Device   : {device}\n")

    num_epochs    = cfg["num_epochs"]
    lr            = cfg["lr"]
    seq_len       = cfg["seq_len"]
    n_samples     = cfg["n_samples"]
    train_frac    = cfg["train_frac"]
    batch_size    = cfg["batch_size"]
    replay_epochs = cfg["replay_epochs"]

    X_train, y_train, X_test, y_test = make_dataset(n_samples, seq_len, train_frac)
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size)

    def make_mlp() -> TinyMLP:
        return TinyMLP(seq_len)

    def make_cnn() -> SmallCNN:
        return SmallCNN()

    def make_transformer() -> TinyTransformer:
        return TinyTransformer()

    # Each tuple: (arch_name, model_factory, replay_modules)
    architectures = [
        ("mlp",         make_mlp,         cfg["replay_modules"]["mlp"]),
        ("cnn",         make_cnn,         cfg["replay_modules"]["cnn"]),
        ("transformer", make_transformer, cfg["replay_modules"]["transformer"]),
    ]

    # ─── Training ───────────────────────────────────────────────────────────
    loss_histories: dict[str, tuple[list[float], list[float]]] = {}

    for arch_name, factory, _ in architectures:
        run_dir = base_dir / arch_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model     = factory().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Identical NI setup — only model_class differs between architectures
        policy = CapturePolicy(
            capture_parameters=True,
            capture_buffers=True,
            capture_optimizer=True,
            replay_activations=True,
            replay_gradients=True,
        )
        meta = RunMetadata(
            training_config={"lr": lr, "batch_size": batch_size},
            optimizer_class="Adam",
            device=str(device),
            model_class=type(model).__name__,
        )
        observer = NeuroInquisitor(
            model,
            log_dir=run_dir,
            compress=True,
            create_new=True,
            capture_policy=policy,
            run_metadata=meta,
        )

        print(f"\n── Training {arch_name} ──")
        train_losses, test_losses = train(
            model, optimizer, train_loader, test_loader,
            observer, device, num_epochs,
        )
        loss_histories[arch_name] = (train_losses, test_losses)

    # ─── Analysis ───────────────────────────────────────────────────────────
    arch_summaries: dict[str, dict] = {}

    for arch_name, factory, modules in architectures:
        run_dir   = base_dir / arch_name
        snapshots = NeuroInquisitor.load(run_dir)

        print(f"\n── Analysis {arch_name} ──")
        summary = analyze(
            snapshots, run_dir, test_loader,
            factory, modules, replay_epochs,
        )
        summary["train_losses"] = loss_histories[arch_name][0]
        summary["test_losses"]  = loss_histories[arch_name][1]
        arch_summaries[arch_name] = summary

    # ─── Summary table ──────────────────────────────────────────────────────
    print("\n── Summary table ──")
    print(f"  {'Architecture':14}  {'Mean weight norm':20}  {'Mean act magnitude':20}")
    print(f"  {'-' * 14}  {'-' * 20}  {'-' * 20}")
    for arch_name, s in arch_summaries.items():
        w_norm  = float(np.mean(list(s["w_norms"].values())))
        act_mag = float(np.mean(list(s["act_mags"].values()))) if s["act_mags"] else float("nan")
        print(f"  {arch_name:14}  {w_norm:20.4f}  {act_mag:20.4f}")

    generate_visualizations(arch_summaries, base_dir)
    print(f"\nAll outputs in: {base_dir}/")


if __name__ == "__main__":
    main()
