"""Capture intermediate activations at a saved checkpoint.

After training, use ReplaySession to replay a checkpoint on a dataset and
record each module's output (its activation) for every sample.

activation_reduction controls output shape per module:
  - "raw"  → full (N, ...) tensor — one row per sample
  - "mean" → mean over the batch dim → (...) — a single average activation
  - "pool" → spatial avg-pool for conv/3-D+ layers → (N, C); identity for linear

Run:
    python examples/specific_actions/track_activations.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor, ReplaySession


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _train_and_save(log_dir: Path) -> None:
    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y = torch.randint(0, 2, (64,)).float().unsqueeze(1).expand(-1, 2)

    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    observer = NeuroInquisitor(model, log_dir=log_dir, create_new=True)
    for epoch in range(5):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)
    observer.close()


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "outputs" / "track_activations" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    _train_and_save(log_dir)

    # Build a DataLoader for the replay dataset (can be a held-out eval set).
    torch.manual_seed(1)
    X_eval = torch.randn(16, 4)
    y_eval = torch.randint(0, 2, (16,))
    dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True)

    # -------------------------------------------------------------------
    # Capture activations at epoch 4 — raw mode (default)
    # Each module produces a (N, output_dim) tensor, one row per sample.
    # -------------------------------------------------------------------
    session = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc1", "relu", "fc2"],
        capture=["activations"],
        activation_reduction="raw",
    )
    result = session.run()

    print("activation_reduction='raw' — shape per module:")
    for name, tensor in result.activations.items():
        print(f"  {name:6s}  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print()

    # -------------------------------------------------------------------
    # Same replay, mean reduction — collapses the sample dimension.
    # Useful when you want a single representative activation vector.
    # -------------------------------------------------------------------
    session_mean = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc1", "relu", "fc2"],
        capture=["activations"],
        activation_reduction="mean",
    )
    result_mean = session_mean.run()

    print("activation_reduction='mean' — shape per module:")
    for name, tensor in result_mean.activations.items():
        print(f"  {name:6s}  shape={tuple(tensor.shape)}")
    print()

    # -------------------------------------------------------------------
    # Convert to NumPy for downstream analysis (e.g. scikit-learn, pandas)
    # -------------------------------------------------------------------
    act_np = result.activations.to_numpy()
    print("as NumPy arrays:")
    for name, arr in act_np.items():
        print(f"  {name:6s}  shape={arr.shape}  mean={arr.mean():.4f}")

    print(f"\nn_samples : {result.metadata.n_samples}")
    print(f"checkpoint: epoch {result.metadata.checkpoint_epoch}")


if __name__ == "__main__":
    main()
