"""Capture output gradients at a saved checkpoint.

After training, use ReplaySession to replay a checkpoint on a dataset and
record the gradient of the model's summed output with respect to each
module's output tensor.  This is useful for sensitivity analysis and
understanding which layers respond most strongly to the input.

gradient_mode controls output shape per module:
  - "per_example"  → full (N, ...) tensor — one gradient vector per sample
  - "aggregated"   → mean over the batch dim → (...) — dataset-level signal

Run:
    python examples/specific_actions/track_gradients.py
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
    log_dir = Path(__file__).parent.parent / "outputs" / "track_gradients" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    _train_and_save(log_dir)

    torch.manual_seed(1)
    X_eval = torch.randn(16, 4)
    y_eval = torch.randint(0, 2, (16,))
    dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, persistent_workers=True)

    # -------------------------------------------------------------------
    # per_example mode — one gradient vector per sample.
    # Shape matches the module's output: (N, output_dim).
    # Lets you compare gradient magnitude across individual inputs.
    # -------------------------------------------------------------------
    session_per = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc1", "relu", "fc2"],
        capture=["gradients"],
        gradient_mode="per_example",
    )
    result_per = session_per.run()

    print("gradient_mode='per_example' — shape per module:")
    for name, tensor in result_per.gradients.items():
        print(f"  {name:6s}  shape={tuple(tensor.shape)}")
    print()

    # -------------------------------------------------------------------
    # aggregated mode — mean gradient across the dataset.
    # Collapses the sample dimension; useful for layer-level summaries.
    # -------------------------------------------------------------------
    session_agg = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc1", "relu", "fc2"],
        capture=["gradients"],
        gradient_mode="aggregated",
    )
    result_agg = session_agg.run()

    print("gradient_mode='aggregated' — shape per module:")
    for name, tensor in result_agg.gradients.items():
        print(f"  {name:6s}  shape={tuple(tensor.shape)}  mean_abs={tensor.abs().mean():.4f}")
    print()

    # -------------------------------------------------------------------
    # Capture both activations and gradients in one pass.
    # -------------------------------------------------------------------
    session_both = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc1", "fc2"],
        capture=["activations", "gradients"],
    )
    result_both = session_both.run()

    print("combined capture — activations and gradients:")
    for name in result_both.activations:
        act_shape = tuple(result_both.activations[name].shape)
        grad_shape = tuple(result_both.gradients[name].shape)
        print(f"  {name:6s}  act={act_shape}  grad={grad_shape}")


if __name__ == "__main__":
    main()
