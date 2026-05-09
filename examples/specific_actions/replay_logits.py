"""Capture raw model output (logits) at a saved checkpoint.

After training, use ReplaySession with capture=["logits"] to collect the
model's output tensor for every sample in a dataset.  Logits are concatenated
across batches so the result is a single (N, num_classes) tensor.

This is useful for:
  - Comparing model predictions before and after fine-tuning
  - Analysing output distributions at different training checkpoints
  - Computing metrics (accuracy, calibration) from a fixed checkpoint

Run:
    python examples/specific_actions/replay_logits.py
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
    log_dir = Path(__file__).parent.parent / "outputs" / "replay_logits" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    _train_and_save(log_dir)

    torch.manual_seed(1)
    X_eval = torch.randn(16, 4)
    y_eval = torch.randint(0, 2, (16,))
    dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, persistent_workers=True)

    # -------------------------------------------------------------------
    # Capture logits only.
    # modules is required by the API even when only logits are captured;
    # the module list does not affect the logit output.
    # -------------------------------------------------------------------
    session = ReplaySession(
        run=log_dir,
        checkpoint=4,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc2"],         # required; pick the final layer
        capture=["logits"],
    )
    result = session.run()

    logits = result.logits
    print(f"logits shape : {tuple(logits.shape)}")   # (16, 2)
    print(f"logits dtype : {logits.dtype}")

    # Convert raw logits to class probabilities and predictions
    probs = logits.softmax(dim=-1)
    preds = logits.argmax(dim=-1)
    accuracy = (preds == y_eval).float().mean().item()

    print(f"\nfirst 5 predictions : {preds[:5].tolist()}")
    print(f"first 5 labels      : {y_eval[:5].tolist()}")
    print(f"accuracy            : {accuracy:.2%}")

    # -------------------------------------------------------------------
    # Compare logits across two checkpoints.
    # -------------------------------------------------------------------
    session_early = ReplaySession(
        run=log_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=dataloader,
        modules=["fc2"],
        capture=["logits"],
    )
    logits_early = session_early.run().logits

    drift = (logits - logits_early).abs().mean().item()
    print(f"\nmean |logit| change from epoch 0 → 4: {drift:.4f}")


if __name__ == "__main__":
    main()
