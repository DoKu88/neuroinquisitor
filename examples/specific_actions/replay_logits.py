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
import yaml

from neuroinquisitor import NeuroInquisitor, ReplaySession


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _train_and_save(log_dir: Path, num_epochs: int, lr: float, n_train: int) -> None:
    torch.manual_seed(0)
    X = torch.randn(n_train, 4)
    y = torch.randint(0, 2, (n_train,)).float().unsqueeze(1).expand(-1, 2)

    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    observer = NeuroInquisitor(model, log_dir=log_dir, create_new=True)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)
    observer.close()


def main() -> None:
    cfg_path = Path(__file__).parent.parent / "configs" / "specific_actions_replay_logits.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent.parent / "outputs" / "replay_logits" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    _train_and_save(log_dir, cfg["num_train_epochs"], cfg["train_lr"], cfg["n_train"])

    torch.manual_seed(1)
    X_eval = torch.randn(cfg["n_eval"], 4)
    y_eval = torch.randint(0, 2, (cfg["n_eval"],))
    dataset = torch.utils.data.TensorDataset(X_eval, y_eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=2, persistent_workers=True)

    # -------------------------------------------------------------------
    # Capture logits only.
    # modules is required by the API even when only logits are captured;
    # the module list does not affect the logit output.
    # -------------------------------------------------------------------
    session = ReplaySession(
        run=log_dir,
        checkpoint=cfg["replay_checkpoint"],
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
    print(f"\nmean |logit| change from epoch 0 → {cfg['replay_checkpoint']}: {drift:.4f}")


if __name__ == "__main__":
    main()
