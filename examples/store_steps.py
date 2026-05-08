"""Store weights: step-based snapshotting (intra-epoch granularity).

Useful when a single epoch spans thousands of batches and you want
finer-grained weight snapshots.  Snapshots can be keyed by step only,
epoch only, or both.

Run:
    python examples/store_steps.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x).relu())


def main() -> None:
    torch.manual_seed(0)
    dataset = torch.randn(128, 4)
    labels = (dataset.sum(1, keepdim=True) > 0).float()

    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = 32
    batches = [
        (dataset[i : i + batch_size], labels[i : i + batch_size])
        for i in range(0, len(dataset), batch_size)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "outputs" / "store_steps" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    observer = NeuroInquisitor(model, log_dir=log_dir)

    global_step = 0
    for epoch in range(2):
        for X, y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

            # --- snapshot by step only ---
            observer.snapshot(step=global_step)

            # --- or tie epoch + step together for richer indexing ---
            # observer.snapshot(epoch=epoch, step=global_step)

            print(f"epoch {epoch}  step {global_step:4d}  loss={loss.item():.4f}")
            global_step += 1

    observer.close()

    col = NeuroInquisitor.load(log_dir)
    print(f"\ntotal snapshots: {len(col)}")
    # step-only snapshots have no epoch, so col.epochs is empty
    print(f"epochs in index: {col.epochs}")
    # access a specific snapshot via the raw index
    entry = col._index.all()[3]
    print(f"snapshot #3 — step={entry.step}  layers={entry.layers}")


if __name__ == "__main__":
    main()
