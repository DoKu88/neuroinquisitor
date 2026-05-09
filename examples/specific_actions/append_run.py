"""Store weights: append to an existing run (create_new=False).

Useful when training is split across multiple sessions (e.g. resuming from
a checkpoint).  Each session opens the same log_dir in append mode and adds
new snapshots without disturbing existing ones.

Run:
    python examples/append_run.py
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
    X = torch.randn(64, 4)
    y = (X.sum(1, keepdim=True) > 0).float()
    loss_fn = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "outputs" / "append_run" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    # --- session 1: epochs 0–4 ---
    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    observer = NeuroInquisitor(model, log_dir=log_dir, create_new=True)
    for epoch in range(5):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)
    observer.close()
    print(f"session 1 done  — {NeuroInquisitor.load(log_dir).epochs}")

    # --- session 2: resume, epochs 5–9 ---
    # model state would normally be restored from a checkpoint here
    observer = NeuroInquisitor(model, log_dir=log_dir, create_new=False)
    for epoch in range(5, 10):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch)
    observer.close()
    print(f"session 2 done  — {NeuroInquisitor.load(log_dir).epochs}")

    # --- load the full run ---
    col = NeuroInquisitor.load(log_dir)
    print(f"\ncombined epochs : {col.epochs}")
    print(f"total snapshots : {len(col)}")

    # weights from session 1 are still intact
    w0 = col.by_epoch(0)
    print(f"\nepoch 0 layers: {list(w0.keys())}")


if __name__ == "__main__":
    main()
