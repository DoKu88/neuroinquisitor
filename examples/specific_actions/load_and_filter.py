"""Load weights: all access patterns for a SnapshotCollection.

Covers:
  - NeuroInquisitor.load()                          load everything (lazy)
  - NeuroInquisitor.load(epochs=..., layers=...)    filter at load time
  - col.by_epoch(N)                                 dict of all layers at one epoch
  - col.by_layer("name")                            one layer across all epochs (parallel)
  - col.select(epochs=..., layers=...)              refine an existing collection

Run:
    python examples/load_and_filter.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
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


def _train_and_save(log_dir: Path) -> None:
    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y = (X.sum(1, keepdim=True) > 0).float()
    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    observer = NeuroInquisitor(model, log_dir=log_dir)
    for epoch in range(10):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()
        observer.snapshot(epoch=epoch, metadata={"loss": loss_fn(model(X), y).item()})
    observer.close()


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "outputs" / "load_and_filter" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    _train_and_save(log_dir)

    # ------------------------------------------------------------------
    # 1. Load everything — lazy, only reads index.json
    # ------------------------------------------------------------------
    col = NeuroInquisitor.load(log_dir)
    print(f"all epochs  : {col.epochs}")
    print(f"all layers  : {col.layers}\n")

    # ------------------------------------------------------------------
    # 2. by_epoch — load all layers for one epoch
    # ------------------------------------------------------------------
    weights_e0 = col.by_epoch(0)   # dict[str, np.ndarray]
    for name, arr in weights_e0.items():
        print(f"epoch 0 | {name:20s}  shape={arr.shape}  dtype={arr.dtype}")
    print()

    # ------------------------------------------------------------------
    # 3. by_layer — load one layer across all epochs (parallel reads)
    # ------------------------------------------------------------------
    fc1_history = col.by_layer("fc1.weight")   # dict[int, np.ndarray]
    for epoch in sorted(fc1_history):
        arr = fc1_history[epoch]
        print(f"epoch {epoch}  fc1.weight  mean={arr.mean():.4f}")
    print()

    # ------------------------------------------------------------------
    # 4. Filter epochs at load time
    # ------------------------------------------------------------------
    late = NeuroInquisitor.load(log_dir, epochs=range(7, 10))
    print(f"late epochs (7-9): {late.epochs}")

    one_epoch = NeuroInquisitor.load(log_dir, epochs=5)
    print(f"single epoch    : {one_epoch.epochs}")
    print()

    # ------------------------------------------------------------------
    # 5. Filter layers at load time
    # ------------------------------------------------------------------
    fc1_only = NeuroInquisitor.load(log_dir, layers="fc1.weight")
    print(f"fc1.weight only layers: {fc1_only.layers}")
    print(f"fc1.weight only epochs: {fc1_only.epochs}")
    print()

    # ------------------------------------------------------------------
    # 6. Filter both at load time
    # ------------------------------------------------------------------
    subset = NeuroInquisitor.load(log_dir, epochs=range(0, 5), layers=["fc1.weight", "fc1.bias"])
    print(f"epochs 0-4, fc1 params — epochs={subset.epochs}  layers={subset.layers}")
    print()

    # ------------------------------------------------------------------
    # 7. Refine an existing collection with select()
    # ------------------------------------------------------------------
    col = NeuroInquisitor.load(log_dir)
    early = col.select(epochs=range(0, 5))
    combined = early.select(layers="fc1.weight")
    print(f"select() chained — epochs={combined.epochs}  layers={combined.layers}")

    # verify weights changed across the selected epochs
    w0 = combined.by_epoch(0)["fc1.weight"]
    w4 = combined.by_epoch(4)["fc1.weight"]
    print(f"fc1.weight changed from epoch 0 to 4: {not np.allclose(w0, w4)}")


if __name__ == "__main__":
    main()
