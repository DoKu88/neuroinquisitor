"""Store weights: epoch-based snapshotting with metadata and compression.

Run:
    python examples/store_epochs.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from neuroinquisitor import NeuroInquisitor


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x).relu())


def main() -> None:
    cfg_path = Path(__file__).parent.parent / "configs" / "specific_actions_store_epochs.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(0)
    X = torch.randn(cfg["n_samples"], 4)
    y = (X.sum(1, keepdim=True) > 0).float()

    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent.parent / "outputs" / "store_epochs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"log_dir: {log_dir}\n")

    # --- snapshot every epoch, attaching metadata and using compression ---
    observer = NeuroInquisitor(
        model,
        log_dir=log_dir,
        compress=True,   # gzip compression inside each .h5 file
        create_new=True,
    )

    for epoch in range(cfg["num_epochs"]):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        observer.snapshot(
            epoch=epoch,
            metadata={"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]},
        )
        print(f"epoch {epoch}  loss={loss.item():.4f}  → snapshot saved")

    observer.close()

    # one .h5 file per epoch + one index.json
    saved = sorted(log_dir.iterdir())
    print(f"\nfiles in log_dir: {[f.name for f in saved]}")

    col = NeuroInquisitor.load(log_dir)
    print(f"epochs recorded : {col.epochs}")
    print(f"layers tracked  : {col.layers}")


if __name__ == "__main__":
    main()
