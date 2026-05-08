"""Basic usage example: train a tiny MLP, snapshot weights each epoch, plot a GIF.

Requires matplotlib:
    pip install "neuroinquisitor[examples]"
    # or: pip install matplotlib
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor

# FC layers to visualise: (snapshot_key, display_label)
FC_LAYERS = [
    ("fc1.weight", "fc1  (16 × 4)"),
    ("fc2.weight", "fc2  (1 × 16)"),
]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _make_gif(
    weight_history: list[dict[str, np.ndarray]],
    out_path: Path,
    fps: int = 4,
) -> None:
    """Save a GIF with one subplot per FC layer, weights as heatmaps over time."""
    n_epochs = len(weight_history)
    n_layers = len(FC_LAYERS)

    # Determine a fixed, symmetric colour scale per layer so changes are visible.
    vlims: list[tuple[float, float]] = []
    for key, _ in FC_LAYERS:
        all_vals = np.concatenate([snap[key].ravel() for snap in weight_history])
        abs_max = float(np.abs(all_vals).max())
        vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(
        1, n_layers, figsize=(5 * n_layers, 4), constrained_layout=True
    )
    if n_layers == 1:
        axes = [axes]

    # Initial frame
    images = []
    for ax, (key, label), (vmin, vmax) in zip(axes, FC_LAYERS, vlims):
        im = ax.imshow(
            weight_history[0][key],
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("input dim")
        ax.set_ylabel("output dim")
        images.append(im)

    epoch_text = fig.text(0.5, 1.01, "", ha="center", va="bottom", fontsize=12)

    def update(frame: int) -> list:
        epoch_text.set_text(f"Epoch {frame}")
        for im, (key, _) in zip(images, FC_LAYERS):
            im.set_data(weight_history[frame][key])
        return [*images, epoch_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_epochs,
        interval=1000 // fps,
        blit=False,
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)

    # --- toy dataset: binary classification ---
    X = torch.randn(128, 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()

    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    log_dir = Path(tempfile.mkdtemp())

    print(f"Writing snapshots to: {log_dir}/")

    observer = NeuroInquisitor(
        model,
        log_dir=log_dir,
        compress=True,
        create_new=True,
    )

    num_epochs = 30
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        observer.snapshot(epoch=epoch, metadata={"loss": loss.item(), "lr": lr})
        print(f"  epoch {epoch:2d}  loss={loss.item():.4f}")

    observer.close()
    print(f"\nDone. {num_epochs} snapshots written.")

    # --- load all snapshots for visualisation ---
    col = NeuroInquisitor.load(log_dir)
    weight_history = [col.by_epoch(e) for e in range(num_epochs)]

    # --- generate GIF ---
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = Path(__file__).parent.parent / "outputs" / "weight_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{timestamp}_weights_over_time.gif"

    print(f"\nGenerating GIF → {gif_path} ...")
    _make_gif(weight_history, gif_path, fps=4)
    print("Done.")


if __name__ == "__main__":
    main()
