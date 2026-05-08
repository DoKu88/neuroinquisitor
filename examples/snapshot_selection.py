"""Demonstrate SnapshotCollection access patterns with labeled GIF output.

Trains a TinyMLP for 40 epochs, then shows five selections:
  1. All layers, all epochs
  2. Early training  (epochs  0–19)
  3. Late  training  (epochs 20–39)
  4. fc1.weight only, all epochs
  5. Late training + fc1.weight  (combined select)

Each selection is saved as a timestamped GIF under outputs/snapshot_selection/.

_make_gif pre-fetches all needed tensors upfront via col.by_layer() — which
reads each layer across all epochs in parallel — then renders frames from an
in-memory cache with zero additional disk I/O.

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

from neuroinquisitor import NeuroInquisitor, SnapshotCollection


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# GIF helper
# ---------------------------------------------------------------------------


def _make_gif(
    col: SnapshotCollection,
    layers: list[str],
    title: str,
    out_path: Path,
    fps: int = 4,
) -> None:
    """Animate *layers* from *col* as heatmaps over epochs and save a GIF.

    Pre-fetches all tensor data upfront using ``col.by_layer()`` (parallel
    reads via ThreadPoolExecutor), then renders every frame from an in-memory
    cache — no disk I/O during animation.
    """
    epochs = col.epochs
    n_layers = len(layers)

    # --- pre-fetch: one parallel batch read per layer ---
    # cache[epoch][layer] = np.ndarray
    layer_data: dict[str, dict[int, np.ndarray]] = {
        layer: col.by_layer(layer) for layer in layers
    }
    cache: dict[int, dict[str, np.ndarray]] = {
        epoch: {layer: layer_data[layer][epoch] for layer in layers}
        for epoch in epochs
    }

    # Fixed symmetric colour scale per layer across all included epochs.
    vlims: list[tuple[float, float]] = []
    for layer in layers:
        all_vals = np.concatenate([arr.ravel() for arr in layer_data[layer].values()])
        abs_max = float(np.abs(all_vals).max()) or 1.0
        vlims.append((-abs_max, abs_max))

    fig, axes = plt.subplots(
        1, n_layers, figsize=(5 * n_layers, 4), constrained_layout=True
    )
    if n_layers == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, y=1.02)

    images = []
    first = cache[epochs[0]]
    for ax, layer, (vmin, vmax) in zip(axes, layers, vlims):
        im = ax.imshow(
            first[layer],
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight value")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("input dim")
        ax.set_ylabel("output dim")
        images.append(im)

    epoch_label = fig.text(0.5, -0.02, "", ha="center", va="top", fontsize=11)

    def update(frame_idx: int) -> list:
        epoch = epochs[frame_idx]
        epoch_label.set_text(f"Epoch {epoch}")
        for im, layer in zip(images, layers):
            im.set_data(cache[epoch][layer])   # in-memory, no disk I/O
        return [*images, epoch_label]

    ani = animation.FuncAnimation(
        fig, update, frames=len(epochs), interval=1000 // fps, blit=False,
    )
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)

    X = torch.randn(128, 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()

    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    log_dir = Path(tempfile.mkdtemp())

    print("Training TinyMLP for 40 epochs …")
    observer = NeuroInquisitor(model, log_dir=log_dir, compress=True, create_new=True)

    num_epochs = 40
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        observer.snapshot(epoch=epoch, metadata={"loss": loss.item()})

    observer.close()
    print(f"Snapshots written to {log_dir}/\n")

    # load_all_snapshots reads only the index — no tensor files opened yet
    col = NeuroInquisitor.load(log_dir)
    print(f"SnapshotCollection: {col}\n")

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = Path(__file__).parent.parent / "outputs" / "snapshot_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_layers = ["fc1.weight", "fc2.weight"]

    # --- Selection 1: all layers, all epochs ---
    print("Selection 1 — all layers, all epochs")
    print("  col.select()")
    _make_gif(col.select(), all_layers, "All layers · all epochs",
              out_dir / f"{ts}_all_layers_all_epochs.gif")

    # --- Selection 2: early training (epochs 0–19) ---
    print("\nSelection 2 — early training, epochs 0–19")
    print("  col.select(epochs=range(0, 20))")
    _make_gif(col.select(epochs=range(0, 20)), all_layers, "All layers · epochs 0–19 (early)",
              out_dir / f"{ts}_early_training_epochs_00_19.gif")

    # --- Selection 3: late training (epochs 20–39) ---
    print("\nSelection 3 — late training, epochs 20–39")
    print("  col.select(epochs=range(20, 40))")
    _make_gif(col.select(epochs=range(20, 40)), all_layers, "All layers · epochs 20–39 (late)",
              out_dir / f"{ts}_late_training_epochs_20_39.gif")

    # --- Selection 4: fc1.weight only, all epochs ---
    print("\nSelection 4 — fc1.weight only, all epochs")
    print('  col.select(layers="fc1.weight")')
    _make_gif(col.select(layers="fc1.weight"), ["fc1.weight"], "fc1.weight only · all epochs",
              out_dir / f"{ts}_fc1_weight_all_epochs.gif")

    # --- Selection 5: late training + fc1.weight (combined) ---
    print("\nSelection 5 — late training + fc1.weight combined")
    print('  col.select(epochs=range(20, 40), layers="fc1.weight")')
    _make_gif(col.select(epochs=range(20, 40), layers="fc1.weight"), ["fc1.weight"],
              "fc1.weight · epochs 20–39", out_dir / f"{ts}_late_fc1_weight.gif")

    print(f"\nAll GIFs written to {out_dir}/")

    # --- console demo of by_layer and by_epoch ---
    print("\n--- by_layer('fc1.weight') — parallel read, epoch → shape ---")
    for epoch, arr in col.by_layer("fc1.weight").items():
        print(f"  epoch {epoch:2d}  shape={arr.shape}")

    print("\n--- by_epoch(0) ---")
    for name, arr in col.by_epoch(0).items():
        print(f"  {name:20s}  shape={arr.shape}")


if __name__ == "__main__":
    main()
