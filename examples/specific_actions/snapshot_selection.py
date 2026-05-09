"""Demonstrate SnapshotCollection access patterns with labeled GIF output.

Trains a TinyMLP for 40 epochs, then shows five selections:
  1. All layers, all epochs
  2. Early training  (epochs  0–19)
  3. Late  training  (epochs 20–39)
  4. fc1.weight only, all epochs
  5. Late training + fc1.weight  (combined select)

Each selection is saved as a timestamped GIF under ../../outputs/snapshot_selection/.

_make_gif pre-fetches all needed tensors upfront via col.by_layer() — which
reads each layer across all epochs in parallel — then renders frames from an
in-memory cache with zero additional disk I/O.

Requires matplotlib:
    pip install "neuroinquisitor[examples]"
    # or: pip install matplotlib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

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

    fig.suptitle(f"{title} — Epoch {epochs[0]}", fontsize=12)

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

    def update(frame_idx: int) -> list:
        epoch = epochs[frame_idx]
        fig.suptitle(f"{title} — Epoch {epoch}", fontsize=12)
        for im, layer in zip(images, layers):
            im.set_data(cache[epoch][layer])   # in-memory, no disk I/O
        return images

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
    cfg_path = Path(__file__).parent.parent / "configs" / "specific_actions_snapshot_selection.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(0)

    X = torch.randn(cfg["n_samples"], 4)
    y = (X.sum(dim=1, keepdim=True) > 0).float()

    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = Path(__file__).parent.parent.parent / "outputs" / "network_weights" / ts
    log_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = cfg["num_epochs"]
    print(f"Training TinyMLP for {num_epochs} epochs …")
    observer = NeuroInquisitor(
        model,
        log_dir=log_dir,
        format="hdf5",
        compress=True,
        create_new=True,
    )

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        observer.snapshot(epoch=epoch, metadata={"loss": loss.item()})

    observer.close()
    print(f"Snapshots written to {log_dir}/\n")

    col = NeuroInquisitor.load(log_dir, format="hdf5")
    print(f"SnapshotCollection: {col}\n")

    out_dir = Path(__file__).parent.parent.parent / "outputs" / "snapshot_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_layers = ["fc1.weight", "fc2.weight"]
    gif_fps    = cfg["gif_fps"]
    mid        = num_epochs // 2

    # --- Selection 1: all layers, all epochs ---
    print("Selection 1 — all layers, all epochs")
    print("  col.select()")
    _make_gif(col.select(), all_layers, "All layers · all epochs",
              out_dir / f"{ts}_all_layers_all_epochs.gif", fps=gif_fps)

    # --- Selection 2: early training (first half) ---
    print(f"\nSelection 2 — early training, epochs 0–{mid - 1}")
    print(f"  col.select(epochs=range(0, {mid}))")
    _make_gif(col.select(epochs=range(0, mid)), all_layers, f"All layers · epochs 0–{mid - 1} (early)",
              out_dir / f"{ts}_early_training_epochs_00_{mid - 1:02d}.gif", fps=gif_fps)

    # --- Selection 3: late training (second half) ---
    print(f"\nSelection 3 — late training, epochs {mid}–{num_epochs - 1}")
    print(f"  col.select(epochs=range({mid}, {num_epochs}))")
    _make_gif(col.select(epochs=range(mid, num_epochs)), all_layers, f"All layers · epochs {mid}–{num_epochs - 1} (late)",
              out_dir / f"{ts}_late_training_epochs_{mid:02d}_{num_epochs - 1:02d}.gif", fps=gif_fps)

    # --- Selection 4: fc1.weight only, all epochs ---
    print("\nSelection 4 — fc1.weight only, all epochs")
    print('  col.select(layers="fc1.weight")')
    _make_gif(col.select(layers="fc1.weight"), ["fc1.weight"], "fc1.weight only · all epochs",
              out_dir / f"{ts}_fc1_weight_all_epochs.gif", fps=gif_fps)

    # --- Selection 5: late training + fc1.weight (combined) ---
    print(f"\nSelection 5 — late training + fc1.weight combined")
    print(f'  col.select(epochs=range({mid}, {num_epochs}), layers="fc1.weight")')
    _make_gif(col.select(epochs=range(mid, num_epochs), layers="fc1.weight"), ["fc1.weight"],
              f"fc1.weight · epochs {mid}–{num_epochs - 1}",
              out_dir / f"{ts}_late_fc1_weight.gif", fps=gif_fps)

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
