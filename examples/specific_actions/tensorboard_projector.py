"""TensorBoard Projector compatibility example (NI-GAMMA-007).

Converts a projection_embed DataFrame to the TSV format TensorBoard Projector
expects, then launches the projector.

Usage:
    pip install tensorboard
    python examples/specific_actions/tensorboard_projector.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from neuroinquisitor.analyzers import projection_embed

try:
    import tensorboard  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    raise SystemExit("Install tensorboard first: pip install tensorboard")

# Synthetic activations — replace with a real ReplayResult in practice.
acts = {"fc1": torch.randn(32, 16)}

df = projection_embed(acts, n_components=2)
layer_df = df[df["layer"] == "fc1"].reset_index(drop=True)

out = Path(tempfile.mkdtemp())
np.savetxt(out / "tensor.tsv", layer_df[["component_0", "component_1"]].values, delimiter="\t")
layer_df[["sample_idx"]].to_csv(out / "metadata.tsv", sep="\t", index=False)

print(f"Wrote tensor.tsv and metadata.tsv to {out}")
print("Launch with: tensorboard --logdir", out, "--port 6007")
print("Then open: http://localhost:6007/#projector")
