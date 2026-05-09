"""FiftyOne embedding compatibility example (NI-GAMMA-007).

Attaches projection_embed coordinates to a FiftyOne dataset as embeddings.

Usage:
    pip install fiftyone
    python examples/specific_actions/fiftyone_embed.py
"""

from __future__ import annotations

import numpy as np
import torch

from neuroinquisitor.analyzers import projection_embed

try:
    import fiftyone as fo  # type: ignore[import-untyped]
    import fiftyone.core.fields as fof  # type: ignore[import-untyped]
except ImportError:
    raise SystemExit("Install fiftyone first: pip install fiftyone")

# Synthetic activations — replace with a real ReplayResult in practice.
n_samples = 16
acts = {"fc1": torch.randn(n_samples, 8)}

df = projection_embed(acts, n_components=2)
layer_df = df[df["layer"] == "fc1"].reset_index(drop=True)

dataset = fo.Dataset()
samples = [fo.Sample(filepath=f"/dev/null/{i}.jpg") for i in range(n_samples)]
dataset.add_samples(samples)

embeddings = layer_df[["component_0", "component_1"]].values
dataset.set_values("embedding", embeddings.tolist())

print(f"Dataset: {dataset.name}  ({len(dataset)} samples)")
print("Embedding shape per sample:", np.array(dataset.values("embedding")).shape)
