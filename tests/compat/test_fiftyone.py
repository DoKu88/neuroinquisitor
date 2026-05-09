"""FiftyOne compatibility test (NI-GAMMA-007).

Verifies that projection_embed output can be attached to a FiftyOne dataset
as embeddings using no NI-specific glue code after the analyzer call.

Skip condition: fiftyone not installed.
Install:        pip install fiftyone

Run standalone:
    pytest tests/compat/test_fiftyone.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

fo = pytest.importorskip(
    "fiftyone",
    reason=(
        "fiftyone is not installed — FiftyOne compatibility test skipped.\n"
        "To run this test: pip install fiftyone"
    ),
)

from neuroinquisitor.analyzers import projection_embed


def test_projection_embed_to_fiftyone_dataset() -> None:
    """projection_embed → FiftyOne dataset embeddings.

    Only standard fiftyone/pandas/numpy calls appear after the analyzer call.
    No NI-specific types in this flow.
    """
    n_samples = 12
    acts = {"fc1": torch.randn(n_samples, 8)}

    df = projection_embed(acts, n_components=2)
    layer_df = df[df["layer"] == "fc1"].reset_index(drop=True)

    dataset = fo.Dataset()
    samples = [fo.Sample(filepath=f"/dev/null/{i}.jpg") for i in range(n_samples)]
    dataset.add_samples(samples)

    embeddings = layer_df[["component_0", "component_1"]].values
    dataset.set_values("embedding", embeddings.tolist())

    stored = np.array(dataset.values("embedding"))
    assert stored.shape == (n_samples, 2)

    dataset.delete()


def test_fiftyone_no_ni_types_in_flow() -> None:
    """After projection_embed, all objects are plain fiftyone/numpy types."""
    acts = {"layer": torch.randn(5, 4)}
    df = projection_embed(acts, n_components=2)
    layer_df = df[df["layer"] == "layer"].reset_index(drop=True)

    # Plain numpy extraction — no NI type
    coords = layer_df[["component_0", "component_1"]].values
    assert isinstance(coords, np.ndarray)

    dataset = fo.Dataset()
    samples = [fo.Sample(filepath=f"/dev/null/{i}.jpg") for i in range(5)]
    dataset.add_samples(samples)
    dataset.set_values("embedding", coords.tolist())
    dataset.delete()
