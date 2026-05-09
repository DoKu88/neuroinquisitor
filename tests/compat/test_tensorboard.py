"""TensorBoard Projector compatibility test (NI-GAMMA-007).

Verifies that projection_embed output can be written to the TSV format
TensorBoard Projector expects using only standard pandas/numpy calls.

Skip condition: tensorboard not installed.
Install:        pip install tensorboard

Run standalone:
    pytest tests/compat/test_tensorboard.py -v
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip(
    "tensorboard",
    reason=(
        "tensorboard is not installed — TensorBoard Projector test skipped.\n"
        "To run this test: pip install tensorboard"
    ),
)

from neuroinquisitor.analyzers import projection_embed


def test_projection_embed_to_tensorboard_tsv(tmp_path: Path) -> None:
    """projection_embed → TSV files readable by TensorBoard Projector.

    Only standard pandas/numpy calls appear after the analyzer call.
    No NI-specific types in this flow.
    """
    acts = {"fc1": torch.randn(32, 16)}

    df = projection_embed(acts, n_components=2)
    layer_df = df[df["layer"] == "fc1"].reset_index(drop=True)

    tensor_tsv = tmp_path / "tensor.tsv"
    metadata_tsv = tmp_path / "metadata.tsv"

    np.savetxt(tensor_tsv, layer_df[["component_0", "component_1"]].values, delimiter="\t")
    layer_df[["sample_idx"]].to_csv(metadata_tsv, sep="\t", index=False)

    # Verify the files are well-formed
    coords = np.loadtxt(tensor_tsv, delimiter="\t")
    assert coords.shape == (32, 2)

    meta = pd.read_csv(metadata_tsv, sep="\t")
    assert "sample_idx" in meta.columns
    assert len(meta) == 32


def test_tensorboard_tsv_no_ni_types(tmp_path: Path) -> None:
    """No NI-specific types appear after the initial projection_embed call."""
    acts = {"fc1": torch.randn(10, 8)}
    df = projection_embed(acts, n_components=2)

    # Everything from here is plain pandas/numpy
    layer_df = df[df["layer"] == "fc1"].reset_index(drop=True)
    assert isinstance(layer_df, pd.DataFrame)

    buf = io.StringIO()
    np.savetxt(buf, layer_df[["component_0", "component_1"]].values, delimiter="\t")
    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 10
