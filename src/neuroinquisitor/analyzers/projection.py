"""projection_embed analyzer (NI-GAMMA-003)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required for neuroinquisitor analyzers. "
        "Install with: pip install neuroinquisitor[analyzers]"
    ) from exc


def _to_numpy_2d(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy().reshape(t.shape[0], -1)


def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA via truncated SVD; no external deps required."""
    X_c = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    coords = X_c @ Vt[:k].T
    if k < n_components:
        pad = np.zeros((X_c.shape[0], n_components - k), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords


def projection_embed(
    activations: dict[str, torch.Tensor],
    *,
    n_components: int = 2,
    method: Literal["pca", "umap"] = "pca",
) -> pd.DataFrame:
    """Project each layer's activations into a low-dimensional embedding.

    Parameters
    ----------
    activations:
        Layer-keyed activation tensors, as returned by ``ReplayResult.activations``.
    n_components:
        Number of embedding dimensions (2 or 3).
    method:
        ``"pca"`` (default, pure numpy) or ``"umap"``
        (requires ``pip install 'neuroinquisitor[umap]'``).

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``sample_idx``, ``component_0``, ``component_1``
        (and ``component_2`` when ``n_components=3``).
    """
    if method == "umap":
        try:
            import umap as _umap  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for UMAP projection. "
                "Install with: pip install 'neuroinquisitor[umap]'"
            ) from exc

    comp_cols = [f"component_{i}" for i in range(n_components)]
    rows: list[dict[str, object]] = []

    for layer, tensor in activations.items():
        X = _to_numpy_2d(tensor)
        n_samples = X.shape[0]

        if method == "pca":
            coords = _pca(X, n_components)
        else:
            reducer = _umap.UMAP(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(X)

        for i in range(n_samples):
            row: dict[str, object] = {"layer": layer, "sample_idx": i}
            for j, col in enumerate(comp_cols):
                row[col] = float(coords[i, j])
            rows.append(row)

    return pd.DataFrame(rows, columns=["layer", "sample_idx"] + comp_cols)
