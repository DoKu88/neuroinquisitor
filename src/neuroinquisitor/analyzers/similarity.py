"""similarity_compare analyzer (NI-GAMMA-004)."""

from __future__ import annotations

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


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between centered feature matrices X (n×p) and Y (n×q).

    Returns NaN when either matrix is all-zeros.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    norm_xx = float(np.linalg.norm(X.T @ X, "fro"))
    norm_yy = float(np.linalg.norm(Y.T @ Y, "fro"))
    if norm_xx == 0.0 or norm_yy == 0.0:
        return float("nan")
    hsic_xy = float(np.linalg.norm(X.T @ Y, "fro") ** 2)
    return hsic_xy / (norm_xx * norm_yy)


def similarity_compare(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
) -> pd.DataFrame:
    """Compute linear CKA between all layer pairs across two activation dicts.

    Useful for comparing layers across epochs (within-run) or across runs.
    For a diagonal comparison (same layer at two epochs), filter the result
    with ``df[df.layer_a == df.layer_b]``.

    Parameters
    ----------
    a, b:
        Layer-keyed activation tensors, e.g. ``ReplayResult.activations`` at
        two different epochs or from two different runs.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer_a``, ``layer_b``, ``cka``.

    Raises
    ------
    ValueError
        If any (layer_a, layer_b) pair has mismatched sample dimensions.
    """
    rows: list[dict[str, object]] = []
    for la, ta in a.items():
        Xa = _to_numpy_2d(ta)
        for lb, tb in b.items():
            Xb = _to_numpy_2d(tb)
            if Xa.shape[0] != Xb.shape[0]:
                raise ValueError(
                    f"Sample dimension mismatch for ({la!r}, {lb!r}): "
                    f"{Xa.shape[0]} vs {Xb.shape[0]}. "
                    "Both dicts must be computed over the same dataset slice."
                )
            rows.append({"layer_a": la, "layer_b": lb, "cka": _linear_cka(Xa, Xb)})

    return pd.DataFrame(rows, columns=["layer_a", "layer_b", "cka"])
