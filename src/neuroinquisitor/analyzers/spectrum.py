"""spectrum_rank analyzer (NI-GAMMA-002)."""

from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required for neuroinquisitor analyzers. "
        "Install with: pip install neuroinquisitor[analyzers]"
    ) from exc

_COLS = ["layer", "epoch", "spectral_norm", "frobenius_norm", "stable_rank", "effective_rank"]


def spectrum_rank(weights: dict[str, np.ndarray], *, epoch: int = 0) -> pd.DataFrame:
    """Compute spectral and rank metrics for each layer at one epoch snapshot.

    Parameters
    ----------
    weights:
        Layer-keyed weight arrays, as returned by ``SnapshotCollection.by_epoch(n)``.
    epoch:
        Epoch index to record in the result; pass the actual epoch when
        aggregating results across multiple calls.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``epoch``, ``spectral_norm``, ``frobenius_norm``,
        ``stable_rank``, ``effective_rank``.
        ``stable_rank`` and ``effective_rank`` are ``NaN`` for zero-weight layers.
    """
    rows = []
    for layer, arr in weights.items():
        mat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        sv = np.linalg.svd(mat, compute_uv=False)  # descending

        spectral = float(sv[0]) if len(sv) > 0 else 0.0
        frob = float(np.sqrt((sv**2).sum()))

        if spectral > 0.0:
            stable = float((sv**2).sum() / sv[0] ** 2)
        else:
            stable = float("nan")

        s2 = sv**2
        s2_sum = s2.sum()
        if s2_sum > 0.0:
            p = s2 / s2_sum
            nz = p > 0
            eff_rank = float(np.exp(-np.sum(p[nz] * np.log(p[nz]))))
        else:
            eff_rank = float("nan")

        rows.append(
            {
                "layer": layer,
                "epoch": epoch,
                "spectral_norm": spectral,
                "frobenius_norm": frob,
                "stable_rank": stable,
                "effective_rank": eff_rank,
            }
        )

    return pd.DataFrame(rows, columns=_COLS)
