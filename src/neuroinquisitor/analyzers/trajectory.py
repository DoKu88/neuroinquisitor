"""trajectory_stats analyzer (NI-GAMMA-001)."""

from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required for neuroinquisitor analyzers. "
        "Install with: pip install neuroinquisitor[analyzers]"
    ) from exc

_COLS = [
    "epoch",
    "l2_from_init",
    "cosine_to_init",
    "cosine_to_final",
    "update_norm",
    "velocity",
    "acceleration",
]


def trajectory_stats(weights: dict[int, np.ndarray]) -> pd.DataFrame:
    """Compute per-epoch trajectory statistics for one layer.

    Parameters
    ----------
    weights:
        Epoch-keyed weight arrays, as returned by
        ``SnapshotCollection.by_layer(name)``.

    Returns
    -------
    pd.DataFrame
        Columns: ``epoch``, ``l2_from_init``, ``cosine_to_init``,
        ``cosine_to_final``, ``update_norm``, ``velocity``, ``acceleration``.
        ``update_norm`` and ``velocity`` are ``NaN`` for the first epoch;
        ``acceleration`` is ``NaN`` for the first two epochs.
    """
    if not weights:
        return pd.DataFrame(columns=_COLS)

    epochs = sorted(weights)
    vecs = {e: weights[e].ravel().astype(np.float64) for e in epochs}
    w_init = vecs[epochs[0]]
    w_final = vecs[epochs[-1]]

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    rows = []
    for i, e in enumerate(epochs):
        w = vecs[e]
        rows.append(
            {
                "epoch": e,
                "l2_from_init": float(np.linalg.norm(w - w_init)),
                "cosine_to_init": _cosine(w, w_init),
                "cosine_to_final": _cosine(w, w_final),
                "update_norm": float("nan")
                if i == 0
                else float(np.linalg.norm(w - vecs[epochs[i - 1]])),
            }
        )

    df = pd.DataFrame(rows)
    df["velocity"] = df["update_norm"]
    df["acceleration"] = df["velocity"].diff()
    return df[_COLS].reset_index(drop=True)
