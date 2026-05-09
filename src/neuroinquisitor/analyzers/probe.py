"""probe_linear analyzer (NI-GAMMA-005)."""

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


def _fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Least-squares linear classifier; returns (train_accuracy, val_accuracy)."""
    n_classes = int(y_train.max()) + 1
    Y = np.zeros((len(y_train), n_classes), dtype=np.float64)
    Y[np.arange(len(y_train)), y_train] = 1.0
    W, _, _, _ = np.linalg.lstsq(X_train.astype(np.float64), Y, rcond=None)
    train_acc = float((np.argmax(X_train @ W, axis=1) == y_train).mean())
    val_acc = float((np.argmax(X_val @ W, axis=1) == y_val).mean())
    return train_acc, val_acc


def probe_linear(
    activations: dict[str, torch.Tensor],
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a linear probe on each layer's activations with a deterministic split.

    Uses a least-squares linear classifier (no external ML library required).
    The train/val split is fully determined by ``random_state``; calling this
    function twice with the same arguments returns identical results.

    Parameters
    ----------
    activations:
        Layer-keyed activation tensors, as returned by ``ReplayResult.activations``.
    labels:
        Integer class labels of shape ``(N,)``.
    test_size:
        Fraction of samples to hold out for validation.
    random_state:
        Seed for the permutation that produces the train/val split.

    Returns
    -------
    pd.DataFrame
        Columns: ``layer``, ``train_accuracy``, ``val_accuracy``.
    """
    y = labels.detach().cpu().numpy().astype(int)
    n = len(y)
    n_val = max(1, int(round(n * test_size)))
    idx = np.random.default_rng(random_state).permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    rows: list[dict[str, object]] = []
    for layer, tensor in activations.items():
        X = _to_numpy_2d(tensor)
        train_acc, val_acc = _fit_probe(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        rows.append({"layer": layer, "train_accuracy": train_acc, "val_accuracy": val_acc})

    return pd.DataFrame(rows, columns=["layer", "train_accuracy", "val_accuracy"])
