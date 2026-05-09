"""Built-in analyzers for neuroinquisitor (NI-GAMMA-001 through NI-GAMMA-005).

All analyzers accept standard Python/NumPy/PyTorch types and return a
``pd.DataFrame`` the caller can immediately plot, filter, or export.

Quick reference
---------------
- :func:`trajectory_stats` — per-epoch weight-trajectory metrics for one layer
- :func:`spectrum_rank`    — singular-value and rank metrics per layer per epoch
- :func:`projection_embed` — PCA / UMAP projection of activation tensors
- :func:`similarity_compare` — linear CKA between two activation dicts
- :func:`probe_linear`    — linear probe accuracy per layer
"""

from neuroinquisitor.analyzers.probe import probe_linear
from neuroinquisitor.analyzers.projection import projection_embed
from neuroinquisitor.analyzers.similarity import similarity_compare
from neuroinquisitor.analyzers.spectrum import spectrum_rank
from neuroinquisitor.analyzers.trajectory import trajectory_stats

__all__ = [
    "trajectory_stats",
    "spectrum_rank",
    "projection_embed",
    "similarity_compare",
    "probe_linear",
]
