"""neuroinquisitor — neural network weight observability for PyTorch."""

from neuroinquisitor.backends import Backend, LocalBackend
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.core import NeuroInquisitor
from neuroinquisitor.formats import Format, HDF5Format
from neuroinquisitor.index import Index, IndexEntry, JSONIndex
from neuroinquisitor.replay import (
    CheckpointSelector,
    ReplayConfig,
    ReplayMetadata,
    ReplayResult,
    ReplaySession,
    TensorMap,
)
from neuroinquisitor.schema import CapturePolicy, RunManifest, RunMetadata

# analyzers sub-package — import the module so `neuroinquisitor.analyzers` resolves
from neuroinquisitor import analyzers  # noqa: F401

__version__ = "0.1.0"
__all__ = [
    "NeuroInquisitor",
    "SnapshotCollection",
    "Backend",
    "LocalBackend",
    "Format",
    "HDF5Format",
    "Index",
    "IndexEntry",
    "JSONIndex",
    "CapturePolicy",
    "RunManifest",
    "RunMetadata",
    "ReplaySession",
    "ReplayConfig",
    "ReplayResult",
    "ReplayMetadata",
    "CheckpointSelector",
    "TensorMap",
    "analyzers",
    "__version__",
]
