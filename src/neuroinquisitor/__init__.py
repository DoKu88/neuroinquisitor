"""neuroinquisitor — neural network weight observability for PyTorch."""

from neuroinquisitor.analysis import (
    AnalyzerRequest,
    AnalyzerResult,
    AnalyzerSpec,
    get_analyzer,
    list_analyzers,
    register,
)
from neuroinquisitor.artifacts import PROVENANCE_COLUMNS, write_derived_table
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
    "AnalyzerRequest",
    "AnalyzerResult",
    "AnalyzerSpec",
    "register",
    "get_analyzer",
    "list_analyzers",
    "write_derived_table",
    "PROVENANCE_COLUMNS",
    "__version__",
]
