"""neuroinquisitor — neural network weight observability for PyTorch."""

from neuroinquisitor.backends import Backend, LocalBackend
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.core import NeuroInquisitor
from neuroinquisitor.formats import Format, HDF5Format, SafeTensorsFormat
from neuroinquisitor.index import Index, IndexEntry, JSONIndex

__version__ = "0.1.0"
__all__ = [
    "NeuroInquisitor",
    "SnapshotCollection",
    "Backend",
    "LocalBackend",
    "Format",
    "HDF5Format",
    "SafeTensorsFormat",
    "Index",
    "IndexEntry",
    "JSONIndex",
    "__version__",
]
