from neuroinquisitor.formats.base import Format
from neuroinquisitor.formats.hdf5_format import HDF5Format

__all__ = ["Format", "HDF5Format"]

try:  # optional: safetensors
    from neuroinquisitor.formats.safetensors_format import SafetensorsFormat

    __all__.append("SafetensorsFormat")
except ImportError:  # pragma: no cover
    pass
