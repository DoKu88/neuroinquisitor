from neuroinquisitor.backends.base import Backend
from neuroinquisitor.backends.local import LocalBackend

__all__ = ["Backend", "LocalBackend"]

try:  # optional: boto3
    from neuroinquisitor.backends.s3 import S3Backend

    __all__.append("S3Backend")
except ImportError:  # pragma: no cover
    pass
