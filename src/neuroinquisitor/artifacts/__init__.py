"""neuroinquisitor.artifacts — derived artifact writers."""

from neuroinquisitor.artifacts.parquet import PROVENANCE_COLUMNS, write_derived_table

__all__ = [
    "write_derived_table",
    "PROVENANCE_COLUMNS",
]
