"""Parquet writer for derived tabular artifacts (NI-DELTA-003).

Every table produced by a NI analyzer must include the provenance columns
defined in :data:`PROVENANCE_COLUMNS`.  Files written here are readable with
``pd.read_parquet()`` without importing neuroinquisitor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    {"run_id", "epoch", "layer", "analyzer_name", "analyzer_version"}
)


def write_derived_table(
    df: Any,  # pandas DataFrame; pandas is an optional dep
    path: str | Path,
) -> Path:
    """Write *df* to a Parquet file at *path*.

    Validates that all required provenance columns are present before writing.
    Creates parent directories as needed.

    Parameters
    ----------
    df:
        DataFrame to write.  Must contain all columns in
        :data:`PROVENANCE_COLUMNS`.
    path:
        Destination file path (including ``.parquet`` suffix).

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    ImportError
        When ``pandas`` or ``pyarrow`` are not installed.
    ValueError
        When required provenance columns are missing from *df*.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for Parquet output. "
            "Install it with: pip install neuroinquisitor[parquet]"
        ) from exc

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyarrow is required for Parquet output. "
            "Install it with: pip install neuroinquisitor[parquet]"
        ) from exc

    missing = PROVENANCE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required provenance columns: {sorted(missing)}. "
            f"Required: {sorted(PROVENANCE_COLUMNS)}"
        )

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output
