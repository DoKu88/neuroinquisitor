"""Tests for the Parquet derived-table writer (NI-DELTA-003).

Verifies:
- write_derived_table writes a file openable with pd.read_parquet() alone
- All required provenance columns are enforced
- Missing provenance columns raise ValueError
- Parquet files contain correct data after round-trip
- Writer utility is importable from stable public path
"""

from __future__ import annotations

from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd

from neuroinquisitor.artifacts import PROVENANCE_COLUMNS, write_derived_table


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _minimal_df(**extra: object) -> pd.DataFrame:
    """Return a DataFrame with all required provenance columns."""
    data: dict[str, list[object]] = {
        "run_id": ["run_a"],
        "epoch": [1],
        "layer": ["fc1.weight"],
        "analyzer_name": ["trajectory_stats"],
        "analyzer_version": ["0.1.0"],
    }
    data.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


def test_write_derived_table_importable_from_artifacts() -> None:
    from neuroinquisitor.artifacts import write_derived_table  # noqa: F401


def test_write_derived_table_importable_from_top_level() -> None:
    from neuroinquisitor import write_derived_table  # noqa: F401


def test_provenance_columns_importable() -> None:
    from neuroinquisitor import PROVENANCE_COLUMNS as PC
    assert isinstance(PC, frozenset)


# ---------------------------------------------------------------------------
# Provenance column enforcement
# ---------------------------------------------------------------------------


def test_all_provenance_columns_required() -> None:
    assert PROVENANCE_COLUMNS == frozenset(
        {"run_id", "epoch", "layer", "analyzer_name", "analyzer_version"}
    )


@pytest.mark.parametrize("missing_col", list(PROVENANCE_COLUMNS))
def test_missing_provenance_column_raises(
    tmp_path: Path,
    missing_col: str,
) -> None:
    df = _minimal_df()
    df = df.drop(columns=[missing_col])
    with pytest.raises(ValueError, match=missing_col):
        write_derived_table(df, tmp_path / "out.parquet")


def test_no_file_written_when_validation_fails(tmp_path: Path) -> None:
    df = _minimal_df().drop(columns=["run_id"])
    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError):
        write_derived_table(df, out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Write and round-trip
# ---------------------------------------------------------------------------


def test_write_creates_parquet_file(tmp_path: Path) -> None:
    df = _minimal_df(value=3.14)
    out = write_derived_table(df, tmp_path / "result.parquet")
    assert out.exists()
    assert out.suffix == ".parquet"


def test_written_file_readable_without_ni(tmp_path: Path) -> None:
    """pd.read_parquet must work with no neuroinquisitor import."""
    df = _minimal_df(score=0.95)
    out = write_derived_table(df, tmp_path / "result.parquet")
    loaded = pd.read_parquet(out)
    assert "score" in loaded.columns
    assert loaded["score"].iloc[0] == pytest.approx(0.95)


def test_parquet_provenance_columns_present_after_read(tmp_path: Path) -> None:
    df = _minimal_df()
    out = write_derived_table(df, tmp_path / "result.parquet")
    loaded = pd.read_parquet(out)
    for col in PROVENANCE_COLUMNS:
        assert col in loaded.columns, f"Missing provenance column: {col}"


def test_parquet_data_integrity(tmp_path: Path) -> None:
    df = _minimal_df(l2_dist=0.42, cosine_sim=0.98)
    out = write_derived_table(df, tmp_path / "result.parquet")
    loaded = pd.read_parquet(out)
    assert loaded["run_id"].iloc[0] == "run_a"
    assert loaded["epoch"].iloc[0] == 1
    assert loaded["layer"].iloc[0] == "fc1.weight"
    assert loaded["analyzer_name"].iloc[0] == "trajectory_stats"
    assert loaded["l2_dist"].iloc[0] == pytest.approx(0.42)


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    deep_path = tmp_path / "a" / "b" / "c" / "result.parquet"
    df = _minimal_df()
    write_derived_table(df, deep_path)
    assert deep_path.exists()


def test_returns_absolute_path(tmp_path: Path) -> None:
    df = _minimal_df()
    out = write_derived_table(df, tmp_path / "result.parquet")
    assert out.is_absolute()


def test_multiple_rows(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "run_id": ["run_a", "run_a", "run_a"],
            "epoch": [0, 1, 2],
            "layer": ["fc1.weight"] * 3,
            "analyzer_name": ["spectrum_rank"] * 3,
            "analyzer_version": ["0.1.0"] * 3,
            "effective_rank": [4.2, 3.8, 3.1],
        }
    )
    out = write_derived_table(rows, tmp_path / "spectrum.parquet")
    loaded = pd.read_parquet(out)
    assert len(loaded) == 3
    assert list(loaded["epoch"]) == [0, 1, 2]
