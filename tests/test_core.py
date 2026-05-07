"""Tests for NeuroInquisitor core class (Sprint 2)."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import pytest
import torch  # noqa: F401
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Linear(4, 2)


@pytest.fixture()
def h5_path(tmp_path: Path) -> Path:
    return tmp_path / "weights.h5"


# ---------------------------------------------------------------------------
# Instantiation & file creation
# ---------------------------------------------------------------------------


def test_instantiation_creates_hdf5(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, filename="weights.h5")
    assert (tmp_path / "weights.h5").exists()
    obs.close()


def test_hdf5_file_is_valid_after_construction(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, filename="weights.h5")
    obs.close()
    with h5py.File(tmp_path / "weights.h5", "r") as f:
        assert isinstance(f, h5py.File)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_finalizes_file(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.close()
    assert obs._closed is True
    assert not obs._file.id.valid  # h5py file handle is no longer open


def test_double_close_is_noop(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.close()
    obs.close()  # must not raise


# ---------------------------------------------------------------------------
# create_new semantics
# ---------------------------------------------------------------------------


def test_create_new_true_raises_when_file_exists(
    simple_model: nn.Module, h5_path: Path
) -> None:
    h5_path.touch()
    with pytest.raises(FileExistsError, match=str(h5_path)):
        NeuroInquisitor(
            simple_model,
            log_dir=h5_path.parent,
            filename=h5_path.name,
            create_new=True,
        )


def test_create_new_false_raises_when_file_missing(
    simple_model: nn.Module, h5_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match=str(h5_path)):
        NeuroInquisitor(
            simple_model,
            log_dir=h5_path.parent,
            filename=h5_path.name,
            create_new=False,
        )


def test_create_new_false_opens_existing_file(
    simple_model: nn.Module, h5_path: Path
) -> None:
    # Create the file first
    obs1 = NeuroInquisitor(
        simple_model,
        log_dir=h5_path.parent,
        filename=h5_path.name,
        create_new=True,
    )
    obs1.close()

    # Now open it in append mode
    obs2 = NeuroInquisitor(
        simple_model,
        log_dir=h5_path.parent,
        filename=h5_path.name,
        create_new=False,
    )
    assert obs2._file.mode == "r+"  # h5py reports append as "r+"
    obs2.close()


# ---------------------------------------------------------------------------
# __del__ ResourceWarning
# ---------------------------------------------------------------------------


def test_del_without_close_emits_resource_warning(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs.__del__()
    assert any(issubclass(w.category, ResourceWarning) for w in caught)


def test_del_after_close_no_warning(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obs.__del__()
    assert not any(issubclass(w.category, ResourceWarning) for w in caught)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_model_name(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    r = repr(obs)
    assert "Linear" in r
    obs.close()


def test_repr_shows_status(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    assert "open" in repr(obs)
    obs.close()
    assert "closed" in repr(obs)


def test_repr_shows_filepath(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, filename="myweights.h5")
    assert "myweights.h5" in repr(obs)
    obs.close()


def test_repr_shows_compress_flag(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, compress=True)
    assert "compress=True" in repr(obs)
    obs.close()


# ---------------------------------------------------------------------------
# log_dir creation
# ---------------------------------------------------------------------------


def test_nested_log_dir_is_created(simple_model: nn.Module, tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    obs = NeuroInquisitor(simple_model, log_dir=nested)
    assert nested.exists()
    obs.close()
