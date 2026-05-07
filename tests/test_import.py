"""Smoke tests: package imports and basic attributes are present."""

import pytest

import neuroinquisitor


def test_version_exists() -> None:
    assert hasattr(neuroinquisitor, "__version__")
    assert isinstance(neuroinquisitor.__version__, str)
    assert neuroinquisitor.__version__ != ""


def test_version_value() -> None:
    assert neuroinquisitor.__version__ == "0.1.0"


def test_neuroinquisitor_class_exported() -> None:
    from neuroinquisitor import NeuroInquisitor  # noqa: F401

    assert NeuroInquisitor is not None


def test_neuroinquisitor_not_yet_implemented() -> None:
    from neuroinquisitor import NeuroInquisitor

    with pytest.raises(NotImplementedError):
        NeuroInquisitor()
