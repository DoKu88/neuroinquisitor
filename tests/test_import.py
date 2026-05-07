"""Smoke tests: package imports and basic attributes are present."""

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


def test_dunder_all_contains_expected_names() -> None:
    import neuroinquisitor

    assert hasattr(neuroinquisitor, "__all__")
    assert "NeuroInquisitor" in neuroinquisitor.__all__
    assert "__version__" in neuroinquisitor.__all__


