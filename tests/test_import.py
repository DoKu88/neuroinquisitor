"""Smoke tests: package imports and public API surface."""

import neuroinquisitor


def test_version_exists() -> None:
    assert hasattr(neuroinquisitor, "__version__")
    assert isinstance(neuroinquisitor.__version__, str)
    assert neuroinquisitor.__version__ != ""


def test_version_value() -> None:
    assert neuroinquisitor.__version__ == "0.1.0"


def test_dunder_all_contains_expected_names() -> None:
    expected = {
        "NeuroInquisitor",
        "SnapshotCollection",
        "Backend",
        "LocalBackend",
        "Format",
        "HDF5Format",
        "Index",
        "IndexEntry",
        "JSONIndex",
        "__version__",
    }
    assert hasattr(neuroinquisitor, "__all__")
    assert expected <= set(neuroinquisitor.__all__)


def test_all_names_actually_importable() -> None:
    for name in neuroinquisitor.__all__:
        assert hasattr(neuroinquisitor, name), f"{name!r} in __all__ but not importable"


def test_neuroinquisitor_public_methods() -> None:
    from neuroinquisitor import NeuroInquisitor

    for method in ("snapshot", "load", "close"):
        assert hasattr(NeuroInquisitor, method), f"NeuroInquisitor missing method {method!r}"


def test_snapshot_collection_public_methods() -> None:
    from neuroinquisitor import SnapshotCollection

    for method in ("by_epoch", "by_layer", "select", "epochs", "layers"):
        assert hasattr(SnapshotCollection, method), f"SnapshotCollection missing {method!r}"


def test_backend_abc_exported() -> None:
    from neuroinquisitor import Backend, LocalBackend

    assert issubclass(LocalBackend, Backend)


def test_format_abc_exported() -> None:
    from neuroinquisitor import Format, HDF5Format

    assert issubclass(HDF5Format, Format)


def test_index_abc_exported() -> None:
    from neuroinquisitor import Index, IndexEntry, JSONIndex

    assert issubclass(JSONIndex, Index)
    entry = IndexEntry()
    assert hasattr(entry, "epoch")
    assert hasattr(entry, "step")
    assert hasattr(entry, "file_key")
    assert hasattr(entry, "layers")
    assert hasattr(entry, "metadata")


def test_safetensors_format_importable_when_available() -> None:
    import importlib

    if importlib.util.find_spec("safetensors") is None:
        return  # safetensors not installed — nothing to test

    from neuroinquisitor.formats import Format
    from neuroinquisitor.formats.safetensors_format import SafetensorsFormat

    assert issubclass(SafetensorsFormat, Format)


def test_s3_backend_importable_when_available() -> None:
    import importlib

    if importlib.util.find_spec("boto3") is None:
        return  # boto3 not installed — nothing to test

    from neuroinquisitor.backends import Backend
    from neuroinquisitor.backends.s3 import S3Backend

    assert issubclass(S3Backend, Backend)
