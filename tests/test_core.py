"""Tests for NeuroInquisitor core class."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroinquisitor import NeuroInquisitor


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Linear(4, 2)


# ---------------------------------------------------------------------------
# Instantiation & index creation
# ---------------------------------------------------------------------------


def test_instantiation_creates_index(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    assert (tmp_path / "index.json").exists()
    obs.close()


def test_index_is_valid_json_after_construction(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    data = json.loads((tmp_path / "index.json").read_text())
    assert "snapshots" in data


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_sets_closed_flag(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.close()
    assert obs._closed is True


def test_double_close_is_noop(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.close()
    obs.close()  # must not raise


# ---------------------------------------------------------------------------
# create_new semantics
# ---------------------------------------------------------------------------


def test_create_new_true_raises_when_index_exists(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, create_new=True)
    obs.close()
    with pytest.raises(FileExistsError, match=str(tmp_path)):
        NeuroInquisitor(simple_model, log_dir=tmp_path, create_new=True)


def test_create_new_false_raises_when_index_missing(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        NeuroInquisitor(simple_model, log_dir=tmp_path, create_new=False)


def test_create_new_false_opens_existing_run(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    obs1 = NeuroInquisitor(simple_model, log_dir=tmp_path, create_new=True)
    obs1.snapshot(epoch=0)
    obs1.close()

    obs2 = NeuroInquisitor(simple_model, log_dir=tmp_path, create_new=False)
    obs2.snapshot(epoch=1)
    obs2.close()

    col = NeuroInquisitor.load(tmp_path)
    assert col.epochs == [0, 1]


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
    assert "Linear" in repr(obs)
    obs.close()


def test_repr_shows_status(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    assert "open" in repr(obs)
    obs.close()
    assert "closed" in repr(obs)


def test_repr_shows_log_dir(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    assert str(tmp_path) in repr(obs)
    obs.close()


def test_repr_shows_compress_flag(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, compress=True)
    assert "compress=True" in repr(obs)
    obs.close()


def test_repr_shows_backend_and_format(simple_model: nn.Module, tmp_path: Path) -> None:
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    r = repr(obs)
    assert "LocalBackend" in r
    assert "HDF5Format" in r
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


def test_log_dir_as_string(simple_model: nn.Module, tmp_path: Path) -> None:
    log_dir = str(tmp_path / "str_dir")
    obs = NeuroInquisitor(simple_model, log_dir=log_dir)
    assert (tmp_path / "str_dir").exists()
    obs.close()


# ---------------------------------------------------------------------------
# close() guard when _file was never set
# ---------------------------------------------------------------------------


def test_close_safe_when_partially_constructed(simple_model: nn.Module) -> None:
    obs = object.__new__(NeuroInquisitor)
    obs._closed = False  # type: ignore[attr-defined]
    obs._log_dir = Path("<none>")  # type: ignore[attr-defined]
    obs.close()  # must not raise


# ---------------------------------------------------------------------------
# Backend / format resolution
# ---------------------------------------------------------------------------


def test_unknown_backend_raises(simple_model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        NeuroInquisitor(simple_model, log_dir=tmp_path, backend="s4")


def test_unknown_format_raises(simple_model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        NeuroInquisitor(simple_model, log_dir=tmp_path, format="parquet")


def test_custom_backend_instance_accepted(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    from neuroinquisitor import LocalBackend

    backend = LocalBackend(tmp_path)
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, backend=backend)
    obs.close()


# ---------------------------------------------------------------------------
# NeuroInquisitor.load() classmethod
# ---------------------------------------------------------------------------


def test_load_classmethod_returns_collection(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    from neuroinquisitor import SnapshotCollection

    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    col = NeuroInquisitor.load(tmp_path)
    assert isinstance(col, SnapshotCollection)
    assert col.epochs == [0]


# ---------------------------------------------------------------------------
# NI-LLM-001: bfloat16 snapshot round-trip
# ---------------------------------------------------------------------------


def test_bfloat16_params_snapshot_no_error(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).to(torch.bfloat16)
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)  # must not raise
    obs.close()


def test_bfloat16_params_values_match_float32_cast(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).to(torch.bfloat16)
    original = {
        name: param.detach().cpu().to(torch.float32).numpy().copy()
        for name, param in model.named_parameters()
    }
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    for name, arr in original.items():
        np.testing.assert_array_equal(loaded[name], arr)


def test_bfloat16_buffers_snapshot_no_error(tmp_path: Path) -> None:
    from neuroinquisitor.schema import CapturePolicy

    model = nn.BatchNorm1d(4).to(torch.bfloat16)
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)  # must not raise
    obs.close()


def test_bfloat16_buffers_values_match_float32_cast(tmp_path: Path) -> None:
    from neuroinquisitor.schema import CapturePolicy

    model = nn.BatchNorm1d(4).to(torch.bfloat16)
    original_buffers = {
        name: buf.detach().cpu().to(torch.float32).numpy().copy()
        for name, buf in model.named_buffers()
        if buf is not None
    }
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    import json as _json
    from neuroinquisitor.formats.hdf5_format import HDF5Format

    snap_path = tmp_path / "epoch_0000.h5"
    loaded_buffers = HDF5Format().read_buffers(snap_path)
    for name, arr in original_buffers.items():
        np.testing.assert_array_equal(loaded_buffers[name], arr)


# ---------------------------------------------------------------------------
# NI-LLM-002: layer_filter
# ---------------------------------------------------------------------------


def _three_layer_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))


def test_layer_filter_restricts_captured_params(tmp_path: Path) -> None:
    model = _three_layer_model()
    target = "0.weight"
    obs = NeuroInquisitor(model, log_dir=tmp_path, layer_filter={target})
    obs.snapshot(epoch=0)
    obs.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    assert set(loaded.keys()) == {target}


def test_layer_filter_index_reflects_filtered_layers(tmp_path: Path) -> None:
    model = _three_layer_model()
    target = "0.weight"
    obs = NeuroInquisitor(model, log_dir=tmp_path, layer_filter={target})
    obs.snapshot(epoch=0)
    obs.close()

    data = json.loads((tmp_path / "index.json").read_text())
    snap = data["snapshots"][0]
    assert snap["layers"] == [target]


def test_layer_filter_capture_policy_stored_in_manifest(tmp_path: Path) -> None:
    model = _three_layer_model()
    obs = NeuroInquisitor(model, log_dir=tmp_path, layer_filter={"0.weight", "0.bias"})
    obs.snapshot(epoch=0)
    obs.close()

    data = json.loads((tmp_path / "index.json").read_text())
    policy = data["capture_policy"]
    assert policy["layer_filter"] is not None
    assert set(policy["layer_filter"]) == {"0.weight", "0.bias"}


def test_layer_filter_none_captures_all_params(tmp_path: Path) -> None:
    model = _three_layer_model()
    obs = NeuroInquisitor(model, log_dir=tmp_path, layer_filter=None)
    obs.snapshot(epoch=0)
    obs.close()

    loaded = NeuroInquisitor.load(tmp_path).by_epoch(0)
    expected = {name for name, _ in model.named_parameters()}
    assert set(loaded.keys()) == expected


# ---------------------------------------------------------------------------
# NI-LLM-006: streaming snapshot path
# ---------------------------------------------------------------------------


from neuroinquisitor import Backend  # noqa: E402


class _PathBackend(Backend):
    """Minimal Backend stand-in that implements ``write_from_path``.

    Records which calls were made so the test can assert the streaming path
    was selected rather than the legacy bytes path.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self.write_calls: list[str] = []
        self.path_calls: list[str] = []
        self.closed = False

    def write(self, key: str, data: bytes) -> None:
        self.write_calls.append(key)
        (self._root / key).parent.mkdir(parents=True, exist_ok=True)
        (self._root / key).write_bytes(data)

    def write_from_path(self, key: str, src) -> None:  # noqa: ANN001
        self.path_calls.append(key)
        from pathlib import Path as _P
        from shutil import move

        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        move(str(_P(src)), str(dest))

    def read_path(self, key: str) -> Path:
        return self._root / key

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()

    def delete(self, key: str) -> None:  # pragma: no cover
        (self._root / key).unlink(missing_ok=True)

    def close(self) -> None:
        self.closed = True


def test_snapshot_uses_streaming_path_when_backend_supports_it(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    backend = _PathBackend(tmp_path)
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, backend=backend)
    obs.snapshot(epoch=0)
    # index.json goes through .write(); snapshot file goes through write_from_path.
    assert "epoch_0000.h5" in backend.path_calls
    assert "epoch_0000.h5" not in backend.write_calls
    obs.close()


def test_close_calls_backend_close(simple_model: nn.Module, tmp_path: Path) -> None:
    backend = _PathBackend(tmp_path)
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path, backend=backend)
    obs.snapshot(epoch=0)
    obs.close()
    assert backend.closed is True


def test_local_backend_uses_legacy_bytes_path(
    simple_model: nn.Module, tmp_path: Path
) -> None:
    """LocalBackend has no write_from_path; behaviour must be unchanged."""
    obs = NeuroInquisitor(simple_model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    assert (tmp_path / "epoch_0000.h5").exists()
