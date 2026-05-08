"""Tests for the pydantic v2 manifest schema (NI-ALPHA-001 through NI-ALPHA-004)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neuroinquisitor import CapturePolicy, NeuroInquisitor, RunManifest, RunMetadata
from neuroinquisitor.schema import (
    SCHEMA_VERSION,
    DerivedArtifactRef,
    LayerMetadata,
    SnapshotRef,
    load_manifest,
)


# ---------------------------------------------------------------------------
# CapturePolicy round-trip
# ---------------------------------------------------------------------------


def test_capture_policy_defaults() -> None:
    p = CapturePolicy()
    assert p.capture_parameters is True
    assert p.capture_buffers is False
    assert p.capture_optimizer is False
    assert p.replay_activations is False
    assert p.replay_gradients is False


def test_capture_policy_round_trip() -> None:
    p = CapturePolicy(capture_buffers=True, capture_optimizer=True)
    restored = CapturePolicy.model_validate(p.model_dump())
    assert restored == p


def test_capture_policy_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        CapturePolicy.model_validate({"capture_parameters": True, "unknown_field": 99})


# ---------------------------------------------------------------------------
# RunMetadata round-trip and optional fields
# ---------------------------------------------------------------------------


def test_run_metadata_all_none() -> None:
    m = RunMetadata()
    assert m.git_commit is None
    assert m.model_class is None


def test_run_metadata_round_trip() -> None:
    m = RunMetadata(
        git_commit="abc123",
        optimizer_class="SGD",
        dtype="torch.float32",
        device="cpu",
        model_class="torch.nn.modules.linear.Linear",
        training_config={"lr": 0.01, "epochs": 10},
    )
    restored = RunMetadata.model_validate(m.model_dump())
    assert restored == m


def test_run_metadata_partial_fields_load() -> None:
    partial = {"git_commit": "abc123"}
    m = RunMetadata.model_validate(partial)
    assert m.git_commit == "abc123"
    assert m.model_class is None


# ---------------------------------------------------------------------------
# LayerMetadata
# ---------------------------------------------------------------------------


def test_layer_metadata_round_trip() -> None:
    lm = LayerMetadata(name="fc1.weight", kind="parameter")
    restored = LayerMetadata.model_validate(lm.model_dump())
    assert restored == lm


def test_layer_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        LayerMetadata.model_validate({"name": "x", "kind": "parameter", "extra": 1})


# ---------------------------------------------------------------------------
# DerivedArtifactRef round-trip
# ---------------------------------------------------------------------------


def test_derived_artifact_ref_round_trip() -> None:
    ref = DerivedArtifactRef(key="run1/spectrum", kind="table", analyzer="spectrum_rank")
    restored = DerivedArtifactRef.model_validate(ref.model_dump())
    assert restored == ref


# ---------------------------------------------------------------------------
# SnapshotRef round-trip
# ---------------------------------------------------------------------------


def test_snapshot_ref_round_trip() -> None:
    ref = SnapshotRef(
        epoch=3,
        step=150,
        file_key="epoch_0003_step_000150.h5",
        layers=["fc1.weight", "fc1.bias"],
        buffers=["running_mean", "running_var"],
        metadata={"loss": 0.42},
        capture_policy=CapturePolicy(capture_buffers=True),
    )
    restored = SnapshotRef.model_validate(ref.model_dump())
    assert restored == ref


def test_snapshot_ref_defaults() -> None:
    ref = SnapshotRef(file_key="step_000001.h5")
    assert ref.layers == []
    assert ref.buffers == []
    assert ref.metadata == {}
    assert ref.capture_policy is None
    assert ref.derived == []


# ---------------------------------------------------------------------------
# RunManifest round-trip
# ---------------------------------------------------------------------------


def test_run_manifest_defaults() -> None:
    m = RunManifest()
    assert m.schema_version == SCHEMA_VERSION
    assert m.snapshots == []
    assert m.derived_artifacts == []


def test_run_manifest_round_trip() -> None:
    m = RunManifest(
        run_metadata=RunMetadata(git_commit="abc"),
        capture_policy=CapturePolicy(capture_buffers=True),
        snapshots=[
            SnapshotRef(epoch=0, file_key="epoch_0000.h5", layers=["w"]),
        ],
    )
    raw = json.loads(m.model_dump_json())
    restored = RunManifest.model_validate(raw)
    assert restored == m


def test_run_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        RunManifest.model_validate(
            {"schema_version": SCHEMA_VERSION, "unexpected_key": True}
        )


# ---------------------------------------------------------------------------
# load_manifest — migration (NI-ALPHA-001 acceptance: existing runs readable)
# ---------------------------------------------------------------------------


def test_load_manifest_migrates_legacy_format() -> None:
    legacy = {
        "snapshots": [
            {
                "epoch": 0,
                "step": None,
                "file_key": "epoch_0000.h5",
                "layers": ["fc1.weight"],
                "metadata": {"loss": 0.9},
            }
        ]
    }
    manifest = load_manifest(legacy)
    assert manifest.schema_version == SCHEMA_VERSION
    assert len(manifest.snapshots) == 1
    assert manifest.snapshots[0].epoch == 0
    assert manifest.run_metadata is None


def test_load_manifest_current_version_passthrough() -> None:
    m = RunManifest(run_metadata=RunMetadata(git_commit="xyz"))
    raw = json.loads(m.model_dump_json())
    restored = load_manifest(raw)
    assert restored.run_metadata is not None
    assert restored.run_metadata.git_commit == "xyz"


def test_load_manifest_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognised"):
        load_manifest({"schema_version": "99"})


# ---------------------------------------------------------------------------
# NI-ALPHA-001: new runs write schema version
# ---------------------------------------------------------------------------


def test_new_run_writes_schema_version(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    assert raw["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# NI-ALPHA-002: richer run metadata stored and loadable
# ---------------------------------------------------------------------------


def test_run_metadata_auto_detected_on_new_run(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    meta = raw.get("run_metadata", {})
    assert meta.get("model_class") is not None
    assert "Linear" in meta["model_class"]


def test_custom_run_metadata_persisted(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    run_meta = RunMetadata(
        git_commit="deadbeef",
        optimizer_class="Adam",
        training_config={"lr": 0.001},
    )
    obs = NeuroInquisitor(model, log_dir=tmp_path, run_metadata=run_meta)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    meta = raw["run_metadata"]
    assert meta["git_commit"] == "deadbeef"
    assert meta["optimizer_class"] == "Adam"
    assert meta["training_config"]["lr"] == 0.001


def test_metadata_missing_fields_do_not_break_load(tmp_path: Path) -> None:
    minimal_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_metadata": {"git_commit": "abc"},
        "snapshots": [],
        "capture_policy": None,
        "derived_artifacts": [],
    }
    (tmp_path / "index.json").write_text(json.dumps(minimal_manifest))
    obs = NeuroInquisitor(nn.Linear(2, 1), log_dir=tmp_path, create_new=False)
    obs.snapshot(epoch=0)
    obs.close()
    col = NeuroInquisitor.load(tmp_path)
    assert col.epochs == [0]


# ---------------------------------------------------------------------------
# NI-ALPHA-003: buffer capture
# ---------------------------------------------------------------------------


class BNModel(nn.Module):
    """Simple model with BatchNorm buffers (running_mean, running_var, num_batches_tracked)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


def test_buffer_capture_off_by_default(tmp_path: Path) -> None:
    model = BNModel()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    snap = raw["snapshots"][0]
    assert snap["buffers"] == []


def test_buffer_capture_stores_buffers_in_manifest(tmp_path: Path) -> None:
    model = BNModel()
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    snap = raw["snapshots"][0]
    assert len(snap["buffers"]) > 0
    assert "bn.running_mean" in snap["buffers"]


def test_buffer_capture_distinguishable_from_parameters(tmp_path: Path) -> None:
    model = BNModel()
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    snap = raw["snapshots"][0]
    param_names = set(snap["layers"])
    buffer_names = set(snap["buffers"])
    assert param_names.isdisjoint(buffer_names), "Parameters and buffers must not overlap"


def test_buffer_capture_reads_back(tmp_path: Path) -> None:
    import numpy as np

    model = BNModel()
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    from neuroinquisitor.backends.local import LocalBackend
    from neuroinquisitor.formats.hdf5_format import HDF5Format
    from neuroinquisitor.index.json_index import JSONIndex

    backend = LocalBackend(tmp_path)
    fmt = HDF5Format()
    index = JSONIndex.load(backend)
    entry = index.all()[0]
    path = backend.read_path(entry.file_key)
    buffers = fmt.read_buffers(path)
    assert "bn.running_mean" in buffers
    assert isinstance(buffers["bn.running_mean"], np.ndarray)


def test_parameter_read_unaffected_by_buffer_capture(tmp_path: Path) -> None:
    import numpy as np

    model = BNModel()
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    col = NeuroInquisitor.load(tmp_path)
    params = col.by_epoch(0)
    assert all(not k.startswith("buffers") for k in params)
    assert "fc.weight" in params
    assert isinstance(params["fc.weight"], np.ndarray)


def test_buffer_capture_no_buffers_model(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    snap = raw["snapshots"][0]
    assert snap["buffers"] == []


# ---------------------------------------------------------------------------
# NI-ALPHA-004: capture policy serialised and referenced per snapshot
# ---------------------------------------------------------------------------


def test_capture_policy_persisted_in_manifest(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    policy = CapturePolicy(capture_buffers=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    assert raw["capture_policy"]["capture_buffers"] is True


def test_snapshot_capture_policy_referenced_per_snapshot(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    policy = CapturePolicy(capture_buffers=False)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    raw = json.loads((tmp_path / "index.json").read_text())
    snap_policy = raw["snapshots"][0]["capture_policy"]
    assert snap_policy["capture_parameters"] is True
    assert snap_policy["capture_buffers"] is False


def test_capture_policy_round_trips_through_index(tmp_path: Path) -> None:
    from neuroinquisitor.backends.local import LocalBackend
    from neuroinquisitor.index.json_index import JSONIndex

    model = nn.Linear(4, 2)
    policy = CapturePolicy(capture_buffers=True, capture_optimizer=True)
    obs = NeuroInquisitor(model, log_dir=tmp_path, capture_policy=policy)
    obs.snapshot(epoch=0)
    obs.close()

    index = JSONIndex.load(LocalBackend(tmp_path))
    entry = index.all()[0]
    assert entry.capture_policy is not None
    assert entry.capture_policy.capture_buffers is True
    assert entry.capture_policy.capture_optimizer is True


# ---------------------------------------------------------------------------
# Backward compatibility: existing runs without schema_version load correctly
# ---------------------------------------------------------------------------


def test_legacy_run_remains_readable(tmp_path: Path) -> None:
    import numpy as np

    legacy_index = {
        "snapshots": [
            {
                "epoch": 0,
                "step": None,
                "file_key": "epoch_0000.h5",
                "layers": ["weight", "bias"],
                "metadata": {"loss": 0.5},
            }
        ]
    }
    model = nn.Linear(2, 1)
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()

    (tmp_path / "index.json").write_text(json.dumps(legacy_index))

    col = NeuroInquisitor.load(tmp_path)
    assert col.epochs == [0]
