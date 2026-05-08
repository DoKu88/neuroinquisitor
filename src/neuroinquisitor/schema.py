"""Schema models for NeuroInquisitor manifests and capture policies (pydantic v2)."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"


class CapturePolicy(BaseModel):
    """Declares which artifacts are captured at snapshot time.

    Persisted into the manifest so replay code can reconstruct what was
    stored without inspecting the snapshot files themselves.
    """

    model_config = ConfigDict(extra="forbid")

    capture_parameters: bool = True
    capture_buffers: bool = False
    capture_optimizer: bool = False
    replay_activations: bool = False
    replay_gradients: bool = False


class RunMetadata(BaseModel):
    """Optional provenance metadata attached to a training run.

    All fields are optional so that missing information does not prevent a
    manifest from loading.
    """

    git_commit: str | None = None
    training_config: dict[str, Any] | None = None
    optimizer_class: str | None = None
    dtype: str | None = None
    device: str | None = None
    model_class: str | None = None


class LayerMetadata(BaseModel):
    """Metadata about a single parameter or buffer tensor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str


class DerivedArtifactRef(BaseModel):
    """Reference to a cached derived artifact produced by an analyzer."""

    model_config = ConfigDict(extra="forbid")

    key: str
    kind: str
    analyzer: str
    created_at: str | None = None


class SnapshotRef(BaseModel):
    """Schema-level representation of one snapshot entry in the manifest."""

    model_config = ConfigDict(extra="forbid")

    epoch: int | None = None
    step: int | None = None
    file_key: str
    layers: list[str] = Field(default_factory=list)
    buffers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    capture_policy: CapturePolicy | None = None
    derived: list[DerivedArtifactRef] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Top-level manifest for a NeuroInquisitor run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_metadata: RunMetadata | None = None
    capture_policy: CapturePolicy | None = None
    snapshots: list[SnapshotRef] = Field(default_factory=list)
    derived_artifacts: list[DerivedArtifactRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _migrate_v0_to_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Promote a legacy (schema_version absent) manifest to v1 shape."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_metadata": None,
        "capture_policy": None,
        "snapshots": raw.get("snapshots", []),
        "derived_artifacts": [],
    }


_MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "0": _migrate_v0_to_v1,
}


def load_manifest(raw: dict[str, Any]) -> RunManifest:
    """Load a manifest dict, applying any needed version migrations.

    Raises ``ValueError`` if the schema version is unrecognised and no
    migration path exists.
    """
    version = raw.get("schema_version", "0")
    if version != SCHEMA_VERSION:
        if version not in _MIGRATIONS:
            raise ValueError(
                f"Unrecognised manifest schema version {version!r}. "
                f"Expected {SCHEMA_VERSION!r} or a migratable version."
            )
        raw = _MIGRATIONS[version](raw)
    return RunManifest.model_validate(raw)
