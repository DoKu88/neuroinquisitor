# Sprint 5: Artifact Schema & Capture Policy

**Goal**: Stabilize the artifact model and prepare the codebase for replay and plugins.

## Tasks
- [ ] `NI-ALPHA-001` Add a versioned manifest schema.
  - Create `src/neuroinquisitor/schema.py`.
  - Define `pydantic` v2 models for run metadata, snapshot refs, layer metadata, and derived artifact refs.
  - Include manifest versioning and migration hooks.
  - Configure strict model behavior (`extra="forbid"` where appropriate) to fail fast on malformed manifests.
- [ ] `NI-ALPHA-002` Add richer run metadata capture.
  - Extend snapshot metadata support to store git commit (if available), training config, optimizer class name, dtype/device, and model class path.
  - Keep metadata optional and backward-compatible.
- [ ] `NI-ALPHA-003` Add optional buffer capture.
  - Add `capture_buffers: bool = False` to capture config.
  - Store buffers under a separate namespace from parameters.
- [ ] `NI-ALPHA-004` Add capture policy objects.
  - Create a serializable `CapturePolicy` `pydantic` model for parameter capture, buffer capture, optimizer capture, and replay capture requests.
  - Reference policies from manifest entries.

## Testing
- Add schema round-trip tests that verify `pydantic` `model_validate`/`model_dump` for manifests.
- Add migration compatibility tests to confirm existing runs can still be read.
- Add tests for optional metadata fields to ensure missing fields do not break load paths.
- Add tests for buffer capture toggling and namespace separation from parameters.
- Add tests that `CapturePolicy` objects serialize and can be rehydrated from manifest references.
- Add strict-validation tests confirming unknown/invalid fields raise clear validation errors.

## Definition of Done
- New runs always record a manifest schema version.
- Existing runs remain readable without manual migration.
- Buffer capture is optional, explicit, and distinguishable from parameter capture.
- Capture policies are persisted and referenced by manifest entries.
- All manifest and capture-policy contracts are implemented as `pydantic` v2 models with deterministic serialization.
- Integration coverage confirms schema + metadata + policy behavior in realistic run flows.
