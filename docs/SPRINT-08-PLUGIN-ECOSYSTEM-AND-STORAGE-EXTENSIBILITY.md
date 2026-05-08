# Sprint 8: Plugin Ecosystem & Storage Extensibility

**Goal**: Open the ecosystem through plugins and richer storage options.

## Tasks
- [ ] `NI-DELTA-001` Add analyzer registry with Python entry points.
  - Define an entry-point group (for example `neuroinquisitor.analyzers`).
  - Auto-discover analyzers at runtime.
  - Provide a demo external package registration path.
- [ ] `NI-DELTA-002` Add a hook-based plugin manager.
  - Introduce hook specs for analyzer registration, artifact materialization, and panel spec generation.
  - Define plugin-facing hook payloads and panel specs as `pydantic` v2 models.
  - Ensure extension works without monkeypatching core code.
- [ ] `NI-DELTA-003` Add derived table storage with Parquet.
  - Persist long-form metrics/tables as Parquet.
  - Keep tables queryable without loading raw tensors.
- [ ] `NI-DELTA-004` Add optional tensor-derived storage extras.
  - Add optional extras for Zarr and Safetensors backends.
  - Document backend tradeoffs and recommended usage.
- [ ] `NI-DELTA-005` Add CLI surface.
  - Add commands: `ni runs list`, `ni analyze run`, `ni plugins list`, `ni manifest show`.
  - Ensure CLI calls the same public APIs used by Python callers.

## Testing
- Add plugin discovery tests using synthetic entry points.
- Add integration tests with a demo external analyzer package.
- Add hook execution tests for registration/materialization/panel spec extension points.
- Add Parquet read/write tests for derived table artifacts.
- Add backend compatibility tests for optional Zarr/Safetensors paths (feature-gated).
- Add CLI command tests that validate output and API parity.
- Add schema tests confirming plugin payloads/panel specs fail fast on invalid fields.

## Definition of Done
- Third-party analyzers can be installed and discovered without core code changes.
- Hook contracts are stable, documented, and covered by tests.
- Derived table artifacts are persisted and queryable via Parquet.
- Optional tensor backends are available through extras and degrade gracefully when not installed.
- CLI commands are functional and rely on public APIs rather than internal shortcuts.
- Plugin extension contracts are formally validated with shared `pydantic` v2 models.
