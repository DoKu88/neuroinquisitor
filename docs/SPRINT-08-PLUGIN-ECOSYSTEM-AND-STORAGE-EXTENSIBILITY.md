# Sprint 8: Simple Extensibility & Storage

**Goal**: Keep extensibility simple for early adoption while improving storage options.

## Tasks
- [ ] `NI-DELTA-001` Add in-repo analyzer/adapters registry.
  - Register built-in analyzers via a simple internal registry (no external entry-point loading yet).
  - Define analyzer metadata (`name`, `version`, `capabilities`) using `pydantic` v2 models.
  - Keep registration code structured so external plugins can be added later without API breaks.
- [ ] `NI-DELTA-002` Define explicit adapter interface.
  - Add a standardized adapter lifecycle: `validate_request -> run -> materialize_artifacts`.
  - Add capability flags (for example: activations, gradients, per-example outputs, transformer-specific).
  - Define adapter request/result payloads and panel specs as `pydantic` v2 models.
- [ ] `NI-DELTA-003` Add lightweight extension points for internal use.
  - Add internal hook points for artifact materialization and panel spec generation.
  - Keep interfaces public and documented, but defer third-party plugin loading to a future sprint.
- [ ] `NI-DELTA-004` Add derived table storage with Parquet.
  - Persist long-form metrics/tables as Parquet.
  - Keep tables queryable without loading raw tensors.
- [ ] `NI-DELTA-005` Add optional tensor-derived storage extras.
  - Add optional extras for Zarr and Safetensors backends.
  - Document backend tradeoffs and recommended usage.
- [ ] `NI-DELTA-006` Add CLI surface.
  - Add commands: `ni runs list`, `ni analyze run`, `ni plugins list`, `ni manifest show`.
  - Ensure CLI calls the same public APIs used by Python callers.
  - For now, `ni plugins list` should list built-in analyzers/adapters and clearly report that external plugins are not yet enabled.
- [ ] `NI-DELTA-007` Define a plugin-readiness gate.
  - Document objective criteria for enabling external plugin loading (for example: number of adapters, contributor demand, stability period).
  - Record a migration plan from internal registry to entry-point discovery.

## Testing
- Add registry tests for built-in analyzer/adapter registration and lookup.
- Add adapter lifecycle tests for `validate_request`, `run`, and `materialize_artifacts`.
- Add capability-flag tests to ensure unsupported requests fail fast with actionable errors.
- Add hook execution tests for internal materialization/panel extension points.
- Add Parquet read/write tests for derived table artifacts.
- Add backend compatibility tests for optional Zarr/Safetensors paths (feature-gated).
- Add CLI command tests that validate output and API parity.
- Add schema tests confirming adapter payloads/panel specs fail fast on invalid fields.

## Definition of Done
- Built-in analyzers/adapters are discoverable through a stable internal registry.
- Adapter contracts and lifecycle are stable, documented, and covered by tests.
- Derived table artifacts are persisted and queryable via Parquet.
- Optional tensor backends are available through extras and degrade gracefully when not installed.
- CLI commands are functional and rely on public APIs rather than internal shortcuts.
- Extension contracts are formally validated with shared `pydantic` v2 models.
- External plugin loading is explicitly deferred, with documented enablement criteria and migration path.
