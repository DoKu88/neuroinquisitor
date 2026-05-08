# Sprint 9: Web GUI Foundation

**Goal**: Release the first web GUI.

## Tasks
- [ ] `NI-EPSILON-001` Build a FastAPI backend.
  - Add endpoints to list runs, fetch manifest, list layers, list epochs/steps, run analyzer jobs, poll job status, and fetch analysis results.
  - Use shared `pydantic` v2 request/response models for all API endpoints.
  - Ensure API docs are available.
- [ ] `NI-EPSILON-002` Add background job execution.
  - Introduce a simple local worker for long-running analysis jobs.
  - Persist job state in a lightweight store.
- [ ] `NI-EPSILON-003` Build first frontend panels.
  - Implement run browser, layer/epoch selector, heatmap panel, spectrum panel, projection panel, and similarity matrix panel.
  - Add compare mode for two selected epochs.
  - Link projection selection to detail panel updates.
- [ ] `NI-EPSILON-004` Add provenance and export.
  - Show analyzer name/version, input selectors, dependency versions, and cache key for each result.
  - Support exporting both result metadata and data.

## Testing
- Add API tests for all baseline endpoints and OpenAPI documentation availability.
- Add job lifecycle tests: submit, run, poll, complete/fail paths.
- Add worker tests ensuring long-running analysis does not block request handling.
- Add frontend integration tests for panel rendering, compare mode, and selection linking.
- Add export tests to verify provenance fields and downloadable artifact outputs.
- Add API validation tests for malformed payloads and schema mismatch errors.

## Definition of Done
- Backend endpoints serve run browsing and analysis workflows end-to-end.
- Background jobs execute reliably and report stable status transitions.
- Core panels support navigation from run -> layer/epoch -> analysis view.
- Compare mode works for two epochs and updates all linked views.
- Result pages always include provenance metadata and support export.
- API contracts are generated from and enforced by shared `pydantic` v2 schemas.
