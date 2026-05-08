# Sprint 10: Web GUI Foundation

**Goal**: Release the first web GUI.

**Prerequisite**: Sprint 9 must be complete. The frontend panels read from Parquet artifacts and call analyzers via the registry; both must exist before this sprint begins.

## Tasks
- [ ] `NI-EPSILON-001` Build a FastAPI backend.
  - Add endpoints to list runs, fetch manifest, list layers, list epochs/steps, run analyzer jobs, poll job status, and fetch analysis results.
  - Use shared `pydantic` v2 request/response models for all API endpoints.
  - Ensure API docs are available.
- [ ] `NI-EPSILON-002` Add background job execution.
  - Introduce a simple local worker for long-running analysis jobs.
  - Persist job state in a lightweight store.
- [ ] `NI-EPSILON-003` Build first frontend panels.
  - Implement the following panels, each reading from a specific derived artifact; trigger the corresponding analyzer job automatically if the artifact is missing:
    - **Run browser**: reads run manifest directly — no analyzer required.
    - **Layer/epoch selector**: reads manifest snapshot refs — no analyzer required.
    - **Heatmap panel**: reads raw parameter tensors from HDF5 for the selected layer and epoch.
    - **Spectrum panel**: reads from the `spectrum_rank` derived Parquet artifact; triggers `spectrum_rank` analyzer if absent.
    - **Projection panel**: reads from the `projection_embed` derived table artifact (PCA/UMAP coordinates); triggers `projection_embed` analyzer if absent.
    - **Similarity matrix panel**: reads from the `similarity_compare` derived Parquet artifact; triggers `similarity_compare` analyzer if absent.
  - Add compare mode for two selected epochs; all panels update to show both epochs side by side.
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
