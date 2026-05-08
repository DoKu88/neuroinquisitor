# Sprint 8: Built-in Analyzers & Caching

**Goal**: Ship the first built-in analyzers and caching layer, conforming to the interface contracts defined in Sprint 7.

**Prerequisite**: Sprint 7 must be complete. All analyzers here must use the base request/result pydantic models, the Parquet writer utility, and register into the analyzer registry defined there.

## Tasks

- [ ] `NI-GAMMA-001` Implement `trajectory_stats` analyzer.
  - Compute L2 distance from init, cosine similarity to init/final, update norm per step, and velocity/acceleration summaries.
  - Export results as a Parquet table artifact using the Sprint 7 writer utility.
- [ ] `NI-GAMMA-002` Implement `spectrum_rank` analyzer.
  - Compute singular values, effective/stable-rank summaries, spectral norm, and Frobenius norm.
  - Cache results as derived artifacts.
- [ ] `NI-GAMMA-003` Implement `projection_embed` analyzer.
  - Support PCA by default.
  - Support optional UMAP via an extra dependency.
  - Save coordinates as a derived Parquet table artifact and return plot-ready API data.
- [ ] `NI-GAMMA-004` Implement `similarity_compare` analyzer.
  - Start with CKA.
  - Add extension hook for SVCCA/PWCCA later.
  - Support epoch comparisons within a run and compatible cross-run comparisons.
- [ ] `NI-GAMMA-005` Implement `probe_linear` analyzer.
  - Train a linear probe on replayed activations.
  - Save metrics and coefficients with deterministic train/val split support.
- [ ] `NI-GAMMA-006` Add TracIn quick-start example using Captum.
  - Do not implement TracIn from scratch. Captum ships `TracInCP` with a complete, tested implementation.
  - Write a short runnable example (under 30 lines) showing how to feed NI checkpoints into Captum's `TracInCP`: load checkpoints via `SnapshotCollection.to_state_dict(epoch)`, reconstruct the model at each checkpoint, and pass to `TracInCP`.
  - Document which NI artifacts `TracInCP` needs (checkpoint state dicts, a loss function, the training dataloader) and where to find them.
  - Acceptance:
    - Example runs end-to-end on a toy classification model.
    - No NI-internal TracIn implementation is added.

- [ ] `NI-GAMMA-007` Validate `projection_embed` output compatibility with TensorBoard Projector and FiftyOne.
  - The purpose of this task is to confirm that `projection_embed` Parquet output is directly usable by two real visualization tools without writing any NI-specific glue.
  - **TensorBoard Projector**: from the `projection_embed` Parquet output, write a short example (under 20 lines) that produces the TSV files TensorBoard Projector expects (`tensors.tsv` + optional `metadata.tsv`) and launches the projector. No transformation of NI output types is permitted — only standard pandas/numpy calls.
  - **FiftyOne Embeddings**: from `projection_embed` or directly from a `ReplaySession` activation dict, write a short example (under 20 lines) that creates a FiftyOne dataset with embeddings attached and opens the FiftyOne App. Again, no NI-specific types in the example after the initial export call.
  - Both examples should include a note on which NI artifact to use as input and where to find it.
  - These are optional-dependency tests; skip gracefully when `tensorboard` or `fiftyone` are not installed.
  - Acceptance:
    - TensorBoard Projector launches and displays the embedding space from NI output.
    - FiftyOne dataset with embeddings is created and viewable from NI output.
    - Both examples are self-contained and under 20 lines.

## Cross-cutting requirements

All analyzers must:
- Accept a typed `pydantic` v2 request model derived from the base type defined in Sprint 7.
- Return a typed `pydantic` v2 result model derived from the base type defined in Sprint 7.
- Emit derived table artifacts using the Sprint 7 Parquet writer with required provenance columns.
- Register into the Sprint 7 analyzer registry with name, version, required_inputs, output_format, and description.
- Cache results deterministically; repeated calls with the same inputs must reuse existing derived artifacts.

## Testing

- Add analyzer contract tests for each built-in analyzer (pydantic input validation + output schema).
- Add regression tests for trajectory and spectrum metrics on toy models.
- Add caching tests that verify cache hits reuse derived artifacts deterministically.
- Add projection tests for PCA output shape and optional UMAP feature gating.
- Add similarity tests for within-run and compatible cross-run CKA comparisons.
- Add linear probe tests for deterministic split and reproducible metrics.
- Add TracIn tests validating helpful/harmful ranking output shape and metadata.
- Add registry membership tests confirming each analyzer appears in `ni plugins list`.
- TensorBoard Projector compatibility test: `projection_embed` output produces valid TSV files and launches the projector (skipped if `tensorboard` not installed).
- FiftyOne compatibility test: `projection_embed` or activation dict output populates a FiftyOne dataset with embeddings (skipped if `fiftyone` not installed).

## Definition of Done

- All six analyzers run via the public analysis API and emit versioned derived Parquet artifacts.
- All analyzers are registered in the Sprint 7 registry and discoverable via `ni plugins list`.
- Analyzer outputs are cacheable, reproducible, and carry provenance metadata.
- Plot-oriented analyzers return data in frontend-friendly formats.
- Optional features (UMAP, TracIn constraints) fail gracefully when unavailable or unsupported.
- All analyzer request/result contracts use the shared pydantic v2 base models from Sprint 7.
- Integration tests cover at least one complete analysis flow from run selection to artifact retrieval.
- TensorBoard Projector and FiftyOne can visualize NI projection output with no NI-specific glue code.
