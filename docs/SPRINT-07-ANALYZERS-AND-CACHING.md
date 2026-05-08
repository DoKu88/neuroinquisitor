# Sprint 7: Built-in Analyzers & Caching

**Goal**: Ship the first built-in analyzers and caching layer.

## Tasks
- [ ] `NI-GAMMA-001` Implement `trajectory_stats` analyzer.
  - Compute L2 distance from init, cosine similarity to init/final, update norm per step, and velocity/acceleration summaries.
  - Export results as a table artifact.
- [ ] `NI-GAMMA-002` Implement `spectrum_rank` analyzer.
  - Compute singular values, effective/stable-rank summaries, spectral norm, and Frobenius norm.
  - Cache results as derived artifacts.
- [ ] `NI-GAMMA-003` Implement `projection_embed` analyzer.
  - Support PCA by default.
  - Support optional UMAP via an extra dependency.
  - Save coordinates as a derived table artifact and return plot-ready API data.
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
- [ ] Cross-cutting: define analyzer I/O contracts.
  - Standardize analyzer request/result payloads as `pydantic` v2 models.
  - Require versioned, schema-validated derived artifact metadata for each analyzer.

## Testing
- Add analyzer contract tests for each built-in analyzer (`pydantic` input validation + output schema).
- Add regression tests for trajectory and spectrum metrics on toy models.
- Add caching tests that verify cache hits reuse derived artifacts deterministically.
- Add projection tests for PCA output shape and optional UMAP feature gating.
- Add similarity tests for within-run and compatible cross-run CKA comparisons.
- Add linear probe tests for deterministic split and reproducible metrics.
- Add TracIn tests validating helpful/harmful ranking output shape and metadata.

## Definition of Done
- All six analyzers run via the public analysis API and emit versioned derived artifacts.
- Analyzer outputs are cacheable, reproducible, and carry provenance metadata.
- Plot-oriented analyzers return data in frontend-friendly formats.
- Optional features (UMAP, TracIn constraints) fail gracefully when unavailable or unsupported.
- Analyzer request/result contracts are validated through shared `pydantic` v2 models.
- Integration tests cover at least one complete analysis flow from run selection to artifact retrieval.
