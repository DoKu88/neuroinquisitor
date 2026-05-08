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
- [ ] `NI-GAMMA-006` Implement `influence_tracin` analyzer.
  - Add first practical checkpoint-based approximation for supported classification/loss shapes.
  - Return top helpful/harmful examples for a query item.
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
