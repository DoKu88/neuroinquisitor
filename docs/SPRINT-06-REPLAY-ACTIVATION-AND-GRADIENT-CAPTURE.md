# Sprint 6: Replay, Activation & Gradient Capture

**Goal**: Introduce replay-based activation and gradient extraction.

## Tasks
- [ ] `NI-BETA-001` Implement `ReplaySession`.
  - Create `src/neuroinquisitor/replay.py`.
  - Support checkpoint selector, model factory/loader callback, dataloader/iterable, module selectors, and capture kinds (`activations`, `gradients`, `logits`).
  - Define replay request/response contracts with `pydantic` v2 models.
  - **Output contract**: activations and gradients are returned as `Dict[str, torch.Tensor]` keyed by module name — identical layout to what a manual `register_forward_hook` / `register_full_backward_hook` would produce. No NI wrapper types in the return value. Logits are returned as a plain `torch.Tensor`.
- [ ] `NI-BETA-002` Add hook-based activation capture.
  - Support forward hooks for selected modules.
  - Support reduction modes: raw batch outputs, mean over batch, and pooled statistics.
- [ ] `NI-BETA-003` Add gradient capture for selected modules.
  - Support backward hooks or explicit autograd collection.
  - Support per-example and aggregated modes.
- [ ] `NI-BETA-004` Add dataset slice abstraction.
  - Support selectors for first N, random N with seed, class-balanced N (when labels exist), and explicit indices.
  - Implement slice selectors as discriminated `pydantic` models.
  - Persist slice metadata in derived artifacts.

## Testing
- Add replay integration tests for a small MLP and CNN example.
- Add validation tests for invalid module names with clear error messages.
- Add activation capture tests that verify each reduction mode output shape.
- Add gradient capture tests for classification examples, including shape correctness checks.
- Add dataset slice tests for deterministic seeded sampling and metadata persistence.
- Add artifact-size reporting tests for replay outputs.
- Add `pydantic` validation tests for replay config and dataset-slice selector contracts.

## Definition of Done
- `ReplaySession` runs end-to-end for at least one MLP and one CNN path.
- Activation and gradient capture both support selective modules and configurable reduction/aggregation modes.
- Dataset slice strategy is recorded in derived artifact metadata.
- Replay and slice configuration contracts are enforced via `pydantic` v2 validation.
- Failures for invalid selectors and unsupported capture paths are explicit and actionable.
- Tests verify correctness of captured tensor shapes and replay metadata provenance.
