# Sprint 7: Standard Formats & Analyzer Interface

**Goal**: Define the data contracts and analyzer interface that all built-in and third-party analyzers will be built on top of. Nothing in Sprint 8 (analyzers) should be written until these contracts are locked.

The guiding constraint: **NI is a data layer, not an analysis framework.** Every output it produces should be expressible as a standard Python/PyTorch/NumPy/Pandas object. No custom wrapper objects in public return values.

## Core design principle

The three universal handshakes are:

| Artifact type | Standard format | Who can consume it |
|---|---|---|
| Weights / parameters at epoch N | `Dict[str, torch.Tensor]` (state dict) or `Dict[str, np.ndarray]` | Anything that loads a PyTorch model, any numerical library |
| Activations / gradients from replay | `Dict[str, torch.Tensor]` keyed by module name | Same as PyTorch forward/backward hooks — any tool expecting hook outputs |
| Derived metrics and tables | `pd.DataFrame` / Parquet on disk | pandas, polars, any data tool; readable without importing NI |

If these are solid, a developer can pipe NI outputs into Captum, TransformerLens, sklearn, scipy, their own package, or a Jupyter notebook without glue code.

## Tasks

- [ ] `NI-DELTA-001` Standardize tensor export surface.
  - Add `SnapshotCollection.to_state_dict(epoch) -> Dict[str, Tensor]`: returns a standard PyTorch state dict for a given epoch, loadable with `model.load_state_dict()` directly.
  - Add `SnapshotCollection.to_numpy(epoch, layers=None) -> Dict[str, np.ndarray]`: returns parameter tensors as numpy arrays keyed by layer name. No NI types in the return value.
  - Confirm `ReplaySession` activation and gradient outputs are plain `Dict[str, Tensor]` keyed by module name (same layout as PyTorch forward hooks). No wrapper objects.
  - Add `ReplaySession.to_numpy() -> Dict[str, np.ndarray]` for cases where numpy is preferred.
  - Acceptance:
    - A developer can call `model.load_state_dict(col.to_state_dict(epoch=5))` with zero NI knowledge beyond that one call.
    - Activation dict from replay matches the exact key/shape layout a manual `register_forward_hook` would produce.
    - Tests confirm no NI-internal types appear in any public return value.

- [ ] `NI-DELTA-002` Add internal analyzer registry.
  - Define the analyzer interface contract: each analyzer is a callable that takes a typed `pydantic` v2 request model and returns a typed `pydantic` v2 result model.
  - Register built-in analyzers (to be implemented in Sprint 8) via a simple dict-based internal registry.
  - Each entry carries: `name`, `version`, `required_inputs` (e.g. `["weights"]`, `["activations", "labels"]`), `output_format` (tensor / table / both), and a short `description`.
  - Keep registration code flat and explicit — no metaclasses, no magic import side-effects.
  - Keep the interface stable and public so a developer can register their own analyzer with two lines of code, but do not build external plugin loading yet.
  - Define the shared `pydantic` v2 base models for analyzer request and result payloads. All built-in analyzers in Sprint 8 must use these base types.
  - Acceptance:
    - `ni plugins list` (from Sprint 9 CLI) reads from this registry.
    - Adding a new analyzer requires only appending to the registry dict and implementing the interface.
    - Request and result base models are importable from a stable public path.

- [ ] `NI-DELTA-003` Persist derived tables as Parquet.
  - Define the standard Parquet output contract for all tabular analyzer outputs (trajectory stats, probe scores, similarity matrices, etc.).
  - Files must be directly readable with `pd.read_parquet()` — no NI import required to open them.
  - Define required provenance columns for every derived table: `run_id`, `epoch`, `layer`, `analyzer_name`, `analyzer_version`.
  - Implement the writer utility that all Sprint 8 analyzers will use to emit Parquet artifacts.
  - Acceptance:
    - A developer can open a result with `pd.read_parquet("runs/exp_a/trajectory_stats.parquet")` and get a usable DataFrame.
    - Parquet files validate against the provenance column requirement.
    - Writer utility is importable and usable by Sprint 8 analyzers without duplication.

- [ ] `NI-DELTA-007` Validate tensor output compatibility with Captum, TorchLens, and TransformerLens.
  - The purpose of this task is to prove that the format contracts from NI-DELTA-001 are genuinely useful — not just aesthetically correct — by feeding NI outputs directly into three real consumers without any glue code.
  - **Captum**: pass `ReplaySession` activation dict to `captum.attr.LayerActivation` and at least one gradient-based attribution method (e.g. `IntegratedGradients`). Verify no type conversion is needed.
  - **TorchLens**: run TorchLens on the same toy model and the same input batch; assert that the layer output dict keys and tensor shapes match what `ReplaySession` produces. This confirms NI replay is a drop-in for TorchLens extraction.
  - **TransformerLens**: for a small transformer toy model, verify that NI `ReplaySession` activation dict keys and shapes are compatible with what TransformerLens hook output would produce for the same modules. Flag any key-naming conventions that diverge.
  - All three checks must use the public NI API only — no reaching into internals.
  - These are optional-dependency tests; skip gracefully with a clear message when Captum, TorchLens, or TransformerLens are not installed.
  - Acceptance:
    - Each of the three tools receives NI output and produces a result without any intermediate transformation.
    - Tests are marked with appropriate skip conditions for missing extras.
    - Any naming or shape divergence found during implementation is resolved before Sprint 8 begins.

## Testing

- Export-format tests: confirm `to_state_dict`, `to_numpy`, and `ReplaySession` dict outputs contain only standard types (Tensor, ndarray, Python scalars).
- Round-trip test: `model.load_state_dict(col.to_state_dict(epoch=N))` produces identical forward-pass outputs to the original model at that epoch.
- Activation key test: replay activation dict keys match the module names passed to the session, with the same shape a manual hook would produce.
- Registry tests: registry is queryable, metadata fields are present and typed correctly, duplicate registration raises an error.
- Parquet writer tests: output files open with `pd.read_parquet` and contain all required provenance columns.
- Analyzer interface contract tests: a minimal stub analyzer that implements the base request/result types passes registry validation.
- Captum compatibility test: `ReplaySession` activation and gradient dicts feed into Captum attribution methods without type conversion (skipped if Captum not installed).
- TorchLens compatibility test: NI replay output keys and shapes match TorchLens extraction for the same model and inputs (skipped if TorchLens not installed).
- TransformerLens compatibility test: NI replay output structure is compatible with TransformerLens hook output for a small transformer (skipped if TransformerLens not installed).

## Definition of Done

- `SnapshotCollection.to_state_dict` and `to_numpy` return only standard types; no NI wrapper types in any public return value.
- `ReplaySession` activation and gradient outputs are structurally identical to what `register_forward_hook` / `register_full_backward_hook` would produce.
- Analyzer interface contract (base request/result models + registry) is defined, importable, and documented.
- Parquet provenance schema is defined and the writer utility is in place.
- Captum, TorchLens, and TransformerLens compatibility is verified end-to-end with no glue code required.
- All Sprint 8 analyzer development can proceed without reopening these contracts.
