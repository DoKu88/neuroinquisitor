# Sprint 7: Standard Formats

**Goal**: Define and lock the data contracts for all NI outputs. Nothing in Sprint 8 (analyzers) should be written until these formats are confirmed.

The guiding constraint: **NI is a data layer, not an analysis framework.** Every output it produces must be a standard Python/PyTorch/NumPy object. No custom wrapper types in public return values.

## Core design principle

The three universal handshakes are:

| Artifact type | Standard format | Who can consume it |
|---|---|---|
| Weights / parameters at epoch N | `Dict[str, torch.Tensor]` (state dict) or `Dict[str, np.ndarray]` | Anything that loads a PyTorch model, any numerical library |
| Activations / gradients from replay | `Dict[str, torch.Tensor]` keyed by module name | Same as PyTorch forward/backward hooks — any tool expecting hook outputs |
| Derived metrics and tables | `pd.DataFrame` | pandas, polars, matplotlib, any data tool |

If these are solid, a developer can pipe NI outputs into Captum, TransformerLens, sklearn, scipy, their own package, or a Jupyter notebook without glue code.

## Tasks

- [ ] `NI-DELTA-001` Standardize tensor export surface.
  - Add `SnapshotCollection.to_state_dict(epoch) -> Dict[str, Tensor]`: returns a standard PyTorch state dict for a given epoch, loadable with `model.load_state_dict()` directly.
  - Add `SnapshotCollection.to_numpy(epoch, layers=None) -> Dict[str, np.ndarray]`: returns parameter tensors as numpy arrays keyed by layer name. No NI types in the return value.
  - `ReplayResult.activations` and `ReplayResult.gradients` are `TensorMap` objects — a transparent `dict[str, torch.Tensor]` subclass keyed by module name, with the same layout as PyTorch forward hooks.  `isinstance(result.activations, dict)` is `True`; tensor values are plain `torch.Tensor`.
  - `TensorMap.to_numpy() -> dict[str, np.ndarray]` is available on both `result.activations` and `result.gradients`, enabling symmetric NumPy conversion.
  - Acceptance:
    - A developer can call `model.load_state_dict(col.to_state_dict(epoch=5))` with zero NI knowledge beyond that one call.
    - Activation dict from replay matches the exact key/shape layout a manual `register_forward_hook` would produce.
    - Tests confirm no NI-internal types appear in any public return value.

- [ ] `NI-DELTA-007` Validate tensor output compatibility with Captum, TorchLens, and TransformerLens.
  - **Captum**: pass `ReplaySession` activation dict to `captum.attr.LayerActivation` and at least one gradient-based attribution method (e.g. `IntegratedGradients`). Verify no type conversion is needed.
  - **TorchLens**: run TorchLens on the same toy model and the same input batch; assert that the layer output dict keys and tensor shapes match what `ReplaySession` produces.
  - **TransformerLens**: for a small transformer toy model, verify that NI `ReplaySession` activation dict keys and shapes are compatible with what TransformerLens hook output would produce for the same modules.
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
- Captum compatibility test: `ReplaySession` activation and gradient dicts feed into Captum attribution methods without type conversion (skipped if Captum not installed).
- TorchLens compatibility test: NI replay output keys and shapes match TorchLens extraction for the same model and inputs (skipped if TorchLens not installed).
- TransformerLens compatibility test: NI replay output structure is compatible with TransformerLens hook output for a small transformer (skipped if TransformerLens not installed).

## Definition of Done

- `SnapshotCollection.to_state_dict` and `to_numpy` return only standard types; no NI wrapper types in any public return value.
- `ReplaySession` activation and gradient outputs are structurally identical to what `register_forward_hook` / `register_full_backward_hook` would produce.
- Captum, TorchLens, and TransformerLens compatibility is verified end-to-end with no glue code required.
- All Sprint 8 analyzer development can proceed without reopening these contracts.
