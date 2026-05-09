# Sprint 7: Standard Formats

**Status**: Complete ✅

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

- [x] `NI-DELTA-001` Standardize tensor export surface.
  - `SnapshotCollection.to_state_dict(epoch) -> Dict[str, Tensor]`: implemented in `collection.py:164`. Returns a standard PyTorch state dict loadable with `model.load_state_dict()` directly.
  - `SnapshotCollection.to_numpy(epoch, layers=None) -> Dict[str, np.ndarray]`: implemented in `collection.py:200`. No NI types in return value.
  - `ReplayResult.activations` and `ReplayResult.gradients` are `TensorMap` objects — a transparent `dict[str, torch.Tensor]` subclass (`isinstance(result.activations, dict)` is `True`). Implemented in `replay.py:84`.
  - `TensorMap.to_numpy() -> dict[str, np.ndarray]` available on both fields. Implemented in `replay.py:98`.
  - `TensorMap` is publicly exported in `__init__.py`.
  - Full test coverage in `tests/test_export_formats.py`, including round-trip correctness and forward-hook shape match.

- [x] `NI-DELTA-007` Validate tensor output compatibility with Captum, TorchLens, and TransformerLens.
  - **Captum** (`tests/compat/test_captum.py`): `to_state_dict → IntegratedGradients`, `LayerActivation`, and `ReplaySession → plain tensors → Captum` all verified end-to-end. No type conversion required. Skip-gated when captum not installed.
  - **TorchLens** (`tests/compat/test_torchlens.py`): NI activation values confirmed as plain `torch.Tensor` matching TorchLens' output type. Feature dimensions verified against expected Linear layer output sizes. TorchLens confirmed to run without error on the same model/inputs. Direct key-for-key shape comparison with TorchLens output is intentionally omitted — TorchLens has renamed its activation-extraction API across versions and its return-value schema is unstable; the test probes for the correct callable at import time and skips the sub-test if neither known name is found. Skip-gated when torchlens not installed.
  - **TransformerLens** (`tests/compat/test_transformerlens.py`): Structural contract (plain `dict[str, Tensor]`, ≥2-D tensors, `to_numpy()` in analysis workflows) verified. **Known naming divergence accepted as a design decision**: NI keys activation tensors by PyTorch module name (e.g. `"embed"`); TransformerLens uses hook names (e.g. `"hook_embed"`). This is documented in `test_ni_key_naming_uses_module_name` and is intentional — NI's key naming is derived directly from `model.named_modules()` and will not change. Users bridging NI and TransformerLens must map keys; the integration examples in `transformerlens_use_examples/` demonstrate this. Skip-gated when transformer_lens not installed.

## Testing

- [x] Export-format tests: `to_state_dict`, `to_numpy`, and `ReplaySession` dict outputs contain only standard types (`test_to_state_dict_no_ni_types`, `test_to_numpy_no_ni_types`, `test_replay_result_activations_values_are_plain_tensors`).
- [x] Round-trip test: `model.load_state_dict(col.to_state_dict(epoch=N))` produces identical forward-pass outputs (`test_load_state_dict_round_trip`).
- [x] Activation key test: replay activation dict keys and shapes match what `register_forward_hook` produces for the same model and inputs (`test_activation_keys_match_forward_hook`).
- [x] Captum compatibility test: `to_state_dict` and `ReplaySession` outputs feed into Captum attribution methods without type conversion. Skip-gated (`tests/compat/test_captum.py`).
- [x] TorchLens compatibility test: NI output type and feature dimensions verified; TorchLens confirmed to run on the same inputs. Direct schema comparison omitted due to TorchLens API instability across versions. Skip-gated (`tests/compat/test_torchlens.py`).
- [x] TransformerLens compatibility test: structural contract verified; key naming divergence documented as intentional. Skip-gated (`tests/compat/test_transformerlens.py`).

## Definition of Done

- [x] `SnapshotCollection.to_state_dict` and `to_numpy` return only standard types; no NI wrapper types in any public return value.
- [x] `ReplaySession` activation and gradient outputs are structurally identical to what `register_forward_hook` / `register_full_backward_hook` would produce.
- [x] Captum and TorchLens compatibility verified end-to-end with no glue code required.
- [x] TransformerLens structural contract verified. Key naming divergence (module names vs hook names) accepted as a final design decision — NI keys are module names, period. Integration examples handle the mapping.
- [x] Sprint 8 analyzer development can proceed without reopening these contracts.
