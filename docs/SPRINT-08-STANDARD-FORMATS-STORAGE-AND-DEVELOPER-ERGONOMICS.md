# Sprint 8: Standard Formats, Storage & Developer Ergonomics

**Goal**: Make every artifact NeuroInquisitor captures accessible in formats a developer already knows — `Dict[str, Tensor]`, `np.ndarray`, `pd.DataFrame` — so that analysis can start immediately with any tool, internal or external, without understanding NI internals.

The guiding constraint for this sprint: **NI is a data layer, not an analysis framework.** Every output it produces should be expressible as a standard Python/PyTorch/NumPy/Pandas object. No custom wrapper objects in public return values.

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
  - Register built-in analyzers from Sprint 7 via a simple dict-based internal registry.
  - Each entry carries: `name`, `version`, `required_inputs` (e.g. `["weights"]`, `["activations", "labels"]`), `output_format` (tensor / table / both), and a short `description`.
  - Keep registration code flat and explicit — no metaclasses, no magic import side-effects.
  - Keep the interface stable and public so a developer can register their own analyzer with two lines of code if they want to, but do not build external plugin loading yet.
  - Acceptance:
    - `ni plugins list` (from CLI task below) reads from this registry.
    - Adding a new analyzer requires only appending to the registry dict and implementing the interface.

- [ ] `NI-DELTA-003` Persist derived tables as Parquet.
  - All tabular analyzer outputs (trajectory stats, probe scores, similarity matrices, etc.) are saved as Parquet files alongside HDF5 artifacts.
  - Files must be directly readable with `pd.read_parquet()` — no NI import required to open them.
  - Include minimal provenance columns in every table: `run_id`, `epoch`, `layer`, `analyzer_name`, `analyzer_version`.
  - Acceptance:
    - A developer can open a result with `pd.read_parquet("runs/exp_a/trajectory_stats.parquet")` and get a usable DataFrame.
    - Parquet files validate against the provenance column requirement.

- [ ] `NI-DELTA-004` Add optional tensor storage backends.
  - Add `safetensors` as an optional extra (`pip install neuroinquisitor[safetensors]`).
    - Safetensors is a portable, safe format widely used for sharing checkpoints. Good default for exporting a single epoch's weights.
  - Add `zarr` as an optional extra (`pip install neuroinquisitor[zarr]`).
    - Zarr is better than HDF5 for large array stacks and object-store-backed runs.
  - Both backends must produce files readable by their respective libraries without NI installed.
  - Acceptance:
    - `SnapshotCollection.export(epoch=5, format="safetensors", path="epoch5.safetensors")` works when the extra is installed.
    - Clear `ImportError` with install instructions when the extra is missing.
    - Docs explain when to use HDF5 vs Safetensors vs Zarr.

- [ ] `NI-DELTA-005` Add CLI surface.
  - Commands:
    - `ni runs list` — list runs and their epoch ranges.
    - `ni manifest show <run>` — print manifest metadata.
    - `ni snapshot export <run> --epoch N --format [numpy|safetensors|hdf5]` — export a snapshot to a file a developer can open without NI.
    - `ni analyze run <run> --analyzer <name>` — run a built-in analyzer.
    - `ni plugins list` — list registered analyzers with name, version, and required inputs.
  - CLI must call the same public Python APIs; no internal shortcuts.
  - Acceptance:
    - `ni snapshot export` produces a file openable by `np.load` or `safetensors.torch.load_file` depending on format.
    - All commands have `--help` output.

- [ ] `NI-DELTA-006` Write developer quick-start guide.
  - This is not a tutorial about NI internals. It answers the question: *"I have captured data — where do I start?"*
  - Cover four concrete starting points, each as a short runnable example:
    1. **Weights over time**: load epochs 0–20 as numpy arrays, compute something (e.g. L2 norm per layer), plot it. No NI API after the export call.
    2. **Reload a checkpoint**: `model.load_state_dict(col.to_state_dict(epoch=10))`, run a forward pass, do whatever you want with the output.
    3. **Activations from replay**: run `ReplaySession`, get a `Dict[str, Tensor]`, pass directly to a linear classifier or to any tool that accepts hook-style dicts.
    4. **Derived table as a starting point**: open a Parquet file with pandas, filter/plot/export — no NI knowledge required.
  - Each example should be self-contained and under 20 lines.
  - Acceptance:
    - All four examples run without error against a toy model.
    - A developer reading only this guide (not the full docs) can start an analysis in under 10 minutes.

## Testing

- Export-format tests: confirm `to_state_dict`, `to_numpy`, and `ReplaySession` dict outputs contain only standard types (Tensor, ndarray, Python scalars).
- Round-trip test: `model.load_state_dict(col.to_state_dict(epoch=N))` produces identical forward-pass outputs to the original model at that epoch.
- Activation key test: replay activation dict keys match the module names passed to the session, with the same shape a manual hook would produce.
- Registry tests: built-in analyzers are discoverable, metadata fields are present and typed correctly.
- Parquet tests: derived table files open with `pd.read_parquet` and contain required provenance columns.
- Backend feature-gate tests: missing optional extras raise `ImportError` with the correct install hint.
- CLI tests: each command produces correct output and calls the public Python API (not internal shortcuts).

## Definition of Done

- Every public return value that carries model data is a standard type: `Dict[str, Tensor]`, `Dict[str, np.ndarray]`, `pd.DataFrame`, or a Python scalar. No NI wrapper types in outputs.
- A checkpoint at any epoch is loadable into a stock PyTorch model in one line.
- Replay activation/gradient dicts are structurally identical to what a manual `register_forward_hook` would produce.
- All tabular derived data is persisted as Parquet and readable without NI installed.
- Optional backends degrade gracefully with actionable install instructions.
- CLI exposes inspection and export without requiring a Python environment.
- Developer quick-start guide gives four concrete, runnable on-ramps into analysis.
