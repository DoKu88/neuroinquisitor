# Sprint 8: Built-in Analyzers

**Goal**: Ship the first built-in analyzers as plain functions scientists can call out of the box. Every analyzer takes standard NI outputs (dicts of tensors or numpy arrays) and returns a `pd.DataFrame`. No custom types, no registration ceremony, no persistence layer — users own their data.

**Prerequisite**: Sprint 7 must be complete. Analyzers consume the format contracts defined there.

## Design principle

Each analyzer is a standalone function importable from `neuroinquisitor.analyzers`. It takes whatever NI already produces and returns a DataFrame the user can immediately plot, filter, export, or pass to another library. No subclassing, no configuration objects.

```python
from neuroinquisitor.analyzers import trajectory_stats

col = ni.collection()
df = trajectory_stats(col.to_numpy())
df.plot()
```

## Tasks

- [ ] `NI-GAMMA-001` Implement `trajectory_stats`.
  - Inputs: `weights: dict[int, np.ndarray]` — the direct output of `SnapshotCollection.by_layer(name)`, where keys are epoch indices and values are the parameter array at that epoch. Callers invoke it once per layer of interest.
  - Typical call pattern:
    ```python
    layer_history = col.by_layer("fc1.weight")  # dict[int, np.ndarray]
    df = trajectory_stats(layer_history)
    ```
  - Computes: L2 distance from init, cosine similarity to init and final, update norm per step, velocity and acceleration summaries.
  - Returns: `pd.DataFrame` with columns `epoch`, and one column per metric. The caller already knows which layer they passed in; no `layer` column is needed unless the caller concatenates results from multiple `by_layer` calls.

- [ ] `NI-GAMMA-002` Implement `spectrum_rank`.
  - Inputs: `weights: dict[str, np.ndarray]`.
  - Computes: singular values, effective rank, stable rank, spectral norm, Frobenius norm.
  - Returns: `pd.DataFrame` with columns `layer`, `epoch`, and one column per metric.

- [ ] `NI-GAMMA-003` Implement `projection_embed`.
  - Inputs: `activations: dict[str, torch.Tensor]` — the direct output of `ReplayResult.activations` (a `TensorMap`, which is a plain `dict` subclass). No unwrapping needed; pass `result.activations` directly.
  - Typical call pattern:
    ```python
    result = ReplaySession(...).run()
    df = projection_embed(result.activations)
    ```
  - Supports PCA by default; UMAP via optional extra (`pip install neuroinquisitor[umap]`), failing gracefully with a clear install hint if missing.
  - Returns: `pd.DataFrame` with columns `layer`, `sample_idx`, `component_0`, `component_1` (and optionally `component_2`).

- [ ] `NI-GAMMA-004` Implement `similarity_compare`.
  - Inputs: `a: dict[str, torch.Tensor]`, `b: dict[str, torch.Tensor]`.
  - Computes CKA by default. Supports within-run epoch comparisons and cross-run comparisons.
  - Returns: `pd.DataFrame` with columns `layer_a`, `layer_b`, `cka`.

- [ ] `NI-GAMMA-005` Implement `probe_linear`.
  - Inputs: `activations: dict[str, torch.Tensor]`, `labels: torch.Tensor`, `test_size: float = 0.2`, `random_state: int = 42`.
  - Trains a linear probe on each layer's activations with a deterministic train/val split.
  - Returns: `pd.DataFrame` with columns `layer`, `train_accuracy`, `val_accuracy`.

- [ ] `NI-GAMMA-006` Add TracIn quick-start example using Captum.
  - Do not implement TracIn from scratch — Captum ships `TracInCP`.
  - Write a short runnable example (under 30 lines) showing how to feed NI checkpoints into `TracInCP`: load checkpoints via `SnapshotCollection.to_state_dict(epoch)`, reconstruct the model at each checkpoint, pass to `TracInCP`.
  - Acceptance:
    - Example runs end-to-end on a toy classification model.
    - No NI-internal TracIn implementation is added.

- [ ] `NI-GAMMA-007` Validate `projection_embed` output compatibility with TensorBoard Projector and FiftyOne.
  - **TensorBoard Projector**: from the `projection_embed` DataFrame, write a short example (under 20 lines) producing the TSV files TensorBoard Projector expects and launching the projector. Only standard pandas/numpy calls permitted after the analyzer call.
  - **FiftyOne**: from `projection_embed` or directly from a `ReplaySession` activation dict, write a short example (under 20 lines) creating a FiftyOne dataset with embeddings attached.
  - These are optional-dependency tests; skip gracefully when `tensorboard` or `fiftyone` are not installed.
  - Acceptance:
    - Both examples are self-contained and under 20 lines.
    - No NI-specific types appear in either example after the initial analyzer call.

## Cross-cutting requirements

All analyzers must:
- Accept only standard types: `dict[str, np.ndarray]`, `dict[str, torch.Tensor]`, Python scalars.
- Return a `pd.DataFrame` with clearly named columns.
- Raise a clear `ImportError` with install instructions when optional dependencies are missing.
- Be importable from `neuroinquisitor.analyzers` directly.

## Testing

- Output type tests: every analyzer returns a `pd.DataFrame`; no NI types in the result.
- Column schema tests: required columns are present and correctly typed for each analyzer.
- Regression tests for trajectory and spectrum metrics on toy models.
- Projection tests for PCA output shape and optional UMAP feature gating.
- Similarity tests for within-run and cross-run CKA comparisons.
- Linear probe tests for deterministic split and reproducible metrics.
- TensorBoard Projector compatibility test (skipped if `tensorboard` not installed).
- FiftyOne compatibility test (skipped if `fiftyone` not installed).

## Definition of Done

- All five analyzers are importable from `neuroinquisitor.analyzers` and return plain DataFrames.
- Optional features (UMAP) fail gracefully with a clear install hint.
- TracIn example runs end-to-end on a toy model.
- TensorBoard Projector and FiftyOne can visualize NI output with no NI-specific glue code.
- Integration tests cover at least one complete flow from replay to analyzer output.
