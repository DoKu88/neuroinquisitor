# Sprint 10: LLM-Scale Core Fixes

**Goal**: Make NeuroInquisitor work correctly with bfloat16 models, add selective layer capture to keep snapshot sizes manageable, and replace the in-memory BytesIO serialization path with a streaming file-based path that does not hold entire checkpoints in RAM.

**Prerequisite**: Sprints 1–8 complete. Sprint 10 is a prerequisite for Sprint 11 (S3 backend) and Sprint 12 (Qwen 7B example).

---

## Tasks

- [ ] `NI-LLM-001` Fix bfloat16 TypeError in parameter extraction.
  - **File**: `src/neuroinquisitor/core.py:206-209`
  - Currently, `param.detach().cpu().numpy()` raises `TypeError` on bfloat16 tensors because numpy has no bfloat16 dtype. This silently blocks all LLM-scale usage.
  - Fix: convert bfloat16 params to float32 before `.numpy()`. Same fix for the buffers block (lines 213-218).
  - Implementation:
    ```python
    params: dict[str, np.ndarray] = {
        name: (param.detach().cpu().to(torch.float32).numpy()
               if param.dtype == torch.bfloat16
               else param.detach().cpu().numpy())
        for name, param in self._model.named_parameters()
    }
    ```
  - Acceptance:
    - Unit test: create a bfloat16 `nn.Linear`, call `snapshot()`, no error raised, stored values match float32 cast of original.
    - Same test for buffers (bfloat16 `BatchNorm`).

- [ ] `NI-LLM-002` Add `layer_filter` to `NeuroInquisitor`.
  - **Files**: `src/neuroinquisitor/core.py`, `src/neuroinquisitor/schema.py`
  - Add `layer_filter: set[str] | None = None` parameter to `NeuroInquisitor.__init__()` (after `capture_policy`). Store as `self._layer_filter`.
  - Apply at parameter extraction in `snapshot()`:
    ```python
    for name, param in self._model.named_parameters()
    if self._layer_filter is None or name in self._layer_filter
    ```
  - Add `layer_filter: list[str] | None = None` field to `CapturePolicy` in `schema.py` so the manifest records what was captured.
  - Populate `CapturePolicy.layer_filter` from `self._layer_filter` when building the `IndexEntry`.
  - Acceptance:
    - Unit test: create a 3-layer model, pass `layer_filter={"layer1.weight"}`, verify the stored snapshot contains only that key.
    - `SnapshotRef.layers` in the index reflects the filtered set.
    - `layer_filter=None` (default) captures all params — existing tests unchanged.

- [ ] `NI-LLM-003` Add streaming `write_to_path()` to `Format` and override in `HDF5Format`.
  - **Files**: `src/neuroinquisitor/formats/base.py`, `src/neuroinquisitor/formats/hdf5_format.py`

  **`Format` base class** — add a concrete (non-abstract) method so existing subclasses need not change:
  ```python
  def write_to_path(
      self,
      dest: Path,
      params: dict[str, np.ndarray],
      metadata: dict[str, object],
      compress: bool = False,
      buffers: dict[str, np.ndarray] | None = None,
  ) -> None:
      """Write snapshot directly to dest. Default falls back to write()."""
      dest.write_bytes(self.write(params, metadata, compress=compress, buffers=buffers))
  ```

  **`HDF5Format`** — override `write_to_path()` to write directly via h5py, eliminating the BytesIO buffer:
  ```python
  def write_to_path(self, dest, params, metadata, compress=False, buffers=None) -> None:
      with h5py.File(str(dest), "w") as f:
          # same body as write() but using dest path instead of BytesIO
  ```
  h5py natively supports writing to a path string — this means a 14 GB snapshot flows straight from tensor memory to disk with no intermediate in-RAM copy beyond h5py's internal chunk buffers (~few MB).

  - Acceptance:
    - Unit test: call `HDF5Format().write_to_path(tmp_path, params, meta)`, then `HDF5Format().read(tmp_path)` — arrays must be numerically identical to the `write()` → `read()` path.
    - Unit test: confirm `write_to_path()` on a format that does NOT override it (e.g., a stub subclass) falls back to `write()` without error.

---

## Testing

- `test_core.py`: bfloat16 snapshot round-trip (NI-LLM-001).
- `test_core.py`: layer_filter — captured layers, index manifest correctness (NI-LLM-002).
- `test_hdf5_format.py`: `write_to_path()` round-trip numerical identity; fallback behavior for base class (NI-LLM-003).
- All existing tests must continue to pass without modification (backward compat).

## Definition of Done

- `snapshot()` works on any bfloat16 model without error.
- `layer_filter` correctly limits captured parameters and records the filter in the manifest.
- `HDF5Format.write_to_path()` writes a snapshot to disk without holding the full checkpoint in RAM.
- Existing test suite passes at 90% coverage threshold.
