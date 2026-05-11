# Sprint 11: Cloud Storage & Modern Formats

**Goal**: Add an S3 backend that uploads snapshots asynchronously so the training loop is never blocked, add `SafetensorsFormat` for fast large-model weight storage, wire both into `core.py`'s snapshot path, and update `NeuroInquisitor.close()` to drain pending uploads before the process exits.

**Prerequisite**: Sprint 10 must be complete (`write_to_path()` on Format base, bfloat16 fix).

---

## Tasks

- [ ] `NI-LLM-004` Add `S3Backend`.
  - **New file**: `src/neuroinquisitor/backends/s3.py`
  - Add `boto3>=1.26` as optional dep: `pip install neuroinquisitor[s3]`.
  - Add `moto>=4.0` to `dev` extra for unit tests.

  **Constructor**:
  ```python
  S3Backend(
      bucket: str,
      prefix: str = "",
      tmp_dir: str | Path | None = None,   # local temp dir; defaults to tempfile.mkdtemp()
      max_workers: int = 4,
      cleanup_after_upload: bool = False,   # delete tmp file after confirmed upload
  )
  ```
  Credentials via standard boto3 chain only (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE`, or IAM role). No credential parameters on the constructor.

  **Write pattern**:
  - `write(key, data: bytes)`:
    - If `key` ends in `.json` (index files): upload synchronously inline — index consistency is more important than throughput.
    - All other keys: write bytes to a local tmp file, enqueue S3 upload to a `ThreadPoolExecutor` background thread, return immediately.
  - `write_from_path(key, src: Path)`: enqueue background upload of an already-written tmp file. Used by `core.py` streaming path (see NI-LLM-006).

  **Read pattern**:
  - `read_path(key) -> Path`: download from S3 to tmp file if not cached locally, return `Path`. Blocking (reads are not on the hot training path).

  **Other methods**: `exists()`, `delete()` — synchronous, standard boto3 calls.

  **Draining**:
  - `flush(timeout=None)`: block until all pending upload futures complete (re-raise any upload exceptions).
  - `close()`: call `flush()` then `executor.shutdown(wait=True)`.

  **Pending future management**: use a `collections.deque` + `threading.Lock()`. Prune completed futures on each `_enqueue_upload()` call to avoid unbounded memory growth during long runs.

  - Acceptance:
    - All tests use `moto` to mock S3 — no real S3 calls in CI.
    - `flush()` re-raises if a background upload raised an exception.
    - `.json` key writes are confirmed synchronous (not in the upload queue).
    - `read_path()` downloads from S3 when the file is not in the local tmp cache.
    - Graceful `ImportError` with `"Install neuroinquisitor[s3]"` hint when boto3 is absent.

- [ ] `NI-LLM-005` Add `SafetensorsFormat`.
  - **New file**: `src/neuroinquisitor/formats/safetensors_format.py`
  - Add `safetensors>=0.4` as optional dep: `pip install neuroinquisitor[safetensors]`.

  **Key design**: safetensors handles bfloat16 natively via `safetensors.torch` — no numpy round-trip needed for the write path. Add `write_tensors_to_path()` alongside the standard interface:
  ```python
  def write_tensors_to_path(
      self, dest: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, object]
  ) -> None:
      # safetensors.torch.save_file(tensors, str(dest), metadata=str_metadata)
      # Handles bfloat16 directly without dtype conversion
  ```
  Metadata stored as `dict[str, str]` (safetensors requirement) — serialize scalars with `str()`, parse on read.

  **Standard interface**:
  - `write_to_path(dest, params: dict[str, np.ndarray], ...)`: use `safetensors.numpy.save_file()`.
  - `write(...)`: serialize to bytes via `safetensors.numpy.save()` for small in-memory use.
  - `read(path, layers=None)`: use `safe_open(path, framework="np")` for memory-mapped selective reads — only the requested layer's bytes are read from disk.
  - `list_layers(path)`: `list(safe_open(path, framework="np").keys())`.

  - Acceptance:
    - Round-trip test: `write_to_path()` → `read()` yields numerically identical arrays.
    - Selective read: `read(path, layers={"layer.weight"})` does not load other layers (verify via `safe_open` key access).
    - `write_tensors_to_path()` round-trip on bfloat16 tensors — no dtype conversion loss.
    - Graceful `ImportError` with `"Install neuroinquisitor[safetensors]"` hint.

- [ ] `NI-LLM-006` Wire streaming path into `core.py` and drain backend on `close()`.
  - **File**: `src/neuroinquisitor/core.py`

  **Streaming path in `snapshot()`** (replaces lines 228-234):

  Detect whether the backend supports path-based writes (duck-typing, no ABC change):
  ```python
  if hasattr(self._backend, "write_from_path"):
      # Streaming path: write to tmp file, enqueue upload
      suffix = self._format.extension
      tmp = Path(tempfile.mktemp(suffix=suffix, dir=getattr(self._backend, "_tmp", None)))
      if hasattr(self._format, "write_tensors_to_path") and raw_tensors is not None:
          # SafetensorsFormat: pass torch tensors directly (bfloat16-safe)
          self._format.write_tensors_to_path(tmp, raw_tensors, file_metadata)
      else:
          self._format.write_to_path(tmp, params, file_metadata,
                                     compress=self._compress, buffers=buffers)
      self._backend.write_from_path(file_key, tmp)
  else:
      # Original path (LocalBackend, any custom backend without write_from_path)
      data = self._format.write(params, file_metadata,
                                compress=self._compress, buffers=buffers)
      self._backend.write(file_key, data)
  ```

  For the safetensors tensor path: before the numpy conversion in param extraction, keep a `raw_tensors: dict[str, torch.Tensor]` dict when `hasattr(self._format, "write_tensors_to_path")`.

  **`close()` backend drain** (extend existing method at line 289):
  ```python
  def close(self) -> None:
      if self._closed:
          return
      self._closed = True
      if hasattr(self._backend, "close"):
          self._backend.close()   # drains S3 upload queue before process exits
      logger.info("NeuroInquisitor closed %s", self._log_dir)
  ```

  - Acceptance:
    - Integration test: `NeuroInquisitor` with a mocked `S3Backend` — verify `write_from_path` is called (not `write`) for snapshot files.
    - `.close()` on an `S3Backend`-backed instance calls `S3Backend.close()`.
    - `LocalBackend`-backed instance: existing behavior unchanged (no regression).

- [ ] `NI-LLM-007` Update package config and public exports.
  - **Files**: `pyproject.toml`, `src/neuroinquisitor/__init__.py`

  `pyproject.toml` additions:
  ```toml
  [project.optional-dependencies]
  s3 = ["boto3>=1.26"]
  safetensors = ["safetensors>=0.4"]
  zarr = ["zarr>=2.16", "numcodecs>=0.12"]
  # add moto>=4.0 to dev extra
  ```

  `__init__.py` conditional exports:
  ```python
  try:
      from neuroinquisitor.backends.s3 import S3Backend
      __all__ += ["S3Backend"]
  except ImportError:
      pass

  try:
      from neuroinquisitor.formats.safetensors_format import SafetensorsFormat
      __all__ += ["SafetensorsFormat"]
  except ImportError:
      pass
  ```

  - Acceptance:
    - `from neuroinquisitor import S3Backend` works when boto3 is installed; raises `ImportError` with install hint when it is not.
    - Same for `SafetensorsFormat`.
    - `pip install neuroinquisitor[s3]` installs boto3; `pip install neuroinquisitor[safetensors]` installs safetensors.

---

## Testing

- `tests/test_s3_backend.py` (new): all S3Backend tests via moto. Tests: write/read/exists/delete/flush/close, index sync writes, cleanup_after_upload, exception propagation from background thread.
- `tests/test_safetensors_format.py` (new): round-trip, selective read, bfloat16 tensor write, ImportError message.
- `tests/test_core.py`: streaming path integration test with mocked S3Backend; `close()` drain.
- `tests/test_import.py`: add `S3Backend` and `SafetensorsFormat` to import presence/absence tests.
- All existing tests must pass unchanged.

## Definition of Done

- S3Backend uploads in background threads; training loop does not block on S3 latency.
- `close()` always drains pending uploads before process exits — no data loss on Modal container shutdown.
- `SafetensorsFormat` writes and reads bfloat16 tensors without dtype conversion.
- All optional extras degrade gracefully with actionable install instructions.
- Coverage threshold (90%) maintained.
