# Sprint 2: Core NeuroInquisitor Class

**Duration**: 1–2 days  
**Goal**: Implement the main user-facing `NeuroInquisitor` class.

## Tasks
- Create `NeuroInquisitor` class in `neuroinquisitor/core.py`
- Support these `__init__` parameters: `model`, `log_dir`, `filename`, `compress`, `create_new`
- Define explicit `create_new` semantics:
  - `create_new=True` + file already exists → raise `FileExistsError` with a clear message
  - `create_new=False` + file doesn't exist → raise `FileNotFoundError` with a clear message
  - `create_new=False` + file exists → open in append mode
- Add explicit `close()` method that finalizes the HDF5 file; calling it twice is a no-op
- Add `__del__` that calls `close()` if the file is still open and emits a `ResourceWarning` (last-resort safety net; callers should always use explicit `.close()`)
- Add `__repr__` and basic logging

## Testing
- Write unit tests in `tests/test_core.py`:
  - Test that the class can be instantiated with a dummy model and opens the HDF5 file
  - Test that `close()` properly finalizes and closes the file
  - Test that calling `close()` twice does not raise
  - Test `create_new=True` raises `FileExistsError` when file already exists
  - Test `create_new=False` raises `FileNotFoundError` when file doesn't exist
  - Test `create_new=False` opens in append mode when file exists
  - Test that `__del__` without a prior `close()` emits `ResourceWarning`
  - Test that `__repr__` returns useful information

## Definition of Done
- Class is fully type-hinted and documented
- All `create_new` edge cases have defined, tested behavior
- Explicit `close()` finalizes the file deterministically; double-close is safe
- `__del__` emits `ResourceWarning` if file was not explicitly closed
- All tests in `tests/test_core.py` pass
