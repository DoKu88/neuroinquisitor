# Sprint 2: Core NeuroInquisitor Class

**Duration**: 1–2 days  
**Goal**: Implement the main user-facing `NeuroInquisitor` class.

## Tasks
- Create `NeuroInquisitor` class in `neuroinquisitor/core.py`
- Support these `__init__` parameters: `model`, `log_dir`, `filename`, `compress`, `create_new`
  - Note: `freq` is deferred to a later phase; Phase 1 keeps snapshots fully explicit (user calls `snapshot()`)
- Add explicit `close()` method that finalizes the HDF5 file
- Add `__repr__` and basic logging
- Handle HDF5 file opening modes safely (open in `__init__`)

## Testing
- Write unit tests in `tests/test_core.py`:
  - Test that the class can be instantiated with a dummy model and opens the HDF5 file
  - Test that `close()` properly finalizes and closes the file
  - Test that `__repr__` returns useful information
  - Test different combinations of `create_new` and file modes

## Definition of Done
- Class is fully type-hinted and documented
- Explicit `close()` finalizes the file deterministically
- All tests in `tests/test_core.py` pass
