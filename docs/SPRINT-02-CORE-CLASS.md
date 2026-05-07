# Sprint 2: Core NeuroInquisitor Class

**Duration**: 1–2 days  
**Goal**: Implement the main user-facing `NeuroInquisitor` class.

## Tasks
- Create `NeuroInquisitor` class in `neuroinquisitor/core.py`
- Support these `__init__` parameters: `model`, `log_dir`, `filename`, `freq`, `compress`, `create_new`
- Add `start()` / `close()` methods and full context-manager support (`__enter__` / `__exit__`)
- Add `__repr__` and basic logging
- Handle HDF5 file opening modes safely

## Testing
- Write unit tests in `tests/test_core.py`:
  - Test that the class can be instantiated with a dummy model
  - Test that `__enter__` / `__exit__` properly opens and closes the file
  - Test that `__repr__` returns useful information
  - Test different combinations of `create_new` and file modes

## Definition of Done
- Class is fully type-hinted and documented
- Context manager works correctly
- All tests in `tests/test_core.py` pass
