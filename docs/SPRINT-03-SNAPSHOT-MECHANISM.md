# Sprint 3: Snapshot Mechanism

**Duration**: 2 days  
**Goal**: Implement reliable weight capture at any training point.

## Tasks
- Add `snapshot(epoch=None, step=None, metadata=None)` method to `NeuroInquisitor`
- Inside `snapshot()`: iterate over `model.named_parameters()`, convert to CPU numpy
- Store optional metadata
- Add safety checks (device handling, duplicate snapshot prevention)

## Testing
- Write unit tests in `tests/test_snapshot.py`:
  - Test snapshot with a small MLP model (check that all parameters are saved)
  - Test that metadata is correctly stored as attributes
  - Test duplicate snapshot prevention
  - Test that snapshot works on both CPU and GPU models (if GPU available)
  - Verify tensor shapes are preserved

## Definition of Done
- `observer.snapshot(epoch=5)` successfully captures weights in memory
- All tests in `tests/test_snapshot.py` pass