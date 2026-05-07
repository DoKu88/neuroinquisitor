# Sprint 4: HDF5 Storage Backend

**Duration**: 2 days  
**Goal**: Persist snapshots to a proper local HDF5 database.

## Tasks
- Implement hierarchical HDF5 layout (`/epoch_0000/`, `_metadata` group)
- Write weights with optional gzip compression and chunking
- Add `flush()` method
- Ensure file is properly closed on errors

## Testing
- Write unit tests in `tests/test_storage.py`:
  - Test that HDF5 file is created with correct group structure
  - Test that weights are saved with correct shapes and values
  - Test compression option (file size comparison)
  - Test `load_snapshot(epoch)` helper (read-back matches original weights)
  - Test multiple snapshots in the same file

## Definition of Done
- After `snapshot()`, the `.h5` file contains correct weights and metadata
- All tests in `tests/test_storage.py` pass