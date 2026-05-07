# Sprint 3: Snapshot Mechanism & HDF5 Storage

**Duration**: 2–3 days  
**Goal**: Implement reliable weight capture and persistence in a single integrated step.

## Tasks
- Add `snapshot(epoch=None, step=None, metadata=None)` method to `NeuroInquisitor`
- Inside `snapshot()`: iterate over `model.named_parameters()`, detach and move to CPU, convert to NumPy, write directly to HDF5 (no in-memory buffer)
- Hierarchical HDF5 layout: `/epoch_{epoch:04d}/` group per snapshot, with each parameter as a named dataset; `_metadata` stored as group attributes
- After each write, call `file.flush()` so prior snapshots survive a process crash
- Duplicate snapshot prevention: if a group for `(epoch, step)` already exists, raise a descriptive `ValueError` rather than silently colliding
- Add optional gzip compression + chunking when `compress=True` (set on each dataset)
- Add `load_snapshot(epoch)` helper method that reads a snapshot back from the HDF5 file and returns a `dict[str, np.ndarray]`

## Testing
- Write unit tests in `tests/test_snapshot.py`:
  - Test snapshot with a small MLP: verify all parameter names and shapes are saved correctly
  - Test that metadata is stored as HDF5 group attributes and round-trips correctly
  - Test duplicate snapshot raises `ValueError` with a descriptive message
  - Test GPU snapshot: skip with `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`; verify parameters are correctly moved to CPU before saving
  - Verify tensor shapes are preserved after save and load
- Write unit tests in `tests/test_storage.py`:
  - Test that HDF5 file is created with the correct group hierarchy
  - Test that weights are saved with correct shapes and dtype
  - Test compression attribute: verify dataset has `compression='gzip'` set when `compress=True` (do NOT compare file sizes — small tensors may be larger after compression)
  - Test `load_snapshot(epoch)` round-trip: values loaded from disk match the original tensors within float tolerance
  - Test multiple snapshots in the same file coexist correctly
  - **Crash-recovery test**: write 3 snapshots, close the HDF5 file object directly without calling `observer.close()`, reopen the file with `h5py.File`, and verify all 3 snapshots are readable

## Definition of Done
- `observer.snapshot(epoch=5)` writes weights directly to the HDF5 file and flushes to disk
- `observer.load_snapshot(epoch=5)` returns a dict matching the original parameter values
- Duplicate snapshot raises a clear `ValueError`
- All tests in `tests/test_snapshot.py` and `tests/test_storage.py` pass
