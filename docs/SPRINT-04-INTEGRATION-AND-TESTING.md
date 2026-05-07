# Sprint 4: Integration, Examples & Distribution

**Duration**: 1–2 days  
**Goal**: Make the library usable in real training loops, verify correctness end-to-end, and prepare for pip distribution.

## Tasks
- Write a minimal training-loop example as a plain Python script: `examples/basic_usage.py` (tiny MLP, multiple epochs, `snapshot()` per epoch, `close()` at end)
- Update `README.md` with clear usage example and pip instructions
- Final package metadata and documentation review
- Verify the package installs cleanly in a fresh virtual environment via `pip install -e .`

## Testing
- Integration test in `tests/test_integration.py`:
  - Full training loop with multiple snapshots: train for N steps, call `snapshot()` each epoch, call `close()`
  - End-to-end round-trip: train → snapshot → `load_snapshot()` → verify loaded weights match the model's current state at that epoch
- Run full test suite with `pytest` and confirm coverage meets the 90% floor

## Definition of Done
- `examples/basic_usage.py` runs end-to-end with `python examples/basic_usage.py`
- All tests across the entire package pass (`pytest`)
- Package installs cleanly in a fresh virtual environment via `pip install -e .`
- Ready for the next development phase
