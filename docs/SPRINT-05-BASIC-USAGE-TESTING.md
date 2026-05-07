# Sprint 5: Basic Usage, Integration & Testing

**Duration**: 2 days  
**Goal**: Make the library usable in real training loops, verify correctness, and prepare for pip distribution.

## Tasks
- Write a minimal training-loop example (tiny MLP or torchvision model)
- Add `load_snapshot(epoch)` helper to read weights back
- Create complete example notebook: `examples/01_basic_weight_saving.ipynb`
- Update `README.md` with clear usage example and pip instructions
- Final package metadata and documentation review

## Testing
- Expand test suite:
  - Integration test: full training loop with multiple snapshots
  - End-to-end test: train → snapshot → load → verify weights match
- Run full test suite with `pytest`

## Definition of Done
- Complete working example notebook
- All tests across the entire package pass (`pytest`)
- Package installs cleanly in a fresh virtual environment via `pip install -e .`
- Ready for the next development phase