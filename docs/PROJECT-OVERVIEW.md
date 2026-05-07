# neuroinquisitor Project – Phase 1 Overview

**Project Name**: neuroinquisitor  
**Repository**: `neuroinquisitor`  
**Goal**: Build a professional, pip-installable Python package for PyTorch weight observability, with a small, **explicit** API: construct **`NeuroInquisitor(model, ...)`** alongside your model, call **`snapshot()`** from the training loop, then **`close()`** when done. Weights are read back with **`load_snapshot()`** from a local HDF5 file. The model is **not wrapped or modified** (details below).

## Objectives
- Build a clean, reusable, and **pip-installable** Python package that reliably snapshots and persistently saves all PyTorch model weights during training.
- Store weights in a local, high-performance HDF5 database (via `h5py`).
- Create a well-documented, type-hinted `NeuroInquisitor` class that can be dropped into any training loop with minimal code changes.
- Follow modern Python packaging best practices so the library can be installed via `pip install neuroinquisitor`.

## Goals
- Make weight saving automatic, safe, efficient, and production-ready.
- Support multiple snapshots over training without data loss or corruption.
- Provide easy read-back of saved weights for later analysis.
- Ensure the package is fully tested and follows PyPI standards.

## Package interface & usage (Phase 1)

The library centers on a single user-facing class, **`NeuroInquisitor`**, that you attach to a PyTorch **`nn.Module`** and use from your training loop. Phase 1 is **weights only** (all named parameters), persisted to a **local HDF5** file.

### Construction

Create an observer with at least the **model** and where to write the database:

| Parameter | Role |
|-----------|------|
| `model` | The `nn.Module` whose parameters are snapshotted. |
| `log_dir` | Directory for the HDF5 file and related output. |
| `filename` | HDF5 filename (for example `weights.h5`). |
| `compress` | Whether weight datasets are stored with HDF5 compression (for example gzip). |
| `create_new` | Controls whether to start a new file or continue an existing one (together with safe HDF5 open modes). |

> **Note on `freq`:** the original sprint plan listed a `freq` parameter for periodic snapshots. Phase 1 keeps snapshots **fully explicit** — the user decides when by calling `snapshot()`. `freq` is therefore deferred to a later phase (auto-snapshot mode); it is not part of the Phase 1 constructor.

### Lifecycle (explicit)

Three explicit steps, no magic:

1. **Construct** — `observer = NeuroInquisitor(model, ...)` opens the HDF5 file.
2. **Snapshot** — call `observer.snapshot(...)` from your loop whenever you want a checkpoint. Each call **flushes** to disk, so prior snapshots survive a crash.
3. **Close** — call `observer.close()` when training ends to finalize the file.

The model is **not wrapped or modified** — the observer is a separate object, so `isinstance`, `optimizer.parameters()`, `DistributedDataParallel`, `torch.compile`, and `state_dict` save/load all behave exactly as before.

### During training: `snapshot()`

Call **`snapshot(epoch=..., step=..., metadata=...)`** whenever you want a checkpoint of **all** current parameter tensors. The implementation walks **`model.named_parameters()`**, moves data to CPU, converts to NumPy, and writes into the HDF5 hierarchy. Optional **`metadata`** is stored for bookkeeping. The design includes **device-safe** handling and **duplicate-snapshot** avoidance at a given logical training point.

### After training: read-back

A **`load_snapshot(epoch)`** helper (name may accept related indices; see implementation) lets you **reload** a saved snapshot from disk for analysis, comparison, or tests (for example train → snapshot → load → verify weights match).

### Typical training loop (sketch)

```python
from neuroinquisitor import NeuroInquisitor

observer = NeuroInquisitor(
    model,
    log_dir="./runs",
    filename="weights.h5",
    compress=True,
    create_new=True,
)

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # ... forward, loss, backward, optimizer.step() ...
        pass
    observer.snapshot(epoch=epoch, metadata={"lr": lr})

observer.close()
# weights.h5 now holds multiple snapshots; reload via observer.load_snapshot(epoch=...)
```

**Intended mental model:** install the package → construct **`NeuroInquisitor(model, ...)`** alongside your model (the model itself is unchanged) → call **`snapshot()`** at chosen steps or epochs → call **`close()`** when done → read weights back with **`load_snapshot()`**.

## Deliverables (End of Phase 1)
- Fully functional, pip-installable neuroinquisitor package
- HDF5-based local database with hierarchical structure
- Explicit `snapshot()` method that flushes on each call
- Explicit `close()` to finalize the HDF5 file
- Complete working example notebook
- Comprehensive unit tests
- Proper package structure (`pyproject.toml`)

**Phase 1 Success Criteria**  
You can run `pip install -e .`, train any PyTorch model, call `observer.snapshot()` periodically, call `observer.close()` at the end, and end up with a usable `weights.h5` file containing all weight tensors from multiple epochs. All tests pass.

**Next Phase**  
Gradients, activations, statistics, PCA/UMAP, Captum integration, and full PyPI release.
