# neuroinquisitor

Neural network weight observability for PyTorch — snapshot, persist, and reload model weights during training, then analyse how they evolve.

- **One observer, one line per snapshot.** Attach `NeuroInquisitor` to any `nn.Module` and call `.snapshot(epoch=…)` (or `step=…`) inside your training loop.
- **One file per snapshot, plus an index.** Each snapshot is an independent HDF5 file; an `index.json` tracks epochs, steps, layer names, and per-snapshot metadata. Append more snapshots to a run later without rewriting anything.
- **Lazy, filterable read-back.** `NeuroInquisitor.load(...)` returns a `SnapshotCollection` that touches no tensor data until you ask for it — and lets you slice by epoch, layer, or both, with parallel reads under the hood.
- **Pluggable storage and format.** Local filesystem + HDF5 are built in; both the `Backend` and `Format` are abstract so you can drop in your own (object stores, Zarr, Safetensors, etc.).

## Installation

```bash
pip install neuroinquisitor
```

To run the visualisation examples (matplotlib):

```bash
pip install "neuroinquisitor[examples]"
```

For development:

```bash
git clone https://github.com/doku88/neuroinquisitor.git
cd neuroinquisitor
pip install -e ".[dev]"
```

## Quick start

```python
import torch
import torch.nn as nn
from neuroinquisitor import NeuroInquisitor

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

X = torch.randn(128, 4)
y = (X.sum(dim=1, keepdim=True) > 0).float()

observer = NeuroInquisitor(
    model,
    log_dir="./runs/quickstart",
    compress=True,
    create_new=True,
)

for epoch in range(10):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
    observer.snapshot(epoch=epoch, metadata={"loss": loss.item()})

observer.close()
```

After training, `./runs/quickstart/` contains one `epoch_XXXX.h5` per snapshot plus a single `index.json`.

## Loading and analysing snapshots

`NeuroInquisitor.load()` does **not** require a model — it's a pure read path for post-training analysis. Nothing is read from disk until you ask for it.

```python
from neuroinquisitor import NeuroInquisitor

col = NeuroInquisitor.load("./runs/quickstart")

col.epochs           # [0, 1, 2, ..., 9]
col.layers           # ['0.weight', '0.bias', '2.weight', '2.bias']
len(col)             # 10

# All layers at one epoch
weights_e3 = col.by_epoch(3)              # dict[str, np.ndarray]

# One layer across all epochs (parallel reads)
fc1_history = col.by_layer("0.weight")    # dict[int, np.ndarray]

# Narrow the view — composes lazily, zero I/O
late_fc1 = col.select(epochs=range(5, 10), layers="0.weight")
```

You can also push the filters into `load()` itself:

```python
late = NeuroInquisitor.load("./runs/quickstart", epochs=range(5, 10))
fc1_only = NeuroInquisitor.load("./runs/quickstart", layers="0.weight")
```

## API reference

### `NeuroInquisitor(model, log_dir, compress, create_new, backend, format)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | `nn.Module` whose parameters are snapshotted |
| `log_dir` | `"."` | Directory for snapshot files and `index.json` |
| `compress` | `False` | Hint to the format to use compression (gzip for HDF5) |
| `create_new` | `True` | `True` → start a fresh run, errors if an index already exists. `False` → append to an existing run, errors if no index is found. |
| `backend` | `"local"` | Storage backend — `"local"` or a `Backend` instance |
| `format` | `"hdf5"` | Snapshot file format — `"hdf5"` or a `Format` instance |

### `observer.snapshot(epoch=None, step=None, metadata=None)`

Writes all current model parameters to a new snapshot file and updates the index.

- At least one of `epoch` or `step` must be supplied; you can also supply both.
- Each `(epoch, step)` combination must be unique within a run.
- `metadata` is an arbitrary `dict` of scalar values stored alongside the snapshot in the index. The keys `"epoch"` and `"step"` are reserved.

### `observer.close()`

Finalises the run. Safe to call multiple times. Forgetting to call `close()` raises a `ResourceWarning` at garbage-collection time.

### `NeuroInquisitor.load(log_dir, backend="local", format="hdf5", epochs=None, layers=None) → SnapshotCollection`

Classmethod (no model needed). Optional `epochs` / `layers` arguments restrict the returned collection up front.

### `SnapshotCollection`

A lazy, filterable view over a run. No tensor data is read until you call `by_epoch` or `by_layer`; `select` only composes filter sets.

| Method / property | Returns | Notes |
|-------------------|---------|-------|
| `col.epochs` | `list[int]` | Sorted epoch indices in the current view |
| `col.layers` | `list[str]` | Parameter names in the current view |
| `len(col)` | `int` | Number of snapshots in the current view |
| `col.by_epoch(epoch)` | `dict[str, np.ndarray]` | All (filtered) layers for one epoch |
| `col.by_layer(name, max_workers=8)` | `dict[int, np.ndarray]` | One layer across all (filtered) epochs, read in parallel |
| `col.select(epochs=…, layers=…)` | `SnapshotCollection` | Narrow the view; composes with existing filters |

`epochs` accepts `int | list[int] | range`; `layers` accepts `str | list[str]`.

## Examples

The [`examples/`](./examples) directory walks through every part of the API. All examples write their output under `outputs/<example_name>/<timestamp>/` so successive runs don't collide.

### `examples/basic_usage.py`
Train a tiny MLP for 30 epochs, snapshot weights each epoch, then render a GIF of the FC layer heatmaps over time. Good first read after the quick start.

### `examples/store_epochs.py`
The minimal **write** path: epoch-keyed snapshots with `metadata={"loss": ..., "lr": ...}` and gzip compression. Lists the resulting `epoch_XXXX.h5` files and prints the metadata back from the index.

### `examples/store_steps.py`
Snapshot at intra-epoch **step** granularity. Shows the `observer.snapshot(step=...)` form (and the optional `epoch=..., step=...` combination) for cases where one epoch spans thousands of batches.

### `examples/append_run.py`
Resume a run with `create_new=False`. Two training "sessions" write to the same `log_dir`; the second session adds epochs 5–9 alongside the first session's 0–4 without disturbing them. Demonstrates that `NeuroInquisitor.load(log_dir)` sees the combined run.

### `examples/load_and_filter.py`
A tour of every read pattern on `SnapshotCollection`:

1. `NeuroInquisitor.load(log_dir)` — load everything (lazy, just `index.json`).
2. `col.by_epoch(N)` — all layers at one epoch.
3. `col.by_layer("fc1.weight")` — one layer across all epochs (parallel reads).
4. `NeuroInquisitor.load(log_dir, epochs=range(7, 10))` — filter epochs at load time.
5. `NeuroInquisitor.load(log_dir, layers="fc1.weight")` — filter layers at load time.
6. Combine both filters in `load()`.
7. Refine an existing collection with chained `col.select(...)` calls.

### `examples/snapshot_selection.py`
Trains a TinyMLP for 40 epochs, then renders five labelled GIFs using different `col.select(...)` slices (all, early, late, single layer, late × single layer). The GIF helper pre-fetches all needed tensors with `col.by_layer()` for one parallel read per layer, then animates from an in-memory cache with zero further I/O — a good template for visualisation tooling.

### `examples/mnist_example.py`
A real training loop: a small CNN on MNIST for 100 epochs with `tqdm` progress bars, snapshotting after every epoch with `{"loss", "test_loss", "accuracy"}` metadata. After training it loads the full collection with `NeuroInquisitor.load(run_dir)` and produces both an MP4 of the conv/FC weights evolving over time and a train/test loss curve PNG. Run names are generated with `petname` so each run lives in its own directory under `outputs/MNIST_example/<run-name>/`.

Run any example with:

```bash
python examples/basic_usage.py
python examples/mnist_example.py    # needs: pip install tqdm petname torchvision
```

## On-disk layout

A run directory looks like this:

```
runs/my_run/
├── index.json
├── epoch_0000.h5
├── epoch_0001.h5
├── ...
└── epoch_0099.h5
```

- One file per snapshot makes append-mode trivial and lets `by_layer` issue independent reads in parallel.
- `index.json` caches each snapshot's epoch, step, file key, layer names, and metadata, so listing what's in a run (`col.epochs`, `col.layers`, `len(col)`) does **no** HDF5 I/O at all.

## Extending: backends and formats

Both `Backend` (where bytes live) and `Format` (how a snapshot is serialised) are abstract base classes. The built-ins are `LocalBackend` (filesystem) and `HDF5Format`. Implement either ABC and pass an instance to `NeuroInquisitor(backend=..., format=...)` or `NeuroInquisitor.load(backend=..., format=...)` to plug in object storage, a different file format, etc.

## Development

Run tests:

```bash
pytest
```

Run lint and type checks:

```bash
ruff check src tests
mypy src
```

## License

MIT
