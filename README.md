# neuroinquisitor

Neural network weight observability for PyTorch — snapshot, persist, and reload model weights during training, then analyse how they evolve.

- **One observer, one line per snapshot.** Attach `NeuroInquisitor` to any `nn.Module` and call `.snapshot(epoch=…)` or `.snapshot(step=…)` inside your training loop.
- **One file per snapshot, plus an index.** Each snapshot is an independent HDF5 file; an `index.json` tracks epochs, steps, layer names, and per-snapshot metadata. Append more snapshots to a run later without rewriting anything.
- **Lazy, filterable read-back.** `NeuroInquisitor.load(...)` returns a `SnapshotCollection` that touches no tensor data until you ask for it — and lets you slice by epoch, layer, or both, with parallel reads under the hood.
- **Built-in analyzers.** Five standalone functions (`probe`, `projection_embed`, `cosine_similarity_matrix`, `spectral_summary`, `weight_trajectory`) accept NumPy/PyTorch tensors and return a plain `pd.DataFrame`.
- **Pluggable storage and format.** Local filesystem + HDF5 are built in; both the `Backend` and `Format` are abstract so you can drop in your own (object stores, Zarr, Safetensors, etc.).

## Installation

Install the core library:

```bash
pip install neuroinquisitor
```

To run **all examples** (includes matplotlib, torchvision, captum, torchlens, transformer-lens, fiftyone, tensorboard, scikit-learn, and more):

```bash
pip install "neuroinquisitor[examples]"
```

For development (adds pytest, ruff, mypy):

```bash
git clone https://github.com/doku88/neuroinquisitor.git
cd neuroinquisitor
pip install -e ".[dev,examples]"
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

`NeuroInquisitor.load()` does **not** require a model — it is a pure read path for post-training analysis. No tensor data is read until you ask for it.

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

Filters can also be pushed into `load()` itself:

```python
late = NeuroInquisitor.load("./runs/quickstart", epochs=range(5, 10))
fc1_only = NeuroInquisitor.load("./runs/quickstart", layers="0.weight")
```

## Built-in analyzers

```python
from neuroinquisitor.analyzers import (
    probe,
    projection_embed,
    cosine_similarity_matrix,
    spectral_summary,
    weight_trajectory,
)
```

Each function accepts a NumPy array or PyTorch tensor and returns a `pd.DataFrame`.

| Function | What it computes |
|---|---|
| `probe(X, y)` | Linear probe accuracy and weights for an activation matrix |
| `projection_embed(X)` | 2-D/3-D coordinates via PCA, t-SNE, or UMAP |
| `cosine_similarity_matrix(X)` | Pairwise cosine similarity between rows |
| `spectral_summary(W)` | Top singular values and explained variance of a weight matrix |
| `weight_trajectory(snapshots)` | L2 distance, cosine similarity, and norm over training |

## Replay sessions

`ReplaySession` restores a checkpoint into a model and runs a forward (and optional backward) pass, capturing activations, gradients, or raw logits without touching the original training state.

```python
from neuroinquisitor import NeuroInquisitor, ReplaySession

col = NeuroInquisitor.load("./runs/quickstart")
session = ReplaySession(model, col)

result = session.run(epoch=5, dataset=my_dataset, capture=["activations", "logits"])
result.activations   # dict[str, torch.Tensor]
result.logits        # torch.Tensor  (N, num_classes)
```

## API reference

### `NeuroInquisitor(model, log_dir, compress, create_new, backend, format)`

| Parameter | Default | Description |
|---|---|---|
| `model` | — | `nn.Module` whose parameters are snapshotted |
| `log_dir` | `"."` | Directory for snapshot files and `index.json` |
| `compress` | `False` | Enable gzip compression in HDF5 snapshots |
| `create_new` | `True` | `True` → fresh run, errors if index exists. `False` → append to existing run, errors if no index found. |
| `backend` | `"local"` | Storage backend — `"local"` or a `Backend` instance |
| `format` | `"hdf5"` | Snapshot file format — `"hdf5"` or a `Format` instance |

### `observer.snapshot(epoch=None, step=None, metadata=None)`

Writes all current model parameters to a new snapshot file and updates the index.

- At least one of `epoch` or `step` must be supplied; both may be supplied together.
- Each `(epoch, step)` combination must be unique within a run.
- `metadata` is an arbitrary `dict` of scalar values stored alongside the snapshot in the index. The keys `"epoch"` and `"step"` are reserved.

### `observer.close()`

Finalises the run. Safe to call multiple times. Forgetting to call `close()` raises a `ResourceWarning` at garbage-collection time.

### `NeuroInquisitor.load(log_dir, backend, format, epochs, layers) → SnapshotCollection`

Classmethod — no model needed. Optional `epochs` / `layers` arguments restrict the returned collection up front.

### `SnapshotCollection`

A lazy, filterable view over a run. No tensor data is read until you call `by_epoch` or `by_layer`; `select` only composes filter sets.

| Method / property | Returns | Notes |
|---|---|---|
| `col.epochs` | `list[int]` | Sorted epoch indices in the current view |
| `col.layers` | `list[str]` | Parameter names in the current view |
| `len(col)` | `int` | Number of snapshots in the current view |
| `col.by_epoch(epoch)` | `dict[str, np.ndarray]` | All (filtered) layers for one epoch |
| `col.by_layer(name, max_workers=8)` | `dict[int, np.ndarray]` | One layer across all snapshots, read in parallel |
| `col.select(epochs=…, layers=…)` | `SnapshotCollection` | Narrow the view; composes with existing filters |
| `col.to_state_dict(epoch)` | `dict[str, torch.Tensor]` | Ready to pass to `model.load_state_dict()` |
| `col.to_numpy(epoch)` | `dict[str, np.ndarray]` | Same as `by_epoch`, explicit alias |

`epochs` accepts `int | list[int] | range`; `layers` accepts `str | list[str]`.

## Examples

Install the examples extra first, then run any script directly or run all of them at once:

```bash
pip install "neuroinquisitor[examples]"

# Run everything in order (writes a YAML status log to outputs/status_run/)
python examples/run_all.py

# Or run individual examples
python examples/multi_arch_showcase.py
python examples/grokking_example.py
```

### Main demos

#### `examples/multi_arch_showcase.py`
The recommended starting point. Trains three architectures — `TinyMLP`, `SmallCNN`, and `TinyTransformer` — on the same synthetic sequence classification task (no dataset downloads). All three use an identical NI setup, demonstrating that the API is architecture-agnostic.

NI features covered: `CapturePolicy`, `RunMetadata`, `snapshot()`, `SnapshotCollection` (`by_epoch`, `by_layer`, `select`, `to_state_dict`), `ReplaySession`.

```bash
python examples/multi_arch_showcase.py
```

#### `examples/grokking_example.py`
A grokking-style modular addition experiment with a 1-layer transformer, tracking 15 000 optimisation steps. Demonstrates step-based snapshotting and shows the phase transition (memorisation → generalisation) as a visible discontinuity in weight space. Outputs a four-panel MP4 (embedding similarity, Fourier power timeline, component norms, output projection) plus an accuracy-curve PNG.

```bash
python examples/grokking_example.py
```

---

### Integration examples

#### `examples/captum_use_examples/grokking_captum.py`
Shows two integration paths between NI checkpoints and Captum attribution:

- **Path A** — `col.to_state_dict(epoch)` → `model.load_state_dict()` → Captum
- **Path B** — `ReplaySession.run()` → `result.activations` → Captum

Attribution methods demonstrated: `LayerIntegratedGradients` on the token embedding, `LayerConductance` on the transformer encoder — both tracked over every checkpoint so you can watch attribution evolve across the grokking phase transition.

```bash
python examples/captum_use_examples/grokking_captum.py
```

#### `examples/torchlens_use_examples/torchlens_cifar10.py`
Combines NI with TorchLens to produce per-operation activation analysis at every training epoch — a film strip of `log_forward_pass` output rather than a single snapshot. Trains a CNN on CIFAR-10 and produces activation-evolution and gradient-flow visualisations over time.

```bash
python examples/torchlens_use_examples/torchlens_cifar10.py
```

#### `examples/transformerlens_use_examples/cifar10_transformerlens.py`
Trains a patch-based Vision Transformer (`PatchViT`) on CIFAR-10 with NI snapshots, then runs five TransformerLens analyses across all checkpoints: activation caching, attention pattern evolution, logit lens, activation patching, and weight SVD over training.

```bash
python examples/transformerlens_use_examples/cifar10_transformerlens.py
```

---

### Specific-action examples (`examples/specific_actions/`)

Focused, single-concept scripts — good templates to copy from.

| Script | What it shows |
|---|---|
| `store_epochs.py` | Minimal epoch-keyed write path with metadata and compression |
| `store_steps.py` | Intra-epoch step-keyed snapshots; `epoch + step` combined keys |
| `append_run.py` | Resume a run with `create_new=False`; two sessions, one `log_dir` |
| `load_and_filter.py` | Every `SnapshotCollection` access pattern in one file |
| `snapshot_selection.py` | Five `col.select()` slices rendered as labelled GIFs |
| `replay_logits.py` | `ReplaySession` with `capture=["logits"]`; useful for comparing pre/post fine-tune |
| `track_activations.py` | `ReplaySession` with `capture=["activations"]`; `activation_reduction` options |
| `track_gradients.py` | `ReplaySession` with `capture=["gradients"]`; sensitivity analysis |
| `tracin_example.py` | Feed NI checkpoints into Captum `TracInCP` for influence estimation |
| `tensorboard_projector.py` | Convert `projection_embed` output to TensorBoard Projector TSV format |
| `fiftyone_embed.py` | Attach `projection_embed` coordinates to a FiftyOne dataset |

```bash
python examples/specific_actions/store_epochs.py
python examples/specific_actions/track_activations.py
# etc.
```

---

## On-disk layout

```
runs/my_run/
├── index.json
├── epoch_0000.h5
├── epoch_0001.h5
├── epoch_0002_step_000250.h5
├── step_000251.h5
└── epoch_0099.h5
```

- One file per snapshot makes append-mode trivial and lets `by_layer` issue independent reads in parallel.
- `index.json` caches each snapshot's epoch, step, file key, layer names, and metadata, so listing what's in a run (`col.epochs`, `col.layers`, `len(col)`) does **no** HDF5 I/O at all.

## Extending: backends and formats

Both `Backend` (where bytes live) and `Format` (how a snapshot is serialised) are abstract base classes. The built-ins are `LocalBackend` (filesystem) and `HDF5Format`. Implement either ABC and pass an instance to `NeuroInquisitor(backend=..., format=...)` or `NeuroInquisitor.load(backend=..., format=...)` to plug in object storage, a different file format, etc.

## Development

```bash
# Run tests
pytest

# Lint and type checks
ruff check src tests
mypy src
```

## License

MIT
