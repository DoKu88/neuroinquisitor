# neuroinquisitor

Neural network weight observability for PyTorch — snapshot, persist, and reload model weights during training.

## Installation

```bash
pip install neuroinquisitor
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
    log_dir="./runs",
    filename="weights.h5",
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

## Loading saved weights

```python
observer = NeuroInquisitor(model, log_dir="./runs", filename="weights.h5", create_new=False)
weights = observer.load_snapshot(epoch=5)   # dict[str, np.ndarray]
observer.close()

for name, arr in weights.items():
    print(name, arr.shape)
```

## API reference

### `NeuroInquisitor(model, log_dir, filename, compress, create_new)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | `nn.Module` whose parameters are snapshotted |
| `log_dir` | `"."` | Directory for the HDF5 file |
| `filename` | `"weights.h5"` | HDF5 filename |
| `compress` | `False` | Store datasets with gzip compression |
| `create_new` | `True` | `True` → new file (raises if exists); `False` → append to existing |

### `snapshot(epoch, step, metadata)`

Writes all current model parameters to HDF5 and flushes to disk. At least one of `epoch` or `step` must be supplied. `metadata` is an optional `dict` of scalar values stored as HDF5 attributes.

### `load_snapshot(epoch) → dict[str, np.ndarray]`

Reads back a previously written snapshot by epoch number.

### `close()`

Finalizes and closes the HDF5 file. Safe to call multiple times.

## Running the example

```bash
python examples/basic_usage.py
```

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
