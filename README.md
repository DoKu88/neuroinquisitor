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
from neuroinquisitor import NeuroInquisitor

observer = NeuroInquisitor(
    model,
    log_dir="./runs",
    filename="weights.h5",
    compress=True,
    create_new=True,
)

for epoch in range(num_epochs):
    # ... training loop ...
    observer.snapshot(epoch=epoch, metadata={"lr": lr})

observer.close()
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
