"""TracIn quick-start: feed NI checkpoints into Captum TracInCP (NI-GAMMA-006).

Usage:
    pip install captum
    python examples/specific_actions/tracin_example.py
"""

from __future__ import annotations

import pathlib
import tempfile

try:
    from captum.influence import TracInCP  # type: ignore[import-untyped]
except ImportError:
    raise SystemExit("Install captum first: pip install captum")

import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.loader import load as ni_load


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


torch.manual_seed(0)
model = Net()
X, y = torch.randn(20, 4), torch.randint(0, 2, (20,))
dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
    torch.utils.data.TensorDataset(X, y)
)
run_dir = pathlib.Path(tempfile.mkdtemp())

obs = NeuroInquisitor(model, log_dir=run_dir)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for ep in range(3):
    nn.CrossEntropyLoss()(model(X), y).backward()
    opt.step()
    opt.zero_grad()
    obs.snapshot(epoch=ep)
obs.close()

# -- Load NI checkpoints and save as .pt files for TracInCP --
col = ni_load(run_dir)
ckpt_paths: list[str] = []
for ep in col.epochs:
    p = run_dir / f"ckpt_{ep}.pt"
    torch.save(col.to_state_dict(ep), str(p))
    ckpt_paths.append(str(p))


def _load_ckpt(net: nn.Module, path: str) -> nn.Module:
    net.load_state_dict(torch.load(path, weights_only=True))
    return net


tracin = TracInCP(
    model=Net(),
    train_dataset=dataset,
    checkpoints=ckpt_paths,
    checkpoints_load_func=_load_ckpt,
    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    batch_size=5,
)

scores, indices = tracin.influence((X[:1], y[:1]), top_k=3)
print("Query: sample 0")
print("Top-3 proponent training indices:", indices)
print("Influence scores:", scores)
