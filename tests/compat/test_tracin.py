"""TracIn / Captum compatibility test (NI-GAMMA-006).

Verifies the TracIn quick-start pattern: NI checkpoints → to_state_dict →
.pt files → TracInCP end-to-end on a toy classification model.

Skip condition: captum not installed.
Install:        pip install captum

Run standalone:
    pytest tests/compat/test_tracin.py -v
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.loader import load as ni_load

captum_influence = pytest.importorskip(
    "captum.influence",
    reason=(
        "captum is not installed — TracIn compatibility test skipped.\n"
        "To run this test: pip install captum"
    ),
)


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture()
def toy_run(tmp_path: pathlib.Path) -> tuple[pathlib.Path, Callable[[], nn.Module]]:
    torch.manual_seed(0)
    model = TinyNet()
    X, y = torch.randn(20, 4), torch.randint(0, 2, (20,))
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for ep in range(3):
        nn.CrossEntropyLoss()(model(X), y).backward()
        opt.step()
        opt.zero_grad()
        obs.snapshot(epoch=ep)
    obs.close()
    return tmp_path, TinyNet


def test_tracin_end_to_end(toy_run: tuple[pathlib.Path, Callable[[], nn.Module]]) -> None:
    """NI checkpoints → to_state_dict → TracInCP influence scores."""
    TracInCP = captum_influence.TracInCP

    run_dir, factory = toy_run
    torch.manual_seed(1)
    X, y = torch.randn(20, 4), torch.randint(0, 2, (20,))
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )

    col = ni_load(run_dir)
    ckpt_paths: list[str] = []
    for ep in col.epochs:
        p = run_dir / f"ckpt_{ep}.pt"
        torch.save(col.to_state_dict(ep), str(p))
        ckpt_paths.append(str(p))

    def _load(net: nn.Module, path: str) -> nn.Module:
        net.load_state_dict(torch.load(path, weights_only=True))
        return net

    tracin = TracInCP(
        model=factory(),
        train_dataset=dataset,
        checkpoints=ckpt_paths,
        checkpoints_load_func=_load,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        batch_size=5,
    )

    scores, indices = tracin.influence((X[:1], y[:1]), top_k=3)
    assert indices.shape == (1, 3)
    assert scores.shape == (1, 3)
    # No NI types in the output — plain tensors
    assert isinstance(scores, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
