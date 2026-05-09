"""TransformerLens compatibility tests (NI-DELTA-007).

Verifies that NI ReplaySession activation output structure is compatible with
TransformerLens hook output format (Dict[str, torch.Tensor] keyed by module
name).  Flags any key-naming divergence before Sprint 8.

Skip condition: transformer_lens not installed.
Install:        pip install transformer-lens

Run standalone:
    pytest tests/compat/test_transformerlens.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.replay import ReplaySession

pytest.importorskip(
    "transformer_lens",
    reason=(
        "transformer_lens is not installed — "
        "TransformerLens compatibility tests skipped.\n"
        "To run these tests: pip install transformer-lens"
    ),
)


# ---------------------------------------------------------------------------
# Shared tiny transformer-style model
# ---------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """Minimal transformer-style model for compatibility checks.

    Uses standard PyTorch MultiheadAttention so TransformerLens hook naming
    conventions can be compared against NI module names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.attn = nn.MultiheadAttention(8, num_heads=2, batch_first=True)
        self.out = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(1))
        h, _ = self.attn(h, h, h)
        return self.out(h.squeeze(1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def transformer_run(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    model = TinyTransformer()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    obs.snapshot(epoch=0)
    obs.close()
    return tmp_path, {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture()
def small_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(10)
    X = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ni_activation_dict_is_plain_dict_of_tensors(
    transformer_run: tuple[Path, dict[str, torch.Tensor]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """NI produces Dict[str, torch.Tensor] — same contract as TransformerLens hooks.

    TransformerLens ActivationCache is a dict-like mapping hook names to
    torch.Tensor.  NI must produce the same fundamental contract so they
    are interchangeable in downstream analysis code.
    """
    run_dir, _ = transformer_run

    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyTransformer,
        dataloader=small_dataloader,
        modules=["embed", "out"],
        capture=["activations"],
        activation_reduction="raw",
    ).run()

    assert isinstance(result.activations, dict)
    for k, v in result.activations.items():
        assert type(v) is torch.Tensor, (
            f"Key {k!r}: expected torch.Tensor, got {type(v).__name__}. "
            "TransformerLens hook outputs are plain tensors; NI must match."
        )


def test_ni_activation_tensors_are_at_least_2d(
    transformer_run: tuple[Path, dict[str, torch.Tensor]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """NI activation tensors are ≥2-D, consistent with TransformerLens hook outputs."""
    run_dir, _ = transformer_run

    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyTransformer,
        dataloader=small_dataloader,
        modules=["embed", "out"],
        capture=["activations"],
        activation_reduction="raw",
    ).run()

    for k, v in result.activations.items():
        assert v.ndim >= 2, (
            f"Key {k!r}: got shape {v.shape} ({v.ndim}-D). "
            "TransformerLens outputs are always ≥2-D (batch × features)."
        )


def test_ni_key_naming_uses_module_name(
    transformer_run: tuple[Path, dict[str, torch.Tensor]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """NI uses the exact module name passed to modules=[...] as the activation key.

    TransformerLens uses its own hook naming scheme (e.g. 'hook_embed').
    This test documents the known naming divergence and confirms NI's behaviour
    is deterministic: the key is always the string passed to modules=[...].

    Downstream code bridging NI and TransformerLens must map keys accordingly.
    """
    run_dir, _ = transformer_run

    requested_modules = ["embed", "out"]

    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyTransformer,
        dataloader=small_dataloader,
        modules=requested_modules,
        capture=["activations"],
        activation_reduction="raw",
    ).run()

    ni_keys = set(result.activations.keys())
    expected_keys = set(requested_modules)

    assert ni_keys == expected_keys, (
        f"NI key naming diverges from requested module names.\n"
        f"  Requested : {sorted(expected_keys)}\n"
        f"  Got       : {sorted(ni_keys)}\n"
        "NI always uses the module name string as the key.  "
        "TransformerLens uses hook names (e.g. 'hook_embed').  "
        "Bridge code should remap NI keys to hook names when needed."
    )


def test_ni_activation_to_numpy_in_transformerlens_workflow(
    transformer_run: tuple[Path, dict[str, torch.Tensor]],
    small_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """TensorMap.to_numpy() works in a TransformerLens-style analysis loop."""
    run_dir, _ = transformer_run

    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyTransformer,
        dataloader=small_dataloader,
        modules=["embed", "out"],
        capture=["activations"],
        activation_reduction="raw",
    ).run()

    # TransformerLens workflows often convert to numpy for sklearn / scipy.
    # This should work identically whether the source is NI or TransformerLens.
    numpy_acts = result.activations.to_numpy()
    assert isinstance(numpy_acts, dict)
    for k, v in numpy_acts.items():
        assert isinstance(v, np.ndarray), f"{k}: expected np.ndarray"
        assert v.ndim >= 2
