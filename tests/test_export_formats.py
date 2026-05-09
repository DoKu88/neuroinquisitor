"""Tests for standard tensor export surface (NI-DELTA-001).

Verifies:
- SnapshotCollection.to_state_dict returns only standard torch.Tensor values
- SnapshotCollection.to_numpy returns only plain np.ndarray values
- model.load_state_dict(col.to_state_dict(epoch)) produces identical forward pass
- ReplayResult.to_numpy returns plain np.ndarray values
- ReplayResult.activations / .gradients are plain dict[str, torch.Tensor]
- Activation dict keys/shapes match what register_forward_hook produces
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.utils.data

from neuroinquisitor import NeuroInquisitor, TensorMap
from neuroinquisitor.collection import SnapshotCollection
from neuroinquisitor.loader import load as ni_load
from neuroinquisitor.replay import ReplayResult, ReplaySession


# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def multi_epoch_run(tmp_path: Path) -> tuple[Path, list[dict[str, torch.Tensor]]]:
    """Run with 3 snapshots; returns the path and original state dicts."""
    torch.manual_seed(0)
    model = TinyMLP()
    obs = NeuroInquisitor(model, log_dir=tmp_path)
    state_dicts = []
    for epoch in range(3):
        for p in model.parameters():
            p.data += 0.01
        obs.snapshot(epoch=epoch)
        state_dicts.append({k: v.clone() for k, v in model.state_dict().items()})
    obs.close()
    return tmp_path, state_dicts


@pytest.fixture()
def mlp_dataloader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, ...]]:
    torch.manual_seed(99)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    ds: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]] = (
        torch.utils.data.TensorDataset(X, y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, persistent_workers=True)


# ---------------------------------------------------------------------------
# to_state_dict — return type
# ---------------------------------------------------------------------------


def test_to_state_dict_returns_only_tensors(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    sd = col.to_state_dict(epoch=0)
    assert isinstance(sd, dict)
    for v in sd.values():
        assert type(v) is torch.Tensor, f"Expected torch.Tensor, got {type(v)}"


def test_to_state_dict_no_ni_types(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    sd = col.to_state_dict(epoch=1)
    for k, v in sd.items():
        assert not k.startswith("neuroinquisitor.")
        assert type(v).__module__.split(".")[0] == "torch"


# ---------------------------------------------------------------------------
# to_state_dict — round-trip correctness
# ---------------------------------------------------------------------------


def test_load_state_dict_round_trip(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    """model.load_state_dict(col.to_state_dict(epoch=N)) must reproduce original outputs."""
    run_dir, original_state_dicts = multi_epoch_run
    col = ni_load(run_dir)
    torch.manual_seed(42)
    x = torch.randn(4, 4)

    for epoch, orig_sd in enumerate(original_state_dicts):
        # Compute reference output using original weights
        ref_model = TinyMLP()
        ref_model.load_state_dict(orig_sd)
        ref_model.eval()
        with torch.no_grad():
            ref_out = ref_model(x)

        # Compute output after loading from NI
        loaded_model = TinyMLP()
        loaded_model.load_state_dict(col.to_state_dict(epoch=epoch))
        loaded_model.eval()
        with torch.no_grad():
            loaded_out = loaded_model(x)

        torch.testing.assert_close(ref_out, loaded_out)


# ---------------------------------------------------------------------------
# to_numpy — return type
# ---------------------------------------------------------------------------


def test_to_numpy_returns_only_ndarrays(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    arrays = col.to_numpy(epoch=0)
    assert isinstance(arrays, dict)
    for v in arrays.values():
        assert isinstance(v, np.ndarray), f"Expected np.ndarray, got {type(v)}"


def test_to_numpy_layer_filter(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    arrays = col.to_numpy(epoch=0, layers=["fc1.weight"])
    assert list(arrays.keys()) == ["fc1.weight"]


def test_to_numpy_no_ni_types(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
) -> None:
    run_dir, _ = multi_epoch_run
    col = ni_load(run_dir)
    arrays = col.to_numpy(epoch=2)
    for v in arrays.values():
        assert type(v).__module__.split(".")[0] == "numpy"


# ---------------------------------------------------------------------------
# ReplayResult — TensorMap container + plain tensors inside
# ---------------------------------------------------------------------------


def test_replay_result_activations_is_tensor_map(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations"],
    ).run()

    assert isinstance(result.activations, TensorMap)
    assert isinstance(result.activations, dict)  # transparent dict subclass


def test_replay_result_gradients_is_tensor_map(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
    ).run()

    assert isinstance(result.gradients, TensorMap)
    assert isinstance(result.gradients, dict)


def test_replay_result_activations_values_are_plain_tensors(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations"],
    ).run()

    for v in result.activations.values():
        assert type(v) is torch.Tensor


def test_replay_result_gradients_values_are_plain_tensors(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
    ).run()

    for v in result.gradients.values():
        assert type(v) is torch.Tensor


# ---------------------------------------------------------------------------
# Activation keys match register_forward_hook layout
# ---------------------------------------------------------------------------


def test_activation_keys_match_forward_hook(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """NI replay activation keys and shapes must match a manual forward hook."""
    run_dir, original_state_dicts = multi_epoch_run

    modules_to_capture = ["fc1", "fc2"]

    # --- NI replay ---
    ni_result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=modules_to_capture,
        capture=["activations"],
        activation_reduction="raw",
    ).run()

    # --- Manual forward hook ---
    manual_hook_outputs: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str) -> Callable:
        def hook(_m: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            if name not in manual_hook_outputs:
                manual_hook_outputs[name] = out.detach()
            else:
                manual_hook_outputs[name] = torch.cat(
                    [manual_hook_outputs[name], out.detach()], dim=0
                )
        return hook

    hook_model = TinyMLP()
    hook_model.load_state_dict(original_state_dicts[0])
    hook_model.eval()

    for name, module in hook_model.named_modules():
        if name in modules_to_capture:
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in mlp_dataloader:
            hook_model(batch[0])

    for h in handles:
        h.remove()

    # Keys must match
    assert set(ni_result.activations.keys()) == set(manual_hook_outputs.keys())

    # Shapes must match
    for name in modules_to_capture:
        ni_shape = ni_result.activations[name].shape
        manual_shape = manual_hook_outputs[name].shape
        assert ni_shape == manual_shape, (
            f"Shape mismatch for {name!r}: NI={ni_shape}, hook={manual_shape}"
        )


# ---------------------------------------------------------------------------
# TensorMap.to_numpy — activations and gradients
# ---------------------------------------------------------------------------


def test_activations_to_numpy_returns_ndarrays(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations"],
    ).run()

    numpy_dict = result.activations.to_numpy()
    assert isinstance(numpy_dict, dict)
    for k, v in numpy_dict.items():
        assert isinstance(v, np.ndarray), f"{k}: expected np.ndarray, got {type(v)}"


def test_gradients_to_numpy_returns_ndarrays(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
    ).run()

    numpy_dict = result.gradients.to_numpy()
    assert isinstance(numpy_dict, dict)
    for k, v in numpy_dict.items():
        assert isinstance(v, np.ndarray), f"{k}: expected np.ndarray, got {type(v)}"


def test_activations_to_numpy_keys_match(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1", "fc2"],
        capture=["activations"],
    ).run()

    numpy_dict = result.activations.to_numpy()
    assert set(numpy_dict.keys()) == set(result.activations.keys())


def test_activations_to_numpy_values_match(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    ).run()

    numpy_dict = result.activations.to_numpy()
    np.testing.assert_array_equal(numpy_dict["fc1"], result.activations["fc1"].numpy())


def test_gradients_to_numpy_values_match(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["gradients"],
    ).run()

    numpy_dict = result.gradients.to_numpy()
    np.testing.assert_array_equal(numpy_dict["fc1"], result.gradients["fc1"].numpy())


def test_tensor_map_to_numpy_returns_plain_dict(
    multi_epoch_run: tuple[Path, list[dict[str, torch.Tensor]]],
    mlp_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
) -> None:
    """to_numpy() returns a plain dict, not a TensorMap."""
    run_dir, _ = multi_epoch_run
    result = ReplaySession(
        run=run_dir,
        checkpoint=0,
        model_factory=TinyMLP,
        dataloader=mlp_dataloader,
        modules=["fc1"],
        capture=["activations"],
    ).run()

    numpy_dict = result.activations.to_numpy()
    assert type(numpy_dict) is dict  # plain dict, not TensorMap
