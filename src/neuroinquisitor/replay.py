"""Replay-based activation and gradient capture (NI-BETA-001 through NI-BETA-004)."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.formats.base import Format
from neuroinquisitor.loader import load as _load_collection

# ---------------------------------------------------------------------------
# Dataset slice models — NI-BETA-004
# ---------------------------------------------------------------------------


class FirstNSlice(BaseModel):
    """Select the first N samples from the iterable."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["first_n"] = "first_n"
    n: int = Field(gt=0)


class RandomNSlice(BaseModel):
    """Select N samples at random with a fixed seed."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["random_n"] = "random_n"
    n: int = Field(gt=0)
    seed: int


class BalancedNSlice(BaseModel):
    """Select N samples with equal representation per class.

    Requires that each dataloader batch yields ``(inputs, labels)``.
    Labels may be class-index scalars or one-hot vectors.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["balanced_n"] = "balanced_n"
    n: int = Field(gt=0)
    seed: int


class ExplicitIndicesSlice(BaseModel):
    """Select specific sample indices (flat, across all batches in order)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["explicit"] = "explicit"
    indices: list[int] = Field(min_length=1)


DatasetSlice: TypeAlias = Annotated[
    Union[FirstNSlice, RandomNSlice, BalancedNSlice, ExplicitIndicesSlice],
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Checkpoint selector
# ---------------------------------------------------------------------------


class CheckpointSelector(BaseModel):
    """Identifies a snapshot by its epoch index."""

    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Serialisable replay config — NI-BETA-001
# ---------------------------------------------------------------------------


class ReplayConfig(BaseModel):
    """Serialisable configuration for a ReplaySession.

    Captures everything needed to reproduce a replay except the
    non-serialisable callables (model_factory, dataloader).
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointSelector
    modules: list[str] = Field(min_length=1)
    capture: list[Literal["activations", "gradients", "logits"]] = Field(
        min_length=1
    )
    activation_reduction: Literal["raw", "mean", "pool"] = "raw"
    gradient_mode: Literal["per_example", "aggregated"] = "aggregated"
    dataset_slice: (
        Annotated[
            Union[FirstNSlice, RandomNSlice, BalancedNSlice, ExplicitIndicesSlice],
            Field(discriminator="kind"),
        ]
        | None
    ) = None


# ---------------------------------------------------------------------------
# Replay metadata (provenance) — NI-BETA-004
# ---------------------------------------------------------------------------


class ReplayMetadata(BaseModel):
    """Provenance metadata attached to a completed replay."""

    model_config = ConfigDict(extra="forbid")

    run: str
    checkpoint_epoch: int
    modules: list[str]
    capture: list[str]
    activation_reduction: str
    gradient_mode: str
    dataset_slice: dict[str, Any] | None = None
    n_samples: int
    artifact_sizes: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Output of :meth:`ReplaySession.run`.

    ``activations`` and ``gradients`` are plain ``dict[str, torch.Tensor]``
    keyed by module name — identical layout to what
    ``register_forward_hook`` / ``register_full_backward_hook`` would
    produce directly.  ``logits`` is a plain ``torch.Tensor``.
    """

    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    logits: torch.Tensor | None = None
    metadata: ReplayMetadata | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _label_to_int(label: torch.Tensor) -> int:
    if label.ndim == 0 or label.numel() == 1:
        return int(label.item())
    return int(label.argmax().item())


def _apply_slice(
    batches: list[tuple[torch.Tensor, ...]],
    slice_config: (
        FirstNSlice | RandomNSlice | BalancedNSlice | ExplicitIndicesSlice | None
    ),
) -> tuple[list[tuple[torch.Tensor, ...]], int]:
    """Apply a dataset slice to a list of collected batches.

    Returns ``(selected_batches, n_samples)`` where *selected_batches*
    contains a single re-stacked batch.
    """
    if not batches:
        return batches, 0

    if slice_config is None:
        return batches, sum(b[0].shape[0] for b in batches)

    n_tensors = len(batches[0])

    # Flatten to individual samples across all batches.
    flat: list[tuple[torch.Tensor, ...]] = []
    for batch in batches:
        batch_size = batch[0].shape[0]
        for i in range(batch_size):
            flat.append(tuple(b[i] for b in batch[:n_tensors]))

    if isinstance(slice_config, FirstNSlice):
        selected = flat[: slice_config.n]

    elif isinstance(slice_config, RandomNSlice):
        rng = random.Random(slice_config.seed)
        k = min(slice_config.n, len(flat))
        selected = rng.sample(flat, k)

    elif isinstance(slice_config, BalancedNSlice):
        if n_tensors < 2:
            raise ValueError(
                "BalancedNSlice requires batches of the form (inputs, labels)."
            )
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(flat):
            groups[_label_to_int(sample[1])].append(idx)
        n_classes = len(groups)
        per_class = max(1, slice_config.n // n_classes)
        rng = random.Random(slice_config.seed)
        indices: list[int] = []
        for class_indices in groups.values():
            k = min(per_class, len(class_indices))
            indices.extend(rng.sample(class_indices, k))
        rng.shuffle(indices)
        selected = [flat[i] for i in indices[: slice_config.n]]

    elif isinstance(slice_config, ExplicitIndicesSlice):
        out_of_range = [i for i in slice_config.indices if i >= len(flat)]
        if out_of_range:
            raise ValueError(
                f"Explicit indices {out_of_range} are out of range "
                f"for dataset of {len(flat)} samples."
            )
        selected = [flat[i] for i in slice_config.indices]

    else:
        selected = flat

    if not selected:
        raise ValueError("Dataset slice produced an empty selection.")

    stacked: tuple[torch.Tensor, ...] = tuple(
        torch.stack([s[i] for s in selected]) for i in range(n_tensors)
    )
    return [stacked], len(selected)


# ---------------------------------------------------------------------------
# ReplaySession — NI-BETA-001 / NI-BETA-002 / NI-BETA-003
# ---------------------------------------------------------------------------


class ReplaySession:
    """Replay a checkpoint and capture activations, gradients, and logits.

    Parameters
    ----------
    run:
        Path to the NeuroInquisitor run directory containing ``index.json``.
    checkpoint:
        Epoch index of the snapshot to load (int or
        :class:`CheckpointSelector`).
    model_factory:
        Callable returning a freshly-instantiated :class:`~torch.nn.Module`.
        Called once per :meth:`run` invocation; weights are loaded from the
        checkpoint via :meth:`~torch.nn.Module.load_state_dict`.
    dataloader:
        Iterable of batches.  Each batch must be a :class:`torch.Tensor` or
        a tuple/list of :class:`torch.Tensor` objects.
    modules:
        Module names (from ``model.named_modules()``) to capture.
    capture:
        Data kinds to capture: ``"activations"``, ``"gradients"``,
        ``"logits"``.
    activation_reduction:
        ``"raw"`` — full ``(N, ...)`` tensor;
        ``"mean"`` — mean over the batch dim → ``(...)``;
        ``"pool"`` — spatial average pool → ``(N, C)`` for ≥3-D tensors,
        identity for 2-D tensors.
    gradient_mode:
        ``"per_example"`` — full ``(N, ...)`` gradient tensor;
        ``"aggregated"`` — mean over the batch dim → ``(...)``.
    dataset_slice:
        Optional selection policy.  ``None`` uses all batches.
    backend:
        Storage backend for loading snapshots (default ``"local"``).
    format:
        Snapshot format for loading snapshots (default ``"hdf5"``).
    """

    def __init__(
        self,
        run: str | os.PathLike[str],
        checkpoint: int | CheckpointSelector,
        model_factory: Callable[[], nn.Module],
        dataloader: Iterable[Any],
        modules: list[str],
        capture: list[Literal["activations", "gradients", "logits"]],
        activation_reduction: Literal["raw", "mean", "pool"] = "raw",
        gradient_mode: Literal["per_example", "aggregated"] = "aggregated",
        dataset_slice: (
            FirstNSlice | RandomNSlice | BalancedNSlice | ExplicitIndicesSlice | None
        ) = None,
        backend: str | Backend = "local",
        format: str | Format = "hdf5",
    ) -> None:
        self._run_dir = Path(run)
        self._model_factory = model_factory
        self._dataloader = dataloader
        self._backend_spec = backend
        self._format_spec = format

        epoch_int = checkpoint if isinstance(checkpoint, int) else checkpoint.epoch
        self.config = ReplayConfig(
            checkpoint=CheckpointSelector(epoch=epoch_int),
            modules=list(modules),
            capture=list(capture),
            activation_reduction=activation_reduction,
            gradient_mode=gradient_mode,
            dataset_slice=dataset_slice,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        epoch = self.config.checkpoint.epoch
        col = _load_collection(
            self._run_dir,
            backend=self._backend_spec,
            format=self._format_spec,
            epochs=[epoch],
        )
        weights = col.by_epoch(epoch)
        model = self._model_factory()
        state_dict = {
            name: torch.from_numpy(arr.copy()) for name, arr in weights.items()
        }
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _validate_modules(self, model: nn.Module) -> None:
        available = {name for name, _ in model.named_modules() if name}
        invalid = [m for m in self.config.modules if m not in available]
        if invalid:
            raise ValueError(
                f"Invalid module name(s): {invalid}. "
                f"Available modules: {sorted(available)}"
            )

    def _collect_batches(self) -> list[tuple[torch.Tensor, ...]]:
        batches: list[tuple[torch.Tensor, ...]] = []
        for batch in self._dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append((batch,))
            elif isinstance(batch, (list, tuple)):
                batches.append(
                    tuple(b for b in batch if isinstance(b, torch.Tensor))
                )
            else:
                raise TypeError(
                    f"Unsupported batch type {type(batch).__name__}. "
                    "Each batch must be a Tensor or a tuple/list of Tensors."
                )
        return batches

    def _reduce_activation(self, t: torch.Tensor) -> torch.Tensor:
        mode = self.config.activation_reduction
        if mode == "mean":
            return t.mean(dim=0)
        if mode == "pool" and t.ndim > 2:
            return t.flatten(start_dim=2).mean(dim=-1)
        return t

    def _reduce_gradient(self, t: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_mode == "aggregated":
            return t.mean(dim=0)
        return t

    def _run_capture(
        self,
        model: nn.Module,
        batches: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor | None,
    ]:
        do_act = "activations" in self.config.capture
        do_grad = "gradients" in self.config.capture
        do_logits = "logits" in self.config.capture

        all_activations: dict[str, list[torch.Tensor]] = {
            m: [] for m in self.config.modules
        }
        all_gradients: dict[str, list[torch.Tensor]] = {
            m: [] for m in self.config.modules
        }
        all_logits: list[torch.Tensor] = []

        for batch in batches:
            inputs = batch[0]
            act_buf: dict[str, torch.Tensor] = {}
            grad_buf: dict[str, torch.Tensor] = {}
            handles: list[Any] = []

            if do_act:
                for name, module in model.named_modules():
                    if name in self.config.modules:

                        def _act_hook(
                            _mod: nn.Module,
                            _inp: Any,
                            out: Any,
                            _n: str = name,
                        ) -> None:
                            t = out if isinstance(out, torch.Tensor) else out[0]
                            act_buf[_n] = t.detach()

                        handles.append(module.register_forward_hook(_act_hook))

            if do_grad:
                for name, module in model.named_modules():
                    if name in self.config.modules:

                        def _grad_hook(
                            _mod: nn.Module,
                            _gin: Any,
                            gout: Any,
                            _n: str = name,
                        ) -> None:
                            if gout[0] is not None:
                                grad_buf[_n] = gout[0].detach()

                        handles.append(
                            module.register_full_backward_hook(_grad_hook)
                        )

            try:
                if do_grad:
                    logits = model(inputs)
                    logits.sum().backward()
                    model.zero_grad()
                else:
                    with torch.no_grad():
                        logits = model(inputs)
            finally:
                for h in handles:
                    h.remove()

            if do_act:
                for m in self.config.modules:
                    if m in act_buf:
                        all_activations[m].append(act_buf[m])

            if do_grad:
                for m in self.config.modules:
                    if m in grad_buf:
                        all_gradients[m].append(grad_buf[m])

            if do_logits:
                all_logits.append(logits.detach())

        final_activations: dict[str, torch.Tensor] = {
            m: self._reduce_activation(torch.cat(ts, dim=0))
            for m, ts in all_activations.items()
            if ts
        }
        final_gradients: dict[str, torch.Tensor] = {
            m: self._reduce_gradient(torch.cat(ts, dim=0))
            for m, ts in all_gradients.items()
            if ts
        }
        final_logits: torch.Tensor | None = (
            torch.cat(all_logits, dim=0) if all_logits else None
        )
        return final_activations, final_gradients, final_logits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> ReplayResult:
        """Execute the replay and return captured artifacts.

        Returns a :class:`ReplayResult` whose ``activations`` and
        ``gradients`` fields are ``dict[str, torch.Tensor]`` keyed by
        module name, and ``logits`` is a plain ``torch.Tensor | None``.
        """
        model = self._load_model()
        self._validate_modules(model)

        raw_batches = self._collect_batches()
        selected_batches, n_samples = _apply_slice(
            raw_batches, self.config.dataset_slice
        )

        activations, gradients, logits = self._run_capture(model, selected_batches)

        artifact_sizes: dict[str, int] = {
            f"activations/{name}": t.numel() * t.element_size()
            for name, t in activations.items()
        }
        artifact_sizes.update(
            {
                f"gradients/{name}": t.numel() * t.element_size()
                for name, t in gradients.items()
            }
        )
        if logits is not None:
            artifact_sizes["logits"] = logits.numel() * logits.element_size()

        metadata = ReplayMetadata(
            run=str(self._run_dir),
            checkpoint_epoch=self.config.checkpoint.epoch,
            modules=list(self.config.modules),
            capture=list(self.config.capture),
            activation_reduction=self.config.activation_reduction,
            gradient_mode=self.config.gradient_mode,
            dataset_slice=(
                self.config.dataset_slice.model_dump()
                if self.config.dataset_slice is not None
                else None
            ),
            n_samples=n_samples,
            artifact_sizes=artifact_sizes,
        )

        return ReplayResult(
            activations=activations,
            gradients=gradients,
            logits=logits,
            metadata=metadata,
        )
