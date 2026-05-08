"""Replay-based activation and gradient capture (NI-BETA-001 through NI-BETA-004)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from neuroinquisitor.backends.base import Backend
from neuroinquisitor.formats.base import Format
from neuroinquisitor.loader import load as _load_collection

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
    non-serialisable callables (model_factory, dataloader, dataset_slice).
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointSelector
    modules: list[str] = Field(min_length=1)
    capture: list[Literal["activations", "gradients", "logits"]] = Field(
        min_length=1
    )
    activation_reduction: Literal["raw", "mean", "pool"] = "raw"
    gradient_mode: Literal["per_example", "aggregated"] = "aggregated"


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


SliceFn = Callable[[list[tuple[torch.Tensor, ...]]], list[tuple[torch.Tensor, ...]]]


def _apply_slice(
    batches: list[tuple[torch.Tensor, ...]],
    slice_fn: SliceFn | None,
) -> tuple[list[tuple[torch.Tensor, ...]], int]:
    if not batches:
        return batches, 0
    if slice_fn is None:
        return batches, sum(b[0].shape[0] for b in batches)

    n_tensors = len(batches[0])
    flat: list[tuple[torch.Tensor, ...]] = []
    for batch in batches:
        batch_size = batch[0].shape[0]
        for i in range(batch_size):
            flat.append(tuple(b[i] for b in batch[:n_tensors]))

    selected = slice_fn(flat)
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
        Optional callable ``(flat_samples) -> selected_samples``.  Use the
        factory helpers :func:`first_n`, :func:`random_n`,
        :func:`balanced_n`, or :func:`explicit_indices`.  ``None`` uses all
        batches.
    slice_metadata:
        Optional dict stored verbatim in :attr:`ReplayMetadata.dataset_slice`
        for provenance.  Callers are responsible for populating this when
        they pass a ``dataset_slice``.
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
        dataset_slice: SliceFn | None = None,
        slice_metadata: dict[str, Any] | None = None,
        backend: str | Backend = "local",
        format: str | Format = "hdf5",
    ) -> None:
        self._run_dir = Path(run)
        self._model_factory = model_factory
        self._dataloader = dataloader
        self._slice_fn = dataset_slice
        self._slice_metadata = slice_metadata
        self._backend_spec = backend
        self._format_spec = format

        epoch_int = checkpoint if isinstance(checkpoint, int) else checkpoint.epoch
        self.config = ReplayConfig(
            checkpoint=CheckpointSelector(epoch=epoch_int),
            modules=list(modules),
            capture=list(capture),
            activation_reduction=activation_reduction,
            gradient_mode=gradient_mode,
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
        """Execute the replay and return captured artifacts."""
        model = self._load_model()
        self._validate_modules(model)

        raw_batches = self._collect_batches()
        selected_batches, n_samples = _apply_slice(raw_batches, self._slice_fn)

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
            dataset_slice=self._slice_metadata,
            n_samples=n_samples,
            artifact_sizes=artifact_sizes,
        )

        return ReplayResult(
            activations=activations,
            gradients=gradients,
            logits=logits,
            metadata=metadata,
        )
