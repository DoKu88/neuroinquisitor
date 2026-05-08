"""Flat, explicit analyzer registry for NI built-ins and user-registered analyzers.

Registration is explicit — no metaclasses, no import side-effects.  Adding an
analyzer requires two steps: implement the interface, then call ``register()``.

The registry is a plain dict so ``ni plugins list`` (Sprint 9 CLI) can iterate
over it without any framework overhead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from neuroinquisitor.analysis.base import AnalyzerRequest, AnalyzerResult

# Public type alias so callers can annotate analyzer callables without
# importing the concrete subclasses they haven't created yet.
AnalyzerFn = Callable[[AnalyzerRequest], AnalyzerResult]


@dataclass
class AnalyzerSpec:
    """Metadata entry for one registered analyzer.

    Fields
    ------
    name:
        Unique analyzer identifier, e.g. ``"trajectory_stats"``.
    version:
        Semantic version string, e.g. ``"0.1.0"``.
    required_inputs:
        Artifact kinds the analyzer needs, e.g. ``["weights"]`` or
        ``["activations", "labels"]``.
    output_format:
        Whether the analyzer emits tensors, a table, or both.
    description:
        One-line human-readable description shown by ``ni plugins list``.
    fn:
        The analyzer callable.  Takes an :class:`AnalyzerRequest` subclass,
        returns an :class:`AnalyzerResult` subclass.
    """

    name: str
    version: str
    required_inputs: list[str]
    output_format: Literal["tensor", "table", "both"]
    description: str
    fn: Any  # Callable[[AnalyzerRequest], AnalyzerResult]


# ---------------------------------------------------------------------------
# Module-level registry dict — intentionally a plain dict for transparency
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, AnalyzerSpec] = {}


def register(spec: AnalyzerSpec) -> None:
    """Add *spec* to the registry.

    Raises :exc:`ValueError` if a spec with the same name is already present.
    """
    if spec.name in _REGISTRY:
        raise ValueError(
            f"Analyzer {spec.name!r} is already registered. "
            "Use a different name or remove the existing entry first."
        )
    _REGISTRY[spec.name] = spec


def get_analyzer(name: str) -> AnalyzerSpec:
    """Return the :class:`AnalyzerSpec` for *name*.

    Raises :exc:`KeyError` with a helpful message when the analyzer is absent.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Analyzer {name!r} not found. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_analyzers() -> list[AnalyzerSpec]:
    """Return all registered analyzers in insertion order."""
    return list(_REGISTRY.values())
