"""neuroinquisitor.analysis — analyzer base types and registry."""

from neuroinquisitor.analysis.base import AnalyzerRequest, AnalyzerResult
from neuroinquisitor.analysis.registry import (
    AnalyzerSpec,
    get_analyzer,
    list_analyzers,
    register,
)

__all__ = [
    "AnalyzerRequest",
    "AnalyzerResult",
    "AnalyzerSpec",
    "register",
    "get_analyzer",
    "list_analyzers",
]
