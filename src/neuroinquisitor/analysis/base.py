"""Base pydantic v2 request and result models for NI analyzers.

All built-in and third-party analyzers must subclass these types so that the
registry, CLI, and API can inspect and route them without knowing the concrete
analyzer implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AnalyzerRequest(BaseModel):
    """Base request model for all NI analyzers.

    Concrete analyzers add their own fields via subclassing.  ``extra="allow"``
    ensures forward-compatibility as the protocol evolves.
    """

    model_config = ConfigDict(extra="allow")

    run: str
    layers: list[str] = []
    epochs: list[int] = []


class AnalyzerResult(BaseModel):
    """Base result model for all NI analyzers.

    Every analyzer result must carry ``analyzer_name`` and
    ``analyzer_version`` so results are self-describing.
    """

    model_config = ConfigDict(extra="allow")

    analyzer_name: str
    analyzer_version: str
    run: str
    artifact_paths: dict[str, str] = {}
