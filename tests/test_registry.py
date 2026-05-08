"""Tests for the analyzer registry (NI-DELTA-002).

Verifies:
- Base models are importable from stable public paths
- Registry is queryable and returns typed metadata
- All required fields are present on AnalyzerSpec
- Duplicate registration raises an error
- A minimal stub analyzer passes registration
- list_analyzers returns all registered entries
"""

from __future__ import annotations

import pytest

from neuroinquisitor.analysis import (
    AnalyzerRequest,
    AnalyzerResult,
    AnalyzerSpec,
    get_analyzer,
    list_analyzers,
    register,
)
from neuroinquisitor.analysis.registry import _REGISTRY


# ---------------------------------------------------------------------------
# Helpers — isolated registry state
# ---------------------------------------------------------------------------


def _make_spec(name: str = "test_analyzer") -> AnalyzerSpec:
    def _fn(req: AnalyzerRequest) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=req.run,
            analyzer_version="0.1.0",
            run=req.run,
        )

    return AnalyzerSpec(
        name=name,
        version="0.1.0",
        required_inputs=["weights"],
        output_format="table",
        description="A stub analyzer for testing.",
        fn=_fn,
    )


@pytest.fixture(autouse=True)
def _clean_registry():  # type: ignore[return]
    """Restore the registry to its original state after each test."""
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


# ---------------------------------------------------------------------------
# Public import paths (acceptance: importable from stable public path)
# ---------------------------------------------------------------------------


def test_analyzer_request_importable_from_top_level() -> None:
    from neuroinquisitor import AnalyzerRequest as AR  # noqa: F401


def test_analyzer_result_importable_from_top_level() -> None:
    from neuroinquisitor import AnalyzerResult as AR  # noqa: F401


def test_analyzer_spec_importable_from_top_level() -> None:
    from neuroinquisitor import AnalyzerSpec as AS  # noqa: F401


def test_registry_functions_importable_from_top_level() -> None:
    from neuroinquisitor import get_analyzer, list_analyzers, register  # noqa: F401


def test_base_models_importable_from_analysis_subpackage() -> None:
    from neuroinquisitor.analysis.base import AnalyzerRequest, AnalyzerResult  # noqa: F401


# ---------------------------------------------------------------------------
# AnalyzerRequest / AnalyzerResult — pydantic v2 base model behaviour
# ---------------------------------------------------------------------------


def test_analyzer_request_is_pydantic_model() -> None:
    from pydantic import BaseModel

    assert issubclass(AnalyzerRequest, BaseModel)


def test_analyzer_result_is_pydantic_model() -> None:
    from pydantic import BaseModel

    assert issubclass(AnalyzerResult, BaseModel)


def test_analyzer_request_required_fields() -> None:
    req = AnalyzerRequest(run="/tmp/run")
    assert req.run == "/tmp/run"
    assert req.layers == []
    assert req.epochs == []


def test_analyzer_result_required_fields() -> None:
    res = AnalyzerResult(
        analyzer_name="trajectory_stats",
        analyzer_version="0.1.0",
        run="/tmp/run",
    )
    assert res.analyzer_name == "trajectory_stats"
    assert res.analyzer_version == "0.1.0"
    assert res.artifact_paths == {}


def test_analyzer_request_allows_extra_fields() -> None:
    req = AnalyzerRequest(run="/tmp/run", custom_param=42)
    assert req.model_extra is not None
    assert req.model_extra["custom_param"] == 42


def test_analyzer_result_round_trip() -> None:
    res = AnalyzerResult(
        analyzer_name="spectrum_rank",
        analyzer_version="0.2.0",
        run="/runs/exp_a",
        artifact_paths={"table": "/runs/exp_a/spectrum.parquet"},
    )
    restored = AnalyzerResult.model_validate(res.model_dump())
    assert restored == res


# ---------------------------------------------------------------------------
# AnalyzerSpec — field validation
# ---------------------------------------------------------------------------


def test_analyzer_spec_fields_present() -> None:
    spec = _make_spec("my_analyzer")
    assert spec.name == "my_analyzer"
    assert spec.version == "0.1.0"
    assert isinstance(spec.required_inputs, list)
    assert spec.output_format in {"tensor", "table", "both"}
    assert isinstance(spec.description, str)
    assert callable(spec.fn)


def test_analyzer_spec_callable_fn_executes() -> None:
    spec = _make_spec("callable_test")
    req = AnalyzerRequest(run="/tmp/run")
    result = spec.fn(req)
    assert isinstance(result, AnalyzerResult)


# ---------------------------------------------------------------------------
# register / get_analyzer / list_analyzers
# ---------------------------------------------------------------------------


def test_register_and_retrieve() -> None:
    spec = _make_spec("my_unique_analyzer")
    register(spec)
    retrieved = get_analyzer("my_unique_analyzer")
    assert retrieved.name == "my_unique_analyzer"
    assert retrieved.version == "0.1.0"


def test_list_analyzers_includes_registered() -> None:
    spec = _make_spec("listed_analyzer")
    register(spec)
    names = [s.name for s in list_analyzers()]
    assert "listed_analyzer" in names


def test_list_analyzers_returns_list_of_specs() -> None:
    spec = _make_spec("typed_analyzer")
    register(spec)
    result = list_analyzers()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, AnalyzerSpec)


def test_duplicate_registration_raises() -> None:
    spec = _make_spec("duplicate_name")
    register(spec)
    with pytest.raises(ValueError, match="already registered"):
        register(_make_spec("duplicate_name"))


def test_get_analyzer_missing_raises_key_error() -> None:
    with pytest.raises(KeyError, match="not found"):
        get_analyzer("nonexistent_analyzer_xyz")


def test_get_analyzer_error_lists_available() -> None:
    spec = _make_spec("available_analyzer")
    register(spec)
    with pytest.raises(KeyError, match="available_analyzer"):
        get_analyzer("nonexistent_xyz")


def test_register_multiple_analyzers() -> None:
    for i in range(3):
        register(_make_spec(f"multi_analyzer_{i}"))
    names = [s.name for s in list_analyzers()]
    for i in range(3):
        assert f"multi_analyzer_{i}" in names


# ---------------------------------------------------------------------------
# Stub analyzer satisfies interface (acceptance criterion)
# ---------------------------------------------------------------------------


def test_stub_analyzer_passes_registration() -> None:
    """A minimal stub implementing the interface must register without errors."""

    class MyRequest(AnalyzerRequest):
        threshold: float = 0.5

    class MyResult(AnalyzerResult):
        score: float

    def my_fn(req: MyRequest) -> MyResult:
        return MyResult(
            analyzer_name="stub",
            analyzer_version="0.1.0",
            run=req.run,
            score=req.threshold * 2,
        )

    spec = AnalyzerSpec(
        name="stub_analyzer",
        version="0.1.0",
        required_inputs=["weights"],
        output_format="table",
        description="Stub analyzer for contract testing.",
        fn=my_fn,
    )
    register(spec)

    retrieved = get_analyzer("stub_analyzer")
    req = MyRequest(run="/tmp/run", threshold=0.3)
    result = retrieved.fn(req)
    assert isinstance(result, AnalyzerResult)
    assert result.score == pytest.approx(0.6)
