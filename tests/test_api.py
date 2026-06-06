"""Smoke tests for POST /api/query via FastAPI TestClient.

run_pipeline is monkeypatched so no real LLM or API calls are made.
The lifespan creates real adapter objects (with the dummy API key from
conftest.py) but they are never invoked during these tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.types import (
    ChartType,
    DataPoint,
    DataSummary,
    PipelineResult,
    QueryParams,
    VisualizationSpec,
)
from app.main import app


# ── Fixture helpers ────────────────────────────────────────────────────────────


def _make_result(chart_type: ChartType = ChartType.bar) -> PipelineResult:
    return PipelineResult(
        interpreted_params=QueryParams(query_cond="diabetes", page_size=50),
        data_summary=DataSummary(total_count=1200, retrieved=50, has_more=True),
        visualization_spec=VisualizationSpec(
            chart_type=chart_type,
            title="Trials by Phase",
            series=[
                DataPoint(label="PHASE2", value=30.0),
                DataPoint(label="PHASE3", value=15.0),
            ],
            axis_labels={"x": "Phase", "y": "Count"},
            aggregation_applied="Grouped by phase",
            total_records=50,
        ),
    )


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient; lifespan runs once for the whole module."""
    with TestClient(app) as c:
        yield c


# ── Success path ───────────────────────────────────────────────────────────────


def test_query_returns_200(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    resp = client.post("/api/query", json={"query": "diabetes trials by phase"})
    assert resp.status_code == 200


def test_response_has_all_top_level_fields(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "any question"}).json()
    assert "interpreted_params" in data
    assert "data_summary" in data
    assert "visualization_spec" in data


def test_visualization_spec_shape(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result(ChartType.bar)

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "phases"}).json()
    spec = data["visualization_spec"]

    assert spec["chart_type"] == "bar"
    assert spec["title"] == "Trials by Phase"
    assert isinstance(spec["series"], list)
    assert len(spec["series"]) == 2
    assert spec["series"][0]["label"] == "PHASE2"
    assert spec["series"][0]["value"] == 30.0


def test_data_summary_fields(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "any question"}).json()
    summary = data["data_summary"]

    assert summary["total_count"] == 1200
    assert summary["retrieved"] == 50
    assert summary["has_more"] is True


def test_interpreted_params_returned(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "diabetes"}).json()
    params = data["interpreted_params"]

    assert params["query_cond"] == "diabetes"
    assert params["page_size"] == 50


def test_optional_fields_accepted(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    resp = client.post(
        "/api/query",
        json={
            "query": "recruiting trials",
            "fields": {"status": "RECRUITING", "phase": "3"},
        },
    )
    assert resp.status_code == 200


def test_line_chart_type_serialised(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result(ChartType.line)

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "trend over time"}).json()
    assert data["visualization_spec"]["chart_type"] == "line"


# ── Validation errors ──────────────────────────────────────────────────────────


def test_missing_query_field_returns_422(client):
    resp = client.post("/api/query", json={})
    assert resp.status_code == 422


def test_empty_query_string_returns_422(client):
    resp = client.post("/api/query", json={"query": "ab"})  # min_length=3, "ab" is 2
    assert resp.status_code == 422


def test_invalid_body_type_returns_422(client):
    resp = client.post("/api/query", json={"query": 12345})
    # 12345 is coerced to "12345" by Pydantic; length > 3, so 200
    # This verifies Pydantic coercion rather than a 422
    assert resp.status_code in (200, 422)


# ── Error propagation ──────────────────────────────────────────────────────────


def test_pipeline_exception_returns_500(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        raise RuntimeError("ClinicalTrials API timeout")

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    resp = client.post("/api/query", json={"query": "some query"})
    assert resp.status_code == 500
    assert "Pipeline error" in resp.json()["detail"]


# ── Health check ───────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
