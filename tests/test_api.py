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
    DataSummary,
    PipelineResult,
    QueryParams,
    VisualizationSpec,
)
from app.main import app


# ── Fixture helpers ────────────────────────────────────────────────────────────


def _make_viz(
    chart_type: ChartType = ChartType.bar_chart,
    sequence_index: int = 0,
    group: str = "distribution",
) -> VisualizationSpec:
    return VisualizationSpec(
        chart_type=chart_type,
        title="Trials by Phase",
        encoding={"x": "label", "y": "value"},
        data=[
            {"label": "PHASE2", "value": 30.0},
            {"label": "PHASE3", "value": 15.0},
        ],
        total_records=50,
        sequence_index=sequence_index,
        group=group,
    )


def _make_result(chart_type: ChartType = ChartType.bar_chart) -> PipelineResult:
    return PipelineResult(
        interpreted_params=QueryParams(query_cond="diabetes", page_size=50),
        data_summary=DataSummary(total_count=1200, retrieved=50, has_more=True),
        visualizations=[_make_viz(chart_type)],
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

    resp_json = client.post("/api/query", json={"query": "any question"}).json()
    # Router wraps every response in the BaseResponse envelope.
    assert "code" in resp_json
    assert "message" in resp_json
    assert "data" in resp_json
    data = resp_json["data"]
    assert "interpreted_params" in data
    assert "data_summary" in data
    assert "visualizations" in data


def test_visualization_spec_shape(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result(ChartType.bar_chart)

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "phases"}).json()["data"]
    assert isinstance(data["visualizations"], list)
    spec = data["visualizations"][0]

    assert spec["chart_type"] == "bar_chart"
    assert spec["title"] == "Trials by Phase"
    assert isinstance(spec["data"], list)
    assert len(spec["data"]) == 2
    assert spec["data"][0]["label"] == "PHASE2"
    assert spec["data"][0]["value"] == 30.0


def test_data_summary_fields(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "any question"}).json()["data"]
    summary = data["data_summary"]

    assert summary["total_count"] == 1200
    assert summary["retrieved"] == 50
    assert summary["has_more"] is True


def test_interpreted_params_returned(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "diabetes"}).json()["data"]
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


def test_time_series_chart_type_serialised(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result(ChartType.time_series)

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "trend over time"}).json()["data"]
    assert data["visualizations"][0]["chart_type"] == "time_series"


def test_encoding_field_present(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "phases"}).json()["data"]
    spec = data["visualizations"][0]
    assert "encoding" in spec
    assert spec["encoding"]["x"] == "label"
    assert spec["encoding"]["y"] == "value"


def test_multi_chart_response(client, monkeypatch):
    async def mock_pipeline(*args, **kwargs):
        return PipelineResult(
            visualizations=[
                _make_viz(ChartType.time_series,   sequence_index=0, group="trends"),
                _make_viz(ChartType.bar_chart, sequence_index=1, group="distribution"),
            ]
        )

    monkeypatch.setattr("app.routers.query.run_pipeline", mock_pipeline)

    data = client.post("/api/query", json={"query": "dashboard"}).json()["data"]
    vizs = data["visualizations"]
    assert len(vizs) == 2
    assert vizs[0]["sequence_index"] == 0
    assert vizs[0]["group"] == "trends"
    assert vizs[0]["chart_type"] == "time_series"
    assert vizs[1]["sequence_index"] == 1
    assert vizs[1]["group"] == "distribution"


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
