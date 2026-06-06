"""Tests for Stages 3b + 4 — chart-type decision and VisualizationSpec assembly.

Tested independently:
1. build_spec()          — Stage 4, pure sync function (agent/spec.py)
2. decide_visualization() — Stage 3b, async LLM call (agent/visualize.py)
3. Combined path         — decide → build (mirrors what pipeline.py does)
"""
from __future__ import annotations

import pytest

from agent.spec import build_spec
from agent.types import AggregatedData, ChartType, VisualizationSpec
from agent.visualize import decide_visualization


# ── Test doubles ───────────────────────────────────────────────────────────────


class FakeLLM:
    """Returns *response* verbatim for every LLM call."""

    def __init__(self, response: dict) -> None:
        self._response = response

    async def complete(self, system: str, user: str, schema: dict) -> dict:
        return self._response


# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_aggregated(**overrides) -> AggregatedData:
    defaults = dict(
        by_phase={"PHASE1": 5, "PHASE2": 10, "PHASE3": 3},
        by_status={"RECRUITING": 20, "COMPLETED": 50, "TERMINATED": 5},
        by_year={"2019": 5, "2020": 8, "2021": 10, "2022": 12},
        by_study_type={"INTERVENTIONAL": 60, "OBSERVATIONAL": 20},
        by_sponsor_class={"INDUSTRY": 40, "NIH": 20, "OTHER": 15},
        enrollment_buckets={"1-100": 5, "101-500": 15, "501-1000": 8, "1001+": 2},
        top_conditions={"Diabetes": 10, "Cancer": 8, "Asthma": 3},
        total_studies=80,
    )
    defaults.update(overrides)
    return AggregatedData(**defaults)


def _decision(
    chart_type: str = "bar",
    agg_key: str = "by_phase",
    title: str = "Chart",
    x_label: str = "X",
    y_label: str = "Count",
    description: str | None = None,
) -> dict:
    d: dict = {
        "chart_type": chart_type,
        "aggregation_key": agg_key,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }
    if description is not None:
        d["description"] = description
    return d


# ── build_spec (pure, synchronous) ────────────────────────────────────────────


def test_build_spec_bar_series_values():
    agg = make_aggregated()
    spec = build_spec(_decision("bar", "by_phase"), agg)

    assert spec.chart_type == ChartType.bar
    labels = {dp.label for dp in spec.series}
    assert labels == {"PHASE1", "PHASE2", "PHASE3"}

    by_label = {dp.label: dp.value for dp in spec.series}
    assert by_label["PHASE2"] == 10.0
    assert by_label["PHASE1"] == 5.0


def test_build_spec_total_records():
    agg = make_aggregated(total_studies=80)
    spec = build_spec(_decision(), agg)
    assert spec.total_records == 80


def test_build_spec_axis_labels():
    spec = build_spec(_decision(x_label="Phase", y_label="Trials"), make_aggregated())
    assert spec.axis_labels["x"] == "Phase"
    assert spec.axis_labels["y"] == "Trials"


def test_build_spec_xy_fields():
    spec = build_spec(_decision(), make_aggregated())
    assert spec.x_field == "label"
    assert spec.y_field == "value"


def test_build_spec_none_chart():
    """chart_type=none should parse without error."""
    spec = build_spec(_decision("none", "by_status"), make_aggregated())
    assert spec.chart_type == ChartType.none


def test_build_spec_description_passed_through():
    spec = build_spec(
        _decision(description="Shows breakdown by phase"),
        make_aggregated(),
    )
    assert spec.description == "Shows breakdown by phase"


def test_build_spec_missing_description_is_none():
    spec = build_spec(_decision(), make_aggregated())
    assert spec.description is None


def test_build_spec_line_year_data():
    agg = make_aggregated()
    spec = build_spec(_decision("line", "by_year", x_label="Year"), agg)

    assert spec.chart_type == ChartType.line
    years = [dp.label for dp in spec.series]
    assert years == sorted(years)  # by_year is stored sorted ascending


def test_build_spec_pie_status():
    agg = make_aggregated()
    spec = build_spec(_decision("pie", "by_status"), agg)
    assert spec.chart_type == ChartType.pie
    assert len(spec.series) == 3


def test_build_spec_enrollment_buckets():
    agg = make_aggregated()
    spec = build_spec(_decision("bar", "enrollment_buckets"), agg)
    labels = {dp.label for dp in spec.series}
    assert labels == {"1-100", "101-500", "501-1000", "1001+"}


def test_build_spec_returns_visualization_spec_instance():
    spec = build_spec(_decision(), make_aggregated())
    assert isinstance(spec, VisualizationSpec)


# ── decide_visualization (async, uses LLM) ────────────────────────────────────


@pytest.mark.asyncio
async def test_decide_returns_dict():
    """decide_visualization returns a raw dict (not a VisualizationSpec)."""
    llm = FakeLLM(_decision("bar", "by_phase", "Phases", "Phase", "Count"))
    agg = make_aggregated()
    result = await decide_visualization("how many trials per phase?", agg, llm)
    assert isinstance(result, dict)
    assert result["chart_type"] == "bar"
    assert result["aggregation_key"] == "by_phase"


@pytest.mark.asyncio
async def test_decide_passes_question_to_llm():
    """The question is embedded in the user message sent to the LLM."""
    received_user: list[str] = []

    class CaptureLLM:
        async def complete(self, system: str, user: str, schema: dict) -> dict:
            received_user.append(user)
            return _decision()

    agg = make_aggregated()
    await decide_visualization("cancer trials by sponsor", agg, CaptureLLM())
    assert "cancer trials by sponsor" in received_user[0]


# ── Combined path (decide → build) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_combined_bar_chart():
    llm = FakeLLM(_decision("bar", "by_phase", "Trials by Phase", "Phase", "Count"))
    agg = make_aggregated()
    decision = await decide_visualization("how many per phase?", agg, llm)
    spec = build_spec(decision, agg)

    assert spec.chart_type == ChartType.bar
    assert len(spec.series) == 3


@pytest.mark.asyncio
async def test_combined_line_chart():
    llm = FakeLLM(_decision("line", "by_year", "Trials Over Time", "Year", "Count"))
    agg = make_aggregated()
    spec = build_spec(await decide_visualization("trend over years", agg, llm), agg)

    assert spec.chart_type == ChartType.line
    assert spec.axis_labels["x"] == "Year"


@pytest.mark.asyncio
async def test_combined_pie_chart():
    llm = FakeLLM(
        _decision("pie", "by_status", "Status Share", "Status", "Share")
    )
    agg = make_aggregated()
    spec = build_spec(await decide_visualization("what % are recruiting?", agg, llm), agg)
    assert spec.chart_type == ChartType.pie


@pytest.mark.asyncio
async def test_combined_none_chart():
    llm = FakeLLM(_decision("none", "by_status", "N/A", "", ""))
    agg = make_aggregated()
    spec = build_spec(await decide_visualization("define phase 3", agg, llm), agg)
    assert spec.chart_type == ChartType.none


@pytest.mark.asyncio
async def test_combined_top_conditions():
    llm = FakeLLM(
        _decision("bar", "top_conditions", "Top Conditions", "Condition", "Count")
    )
    agg = make_aggregated()
    spec = build_spec(
        await decide_visualization("most common conditions", agg, llm), agg
    )

    assert spec.chart_type == ChartType.bar
    labels = {dp.label for dp in spec.series}
    assert "Diabetes" in labels
    assert "Cancer" in labels


@pytest.mark.asyncio
async def test_combined_total_records():
    llm = FakeLLM(_decision())
    agg = make_aggregated(total_studies=123)
    spec = build_spec(await decide_visualization("any question", agg, llm), agg)
    assert spec.total_records == 123
