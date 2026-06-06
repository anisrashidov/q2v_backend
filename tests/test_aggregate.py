"""Tests for Stage 3a — deterministic aggregation.

No LLM or network calls — pure unit tests on aggregate_studies().
All assertions check exact counts so regressions are immediately obvious.
"""
from __future__ import annotations

from agent.aggregate import aggregate_studies
from agent.types import Study


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_study(**kwargs) -> Study:
    """Build a Study with sensible defaults; override with kwargs."""
    defaults: dict = dict(
        nct_id="NCT000",
        brief_title="Test Study",
        overall_status="RECRUITING",
        phases=["PHASE2"],
        start_date="2020-01",
        completion_date="2022-12",
        conditions=["Diabetes"],
        interventions=["Metformin"],
        enrollment=100,
        sponsor="Test Sponsor",
        sponsor_class="INDUSTRY",
        study_type="INTERVENTIONAL",
    )
    defaults.update(kwargs)
    return Study(**defaults)


# ── Phase aggregation ──────────────────────────────────────────────────────────


def test_count_by_phase_single():
    studies = [
        make_study(nct_id="1", phases=["PHASE1"]),
        make_study(nct_id="2", phases=["PHASE2"]),
        make_study(nct_id="3", phases=["PHASE2"]),
    ]
    result = aggregate_studies(studies)
    assert result.by_phase["PHASE1"] == 1
    assert result.by_phase["PHASE2"] == 2


def test_count_by_phase_multi_phase():
    """A study listing PHASE1/PHASE2 increments both counters."""
    studies = [make_study(nct_id="1", phases=["PHASE1", "PHASE2"])]
    result = aggregate_studies(studies)
    assert result.by_phase["PHASE1"] == 1
    assert result.by_phase["PHASE2"] == 1


def test_count_by_phase_missing_phase_is_na():
    studies = [make_study(nct_id="1", phases=[])]
    result = aggregate_studies(studies)
    assert result.by_phase.get("N/A", 0) == 1


# ── Status aggregation ─────────────────────────────────────────────────────────


def test_count_by_status():
    studies = [
        make_study(nct_id="1", overall_status="RECRUITING"),
        make_study(nct_id="2", overall_status="RECRUITING"),
        make_study(nct_id="3", overall_status="COMPLETED"),
        make_study(nct_id="4", overall_status=None),
    ]
    result = aggregate_studies(studies)
    assert result.by_status["RECRUITING"] == 2
    assert result.by_status["COMPLETED"] == 1
    assert result.by_status["N/A"] == 1


# ── Year aggregation ───────────────────────────────────────────────────────────


def test_count_by_year():
    studies = [
        make_study(nct_id="1", start_date="2020-01"),
        make_study(nct_id="2", start_date="2020-06"),
        make_study(nct_id="3", start_date="2021-03"),
        make_study(nct_id="4", start_date=None),
    ]
    result = aggregate_studies(studies)
    assert result.by_year["2020"] == 2
    assert result.by_year["2021"] == 1
    assert "None" not in result.by_year
    assert len([k for k in result.by_year if not k.isdigit()]) == 0


def test_by_year_sorted_ascending():
    studies = [
        make_study(nct_id="3", start_date="2022-01"),
        make_study(nct_id="1", start_date="2020-01"),
        make_study(nct_id="2", start_date="2021-01"),
    ]
    result = aggregate_studies(studies)
    assert list(result.by_year.keys()) == ["2020", "2021", "2022"]


def test_invalid_year_prefix_ignored():
    studies = [make_study(nct_id="1", start_date="???")]
    result = aggregate_studies(studies)
    assert result.by_year == {}


# ── Enrollment buckets ─────────────────────────────────────────────────────────


def test_enrollment_buckets_all_bands():
    studies = [
        make_study(nct_id="a", enrollment=50),
        make_study(nct_id="b", enrollment=200),
        make_study(nct_id="c", enrollment=750),
        make_study(nct_id="d", enrollment=2000),
    ]
    result = aggregate_studies(studies)
    assert result.enrollment_buckets["1-100"] == 1
    assert result.enrollment_buckets["101-500"] == 1
    assert result.enrollment_buckets["501-1000"] == 1
    assert result.enrollment_buckets["1001+"] == 1


def test_enrollment_none_not_bucketed():
    studies = [
        make_study(nct_id="a", enrollment=None),
        make_study(nct_id="b", enrollment=100),
    ]
    result = aggregate_studies(studies)
    assert result.enrollment_buckets["1-100"] == 1
    total = sum(result.enrollment_buckets.values())
    assert total == 1


def test_enrollment_boundary_100():
    """Exactly 100 goes into the '1-100' bucket."""
    studies = [make_study(nct_id="1", enrollment=100)]
    result = aggregate_studies(studies)
    assert result.enrollment_buckets["1-100"] == 1
    assert result.enrollment_buckets["101-500"] == 0


# ── Top conditions ─────────────────────────────────────────────────────────────


def test_top_conditions_capped_at_10():
    studies = [
        make_study(nct_id=str(i), conditions=[f"Condition{i}"])
        for i in range(15)
    ]
    result = aggregate_studies(studies)
    assert len(result.top_conditions) == 10


def test_top_conditions_counts_correctly():
    studies = [
        make_study(nct_id="1", conditions=["Cancer", "Diabetes"]),
        make_study(nct_id="2", conditions=["Cancer"]),
        make_study(nct_id="3", conditions=["Cancer"]),
    ]
    result = aggregate_studies(studies)
    assert result.top_conditions["Cancer"] == 3
    assert result.top_conditions["Diabetes"] == 1


def test_condition_cap_per_study():
    """Only the first 5 conditions per study are counted."""
    studies = [
        make_study(nct_id="1", conditions=[f"Cond{i}" for i in range(10)])
    ]
    result = aggregate_studies(studies)
    assert sum(result.top_conditions.values()) == 5


# ── Total studies ──────────────────────────────────────────────────────────────


def test_total_studies():
    studies = [make_study(nct_id=str(i)) for i in range(7)]
    result = aggregate_studies(studies)
    assert result.total_studies == 7


def test_empty_input():
    result = aggregate_studies([])
    assert result.total_studies == 0
    assert result.by_phase == {}
    assert result.by_status == {}
    assert result.by_year == {}


# ── Sponsor class ──────────────────────────────────────────────────────────────


def test_sponsor_class_counted():
    studies = [
        make_study(nct_id="1", sponsor_class="INDUSTRY"),
        make_study(nct_id="2", sponsor_class="INDUSTRY"),
        make_study(nct_id="3", sponsor_class="NIH"),
        make_study(nct_id="4", sponsor_class=None),
    ]
    result = aggregate_studies(studies)
    assert result.by_sponsor_class["INDUSTRY"] == 2
    assert result.by_sponsor_class["NIH"] == 1
    assert "None" not in result.by_sponsor_class
