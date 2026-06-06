"""Stage 3a — ANALYZE.

Deterministic aggregation over a list of studies.
No LLM, no I/O — pure Python, fully testable in isolation.
"""
from __future__ import annotations

from collections import Counter
from typing import List

from agent.types import AggregatedData, Study


def aggregate_studies(studies: List[Study]) -> AggregatedData:
    """Compute all standard aggregations over *studies*.

    Aggregations produced
    ─────────────────────
    by_phase          : count per clinical-phase string
    by_status         : count per overall-status string
    by_year           : count per 4-digit start year (sorted ascending)
    by_study_type     : count per study-type string
    by_sponsor_class  : count per sponsor-class string
    enrollment_buckets: count per enrollment-size band (1-100 / 101-500 / …)
    top_conditions    : top-10 conditions by frequency
    total_studies     : len(studies)
    """
    by_phase:        Counter = Counter()
    by_status:       Counter = Counter()
    by_year:         Counter = Counter()
    by_study_type:   Counter = Counter()
    by_sponsor_class: Counter = Counter()
    enrollment_buckets: Counter = Counter(
        {"1-100": 0, "101-500": 0, "501-1000": 0, "1001+": 0}
    )
    condition_counter: Counter = Counter()

    for study in studies:
        # Phase — a study can list multiple phases; each gets a count
        if study.phases:
            for phase in study.phases:
                by_phase[phase] += 1
        else:
            by_phase["N/A"] += 1

        # Overall status
        by_status[study.overall_status or "N/A"] += 1

        # Start year — require a valid 4-digit prefix
        if study.start_date and len(study.start_date) >= 4:
            year = study.start_date[:4]
            if year.isdigit():
                by_year[year] += 1

        # Study type
        by_study_type[study.study_type or "N/A"] += 1

        # Sponsor class
        if study.sponsor_class:
            by_sponsor_class[study.sponsor_class] += 1

        # Conditions — cap at 5 per study to avoid count inflation
        for cond in study.conditions[:5]:
            if cond:
                condition_counter[cond] += 1

        # Enrollment buckets
        if study.enrollment is not None:
            e = study.enrollment
            if e <= 100:
                enrollment_buckets["1-100"] += 1
            elif e <= 500:
                enrollment_buckets["101-500"] += 1
            elif e <= 1000:
                enrollment_buckets["501-1000"] += 1
            else:
                enrollment_buckets["1001+"] += 1

    return AggregatedData(
        by_phase=dict(by_phase),
        by_status=dict(by_status),
        by_year=dict(sorted(by_year.items())),
        by_study_type=dict(by_study_type),
        by_sponsor_class=dict(by_sponsor_class),
        enrollment_buckets=dict(enrollment_buckets),
        top_conditions=dict(condition_counter.most_common(10)),
        total_studies=len(studies),
    )
