"""Top-level pipeline orchestrator.

Runs the four stages in order and returns a PipelineResult.

Stage 1  INTERPRET  agent/interpret.py   — NL question → QueryParams      (LLM)
Stage 2  RETRIEVE   agent/retrieve.py    — QueryParams → StudyData         (API)
Stage 3a ANALYZE    agent/aggregate.py   — StudyData → AggregatedData      (pure)
Stage 3b DECIDE     agent/visualize.py   — question + agg → chart decision (LLM)
Stage 4  SPEC       agent/spec.py        — decision + agg → VisualizationSpec (pure)
"""
from __future__ import annotations

import logging
from typing import Optional

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.aggregate import aggregate_studies
from agent.interpret import interpret_query
from agent.ports import LLMPort
from agent.retrieve import retrieve_studies
from agent.spec import build_spec
from agent.types import DataSummary, PipelineResult, QueryParams
from agent.visualize import decide_visualization

logger = logging.getLogger(__name__)


async def run_pipeline(
    question: str,
    llm: LLMPort,
    ct_api: ClinicalTrialsAPI,
    max_pages: int = 5,
    param_overrides: Optional[dict] = None,
) -> PipelineResult:
    """Execute the full 4-stage pipeline end-to-end.

    Parameters
    ----------
    question:
        Raw natural-language question from the user.
    llm:
        LLM adapter (used in stages 1 and 3b).
    ct_api:
        ClinicalTrialsAPI adapter (used in stage 2).
    max_pages:
        Maximum pages of results to fetch from the API.
    param_overrides:
        Optional dict of QueryParams field overrides applied after NL
        interpretation.  Explicit request-body fields take precedence over
        the LLM's inferred values.
    """
    logger.info(f"Pipeline start | question={question}")

    # Stage 1 — Interpret
    logger.debug("Stage 1 INTERPRET | start")
    params: QueryParams = await interpret_query(
        question=question, llm=llm
    )
    if param_overrides:
        params = params.model_copy(update=param_overrides)
        logger.debug("Stage 1 INTERPRET | overrides applied: %s", list(param_overrides))
    logger.info(
        f"Stage 1 INTERPRET | done | params: \n {params.model_dump_json(indent=2)} "
    )

    # Stage 2 — Retrieve
    logger.debug("Stage 2 RETRIEVE | start | max_pages=%d", max_pages)
    study_data = await retrieve_studies(params, ct_api, max_pages=max_pages)
    logger.info(
        "Stage 2 RETRIEVE | done | retrieved=%d total_count=%d has_more=%s",
        len(study_data.studies),
        study_data.total_count,
        study_data.next_page_token is not None,
    )

    # Stage 3a — Analyze (deterministic)
    logger.debug("Stage 3a AGGREGATE | start")
    aggregated = aggregate_studies(study_data.studies)
    logger.info(
        "Stage 3a AGGREGATE | done | phases=%d statuses=%d top_conditions=%d",
        len(aggregated.by_phase),
        len(aggregated.by_status),
        len(aggregated.top_conditions),
    )

    # Stage 3b — Decide (LLM)
    logger.debug("Stage 3b DECIDE | start")
    decision = await decide_visualization(
        question=question, aggregated=aggregated, llm=llm
    )
    logger.info(
        "Stage 3b DECIDE | done | chart_type=%r agg_key=%r",
        decision.get("chart_type"),
        decision.get("aggregation_key"),
    )

    # Stage 4 — Spec (pure)
    logger.debug("Stage 4 SPEC | start")
    viz_spec = build_spec(decision, aggregated)
    logger.info(
        "Stage 4 SPEC | done | chart_type=%s series_len=%d",
        viz_spec.chart_type,
        len(viz_spec.series),
    )

    logger.info("Pipeline complete | chart_type=%s", viz_spec.chart_type)

    return PipelineResult(
        interpreted_params=params,
        data_summary=DataSummary(
            total_count=study_data.total_count,
            retrieved=len(study_data.studies),
            has_more=study_data.next_page_token is not None,
        ),
        visualization_spec=viz_spec,
    )
