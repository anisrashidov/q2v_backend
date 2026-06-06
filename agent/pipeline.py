"""Pipeline orchestrator — agentic version.

All reasoning, retrieval, aggregation, and visualization decisions are
delegated to run_agent(), which drives the process via OpenAI tool-calling.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Tuple

import openai

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.loop import run_agent
from agent.planner import plan_query
from agent.types import PipelineResult, QueryPlan

logger = logging.getLogger(__name__)

_DEFAULT_CLARIFICATION = (
    "Your query does not contain enough information to search clinical trials. "
    "Please include at least one filter such as a condition, drug name, sponsor, "
    "phase, status, or location."
)


async def run_pipeline(
    question: str,
    openai_client: openai.AsyncOpenAI,
    model: str,
    ct_api: ClinicalTrialsAPI,
    max_pages: int = 5,
    time_period: Optional[Tuple[date, date]] = None,
) -> PipelineResult:
    logger.info("Pipeline start | question=%s", question)

    if time_period is None:
        today = date.today()
        time_period = (date(today.year - 3, today.month, today.day), today)

    # Plan first. The plan is authoritative for the clarification gate and seeds
    # the execution loop. If planning itself fails, degrade gracefully: run the
    # loop without a plan rather than failing the whole request.
    plan: Optional[QueryPlan] = None
    try:
        plan = await plan_query(
            question=question,
            openai_client=openai_client,
            model=model,
            time_period=time_period,
        )
        logger.info("Plan | %s", plan.model_dump())
    except Exception:  # noqa: BLE001 — planner is best-effort; never fatal
        logger.exception("Planner failed; continuing without a plan")

    # Gate: reject non-clinical or under-specified queries before any CT API call.
    # ValueError surfaces as an HTTP 200 / code 400 envelope via the router.
    if plan is not None and (plan.needs_clarification or not plan.is_clinical_trial_query):
        message = plan.clarification_message or _DEFAULT_CLARIFICATION
        logger.info("Pipeline rejected at planning gate | %s", message)
        raise ValueError(message)

    visualizations = await run_agent(
        question=question,
        ct_api=ct_api,
        openai_client=openai_client,
        model=model,
        max_pages=max_pages,
        time_period=time_period,
        plan=plan,
    )

    logger.info(
        "Pipeline complete | charts=%d types=%s",
        len(visualizations),
        [v.chart_type for v in visualizations],
    )
    return PipelineResult(visualizations=visualizations)
