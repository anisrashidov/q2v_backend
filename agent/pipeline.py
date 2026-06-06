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
from agent.types import PipelineResult

logger = logging.getLogger(__name__)


async def run_pipeline(
    question: str,
    openai_client: openai.AsyncOpenAI,
    model: str,
    ct_api: ClinicalTrialsAPI,
    max_pages: int = 5,
    time_period: Optional[Tuple[date, date]] = None,
) -> PipelineResult:
    logger.info("Pipeline start | question=%s", question)

    visualizations = await run_agent(
        question=question,
        ct_api=ct_api,
        openai_client=openai_client,
        model=model,
        max_pages=max_pages,
        time_period=time_period,
    )

    logger.info(
        "Pipeline complete | charts=%d types=%s",
        len(visualizations),
        [v.chart_type for v in visualizations],
    )
    return PipelineResult(visualizations=visualizations)
