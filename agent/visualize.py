"""Stage 3b — DECIDE.

LLM classifies the user's question into a chart type and selects which
pre-computed aggregation to visualise.

The LLM receives the aggregated numbers and READS them; it never computes,
modifies, or invents data.  Prompt constants live in agent/prompts.py.
"""
from __future__ import annotations

import json

from agent.ports import LLMPort
from agent.prompts import DECISION_SCHEMA, DECISION_SYSTEM
from agent.types import AggregatedData


async def decide_visualization(
    question: str,
    aggregated: AggregatedData,
    llm: LLMPort,
) -> dict:
    """Stage 3b — ask the LLM to choose a chart type and aggregation key.

    Parameters
    ----------
    question:
        Original user question (context for the LLM's decision).
    aggregated:
        Pre-computed aggregations from :func:`agent.aggregate.aggregate_studies`.
    llm:
        Any object implementing :class:`agent.ports.LLMPort`.

    Returns
    -------
    dict
        Raw decision dict conforming to ``DECISION_SCHEMA``.
        Pass this directly to :func:`agent.spec.build_spec`.
    """
    user_msg = (
        f"User question: {question}\n\n"
        f"Pre-computed aggregations:\n"
        f"{json.dumps(aggregated.model_dump(), indent=2)}"
    )
    return await llm.complete(
        system=DECISION_SYSTEM,
        user=user_msg,
        schema=DECISION_SCHEMA,
    )
