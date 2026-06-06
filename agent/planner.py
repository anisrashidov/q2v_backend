"""Planning stage — produces a structured QueryPlan before execution.

plan_query() makes a single LLM call with structured outputs (no tools) and
returns a validated QueryPlan ("thinking process"). It does NOT retrieve data
or build visualizations; the execution loop (agent/loop.py) does that.

Step 1 of the planner rollout: the plan is produced and logged but does not yet
gate or seed execution — see pipeline.py.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional, Tuple

import openai

from agent.prompts import PLANNER_SYSTEM_PROMPT
from agent.types import QueryPlan

logger = logging.getLogger(__name__)


async def plan_query(
    question: str,
    openai_client: openai.AsyncOpenAI,
    model: str,
    time_period: Optional[Tuple[date, date]] = None,
) -> QueryPlan:
    """Produce a structured QueryPlan for *question* via a single LLM call.

    Parameters mirror run_agent so the planner can share the request's model and
    default time window. Returns a validated QueryPlan; raises on refusal or a
    response the model could not structure.
    """
    user_content = f"Question: {question}"
    if time_period:
        start, end = time_period
        user_content += (
            f"\nDefault time period: {start} to {end} "
            f"(start_year={start.year}, end_year={end.year}). "
            f"If the question specifies a different date range, use that instead."
        )

    completion = await openai_client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=QueryPlan,
        temperature=0.1,
    )

    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Planner refused to plan the query: {message.refusal}")
    plan = message.parsed
    if plan is None:
        raise ValueError("Planner returned no parsable plan.")

    logger.info(
        "Plan | clinical=%s clarify=%s charts=%s",
        plan.is_clinical_trial_query,
        plan.needs_clarification,
        [c.chart_type.value for c in plan.charts],
    )
    return plan
