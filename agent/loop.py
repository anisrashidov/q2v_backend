"""Agentic tool-calling loop.

run_agent() manages a local `messages` list (the agent's working memory for
one query) and dispatches OpenAI tool calls until the LLM stops.
build_visualization may be called multiple times; the loop accumulates all
results and returns them as an ordered list.  The messages list is local
state and is discarded on return.
"""
from __future__ import annotations

import inspect
import json
import logging
from datetime import date
from typing import Any, List, Optional, Tuple

import openai

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, ToolRegistry
from agent.types import QueryPlan, VisualizationSpec

logger = logging.getLogger(__name__)

# Iteration budget scales with how many charts the plan intends, so a dashboard
# query gets more room than a single-chart query without an open-ended loop.
_BASE_ITERATIONS = 15
_ITERATIONS_PER_CHART = 6


def _format_plan(plan: QueryPlan) -> str:
    """Render a QueryPlan as guidance text injected into the execution loop."""
    lines = [
        "PLAN (follow this unless tool results require deviating):",
        f"Search strategy: {plan.search_strategy}",
        "Intended charts:",
    ]
    for i, chart in enumerate(plan.charts, 1):
        lines.append(f'  {i}. {chart.chart_type.value} — "{chart.title}" ({chart.rationale})')
        if chart.tool_sequence:
            lines.append(f"     tools: {' → '.join(chart.tool_sequence)}")
    return "\n".join(lines)


async def run_agent(
    question: str,
    ct_api: ClinicalTrialsAPI,
    openai_client: openai.AsyncOpenAI,
    model: str,
    max_pages: int = 5,
    time_period: Optional[Tuple[date, date]] = None,
    plan: Optional[QueryPlan] = None,
) -> List[VisualizationSpec]:
    """Run the agentic tool-calling loop for one user question.

    Parameters
    ----------
    question:
        Raw natural-language question from the user.
    ct_api:
        ClinicalTrialsAPI adapter for searching studies.
    openai_client:
        Shared async OpenAI client from the app lifespan.
    model:
        OpenAI model ID (e.g. "gpt-4.1").
    max_pages:
        Maximum pagination depth forwarded to search_trials.
    time_period:
        Optional default [start, end] date range injected into the user message.
    plan:
        Optional QueryPlan from the planning stage. When present it is injected as
        guidance and its chart count sizes the iteration budget. Execution may
        still deviate when real tool output requires it.

    Returns
    -------
    List[VisualizationSpec]
        One or more specs ordered by sequence_index.  A simple query returns
        a single-element list; a dashboard query returns multiple elements.
    """
    user_content = f"Question: {question}"
    if time_period:
        start, end = time_period
        user_content += (
            f"\nDefault time period: {start} to {end} "
            f"(start_year={start.year}, end_year={end.year}). "
            f"If the question specifies a different date range, use that instead."
        )
    if plan is not None:
        user_content += "\n\n" + _format_plan(plan)

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    results: List[VisualizationSpec] = []
    search_called = False

    expected_charts = len(plan.charts) if (plan and plan.charts) else 1
    max_iterations = _BASE_ITERATIONS + _ITERATIONS_PER_CHART * expected_charts

    with ToolRegistry(ct_api=ct_api, max_pages=max_pages) as registry:
        iteration = 1
        while iteration <= max_iterations:
            logger.debug("Agent iteration %d | messages=%d", iteration, len(messages))

            response = await openai_client.chat.completions.create(
                model=model,
                tools=TOOL_SCHEMAS,
                # Force a tool call on the first turn so the LLM can't answer in plain text
                tool_choice="required" if iteration == 1 else "auto",
                messages=messages,
                temperature=0.1
            )

            choice = response.choices[0]
            logger.debug(
                "Agent iteration %d | finish_reason=%s | charts_so_far=%d",
                iteration, choice.finish_reason, len(results),
            )

            # Append the assistant message to maintain conversation context
            assistant_msg: dict = {"role": "assistant"}
            if choice.message.content:
                assistant_msg["content"] = choice.message.content
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            if choice.finish_reason == "tool_calls":
                for tc in choice.message.tool_calls:
                    name = tc.function.name
                    kwargs = json.loads(tc.function.arguments)
                    logger.debug("Tool call | %s | args=%s", name, list(kwargs))

                    if name not in registry:
                        raise ValueError(f"LLM invoked unknown tool: {name!r}")

                    # Query sufficiency is enforced upstream by the planning gate
                    # (pipeline.run_pipeline). Here we only track that a search
                    # happened so we never build a chart from no data.
                    if name == "search_trials":
                        search_called = True

                    if name == "build_visualization" and not search_called:
                        raise ValueError(
                            "Your query does not appear to be a clinical-trial question. "
                            "Please ask something about clinical trials."
                        )

                    fn = registry[name]
                    result = await fn(**kwargs) if inspect.iscoroutinefunction(fn) else fn(**kwargs)

                    if name == "build_visualization":
                        # Auto-assign sequence_index in call order; group is LLM-provided
                        result.sequence_index = len(results)
                        results.append(result)
                        result_payload: Any = result.model_dump()
                        logger.debug(
                            "build_visualization | sequence_index=%d group=%s chart_type=%s",
                            result.sequence_index, result.group, result.chart_type,
                        )
                    else:
                        result_payload = result

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result_payload, default=str),
                    })

            elif choice.finish_reason == "stop":
                if results:
                    logger.info(
                        "Agent complete | charts=%d iterations=%d",
                        len(results), iteration,
                    )
                    return results
                # LLM stopped before calling build_visualization — nudge it
                logger.warning("Agent stopped before build_visualization; prompting to continue")
                messages.append({
                    "role": "user",
                    "content": "You must call build_visualization at least once to complete your answer.",
                })

            else:
                raise RuntimeError(f"Unexpected finish_reason: {choice.finish_reason!r}")

            iteration += 1

        raise RuntimeError(
            f"Agent did not call build_visualization within {max_iterations} iterations"
        )
