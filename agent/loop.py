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
from agent.types import VisualizationSpec

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 20

async def run_agent(
    question: str,
    ct_api: ClinicalTrialsAPI,
    openai_client: openai.AsyncOpenAI,
    model: str,
    max_pages: int = 5,
    time_period: Optional[Tuple[date, date]] = None,
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
    param_overrides:
        Optional QueryParams-style field overrides from the request body.
        Translated to search_trials parameter names and injected as explicit
        constraints in the user message so the LLM honours them.

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

    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    results: List[VisualizationSpec] = []
    search_called = False

    with ToolRegistry(ct_api=ct_api, max_pages=max_pages) as registry:
        iteration = 1
        factor = 1
        while iteration <= _MAX_ITERATIONS * factor:
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

                    if name == "search_trials":
                        if iteration == 1:
                            _SEARCH_FILTER_KEYS = {
                                "condition", "drug", "sponsor", "phase", "status",
                                "study_type", "intervention_type", "location",
                                "start_year", "end_year", "sex", "age_group",
                            }
                            if not any(kwargs.get(k) for k in _SEARCH_FILTER_KEYS):
                                raise ValueError(
                                    "Your query does not contain enough information to search clinical trials. "
                                    "Please include at least one filter such as a condition, drug name, "
                                    "sponsor, phase, status, or location."
                                )
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
                        logger.debug(f"MAX_ITER factor increased by 1.5 times from {factor} to {factor * 1.5}")
                        factor *= 1.5
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
            f"Agent did not call build_visualization within {_MAX_ITERATIONS} iterations"
        )
