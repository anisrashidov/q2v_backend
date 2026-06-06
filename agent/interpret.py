"""Stage 1 — INTERPRET.

Converts a natural-language question (plus optional structured hints) into a
validated QueryParams model using an LLM.

Prompt constants live in agent/prompts.py so they can be read and edited
without touching pipeline logic.
"""
from __future__ import annotations

import json
from typing import Optional

from agent.ports import LLMPort
from agent.prompts import INTERPRET_SCHEMA, INTERPRET_SYSTEM
from agent.types import QueryParams


async def interpret_query(
    question: str,
    llm: Optional[LLMPort] = None,
) -> QueryParams:
    """Stage 1 — convert *question* to a validated :class:`QueryParams`.

    Parameters
    ----------
    question:
        Raw natural-language question from the user.
    extra_fields:
        Optional dict of structured hints appended to the user message.
    llm:
        Any object implementing :class:`agent.ports.LLMPort`.
    """
    if llm is None:
        raise ValueError("An LLMPort implementation must be provided.")

    user_msg = f"Question: {question}"

    raw = await llm.complete(
        system=INTERPRET_SYSTEM, user=user_msg, schema=INTERPRET_SCHEMA
    )

    # Drop null/None so Pydantic uses its own defaults
    clean = {k: v for k, v in raw.items() if v is not None}
    return QueryParams(**clean)
