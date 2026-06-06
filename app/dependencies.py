"""FastAPI dependency providers.

These are injected via ``Depends()`` into route handlers, keeping the router
decoupled from ``app.state`` access details.
"""
from __future__ import annotations

from fastapi import Request

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.ports import LLMPort


def get_llm(request: Request) -> LLMPort:
    """Return the shared LLM adapter stored in app state."""
    return request.app.state.llm


def get_ct_api(request: Request) -> ClinicalTrialsAPI:
    """Return the shared ClinicalTrials API adapter stored in app state."""
    return request.app.state.ct_api


def get_max_pages(request: Request) -> int:
    """Return the configured maximum pagination depth."""
    return getattr(request.app.state, "ct_max_pages", 5)
