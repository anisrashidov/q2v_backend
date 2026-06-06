"""FastAPI dependency providers.

These are injected via ``Depends()`` into route handlers, keeping the router
decoupled from ``app.state`` access details.
"""
from __future__ import annotations

import openai
from fastapi import Request

from adapters.clinicaltrials import ClinicalTrialsAPI


def get_llm(request: Request) -> openai.AsyncOpenAI:
    """Return the shared OpenAI client stored in app state."""
    return request.app.state.llm


def get_model(request: Request) -> str:
    """Return the configured OpenAI model name stored in app state."""
    return request.app.state.llm_model


def get_ct_api(request: Request) -> ClinicalTrialsAPI:
    """Return the shared ClinicalTrials API adapter stored in app state."""
    return request.app.state.ct_api


def get_max_pages(request: Request) -> int:
    """Return the configured maximum pagination depth."""
    return getattr(request.app.state, "ct_max_pages", 5)
