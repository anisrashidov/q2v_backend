"""FastAPI application entry-point.

Run with:
    python -m uvicorn app.main:app --reload

The lifespan context manager owns shared resources:
* An ``aiohttp.ClientSession`` shared across all requests.
* An ``OpenAILLM`` instance (stateless, thread-safe).
* A ``ClinicalTrialsAPI`` instance that reuses the shared HTTP client.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import aiohttp
import openai
from fastapi import FastAPI

from adapters.clinicaltrials import ClinicalTrialsAPI
from app.config import settings
from app.routers import meta as meta_router
from app.routers import query as query_router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s - %(message)s",
)
logging.getLogger("app").setLevel(logging.DEBUG)
logging.getLogger("agent").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting the http_client")
    http_client = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=settings.ct_timeout),
    )

    logger.info("Starting the OpenAI client")
    app.state.llm = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    app.state.llm_model = settings.openai_model

    logger.info("Starting the Clinical Trials API client")
    app.state.ct_api = ClinicalTrialsAPI(
        client=http_client,
        base_url=f"{settings.ct_base_url}/studies",
        max_retries=settings.ct_max_retries,
    )
    app.state.ct_max_pages = settings.ct_max_pages

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await http_client.close()


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Transforms natural-language questions about clinical trials into "
        "rich, library-agnostic visualization specifications using the "
        "ClinicalTrials.gov v2 API and OpenAI."
    ),
    lifespan=lifespan,
    debug=settings.debug,
)

app.include_router(meta_router.router)
app.include_router(query_router.router, prefix="/api", tags=["query"])
