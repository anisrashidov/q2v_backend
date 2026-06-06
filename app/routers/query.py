"""POST /api/query — the single public endpoint."""
from __future__ import annotations

import openai
from fastapi import APIRouter, Depends, HTTPException

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.pipeline import run_pipeline
from app.dependencies import get_ct_api, get_llm, get_max_pages, get_model
from app.routers.schemas import BaseResponse, QueryRequest

router = APIRouter()


@router.post(
    "/query",
    response_model=BaseResponse,
    summary="Run the NL → visualization pipeline",
    response_description="Visualization result or a client-facing error envelope.",
)
async def query_endpoint(
    body: QueryRequest,
    openai_client: openai.AsyncOpenAI = Depends(get_llm),
    model: str = Depends(get_model),
    ct_api: ClinicalTrialsAPI = Depends(get_ct_api),
    max_pages: int = Depends(get_max_pages),
) -> BaseResponse:
    try:
        result = await run_pipeline(
            question=body.query,
            openai_client=openai_client,
            model=model,
            ct_api=ct_api,
            max_pages=max_pages,
            time_period=body.time_period,
        )
        return BaseResponse(code=200, message="OK", data=result)

    except ValueError as exc:
        return BaseResponse(code=400, message=str(exc), data=None)

    except openai.AuthenticationError:
        return BaseResponse(code=401, message="Invalid OpenAI API key.", data=None)

    except openai.RateLimitError:
        return BaseResponse(code=429, message="OpenAI rate limit or budget exceeded. Please try again later.", data=None)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc
