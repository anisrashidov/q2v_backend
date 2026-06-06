from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseResponse(BaseModel):
    """Envelope for all /api/query responses.

    HTTP status is always 200 for client-facing errors; the `code` field carries
    the semantic status and `message` describes the outcome.
    """

    code: int = Field(..., description="200 success · 400 bad input · 401 auth · 429 rate-limit")
    message: str = Field(..., description="Human-readable outcome description")
    data: Any = Field(None, description="PipelineResult on success, null on error")


class QueryRequest(BaseModel):
    """Body accepted by POST /api/query."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How many phase 3 cancer trials are currently recruiting?",
                "time_period": ["2018-01-01", "2023-12-31"],
            }
        }
    )

    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language question about clinical trials",
        examples=["How many phase 3 cancer trials are currently recruiting?"],
    )
    time_period: Optional[tuple[date, date]] = Field(
        None,
        description="Inclusive date range [start, end] to restrict trial start dates.",
        examples=[["2018-01-01", "2023-12-31"]],
    )
