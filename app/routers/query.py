"""POST /api/query — the single public endpoint.

Accepts a natural-language question, runs the 4-stage pipeline, and returns
interpreted parameters, a data summary, and a visualization specification.

Dependencies are injected via ``Depends()`` so the handler never touches
``app.state`` directly, making it trivial to test with mock adapters.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.pipeline import run_pipeline
from agent.ports import LLMPort
from agent.types import PipelineResult
from app.dependencies import get_ct_api, get_llm, get_max_pages

router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Body accepted by POST /api/query.

    Only ``query`` is required.  All other fields are optional structured hints
    that override the LLM's interpretation of the natural-language query.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How many phase 3 cancer trials are currently recruiting?"
            }
        }
    )

    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language question about clinical trials",
        examples=["How many phase 3 cancer trials are currently recruiting?"],
    )

    # ── Structured overrides (all optional) ────────────────────────────────
    condition: Optional[str] = Field(
        None, description="Medical condition or disease (overrides query.cond)"
    )
    drug_name: Optional[str] = Field(
        None, description="Drug or intervention name (overrides query.intr)"
    )
    sponsor: Optional[str] = Field(
        None, description="Sponsor or organisation name (overrides query.spons)"
    )
    trial_phase: Optional[str] = Field(
        None,
        description=(
            "Trial phase override (aggFilters=phase:...). "
            "Valid: EARLY_PHASE1 | PHASE1 | PHASE2 | PHASE3 | PHASE4 | NA"
        ),
    )
    recruitment_status: Optional[str] = Field(
        None,
        description=(
            "Recruitment status override (filter.overallStatus). "
            "Valid: RECRUITING | NOT_YET_RECRUITING | COMPLETED | TERMINATED | WITHDRAWN | SUSPENDED"
        ),
    )
    study_type: Optional[str] = Field(
        None,
        description=(
            "Study type override (aggFilters=studyType:...). "
            "Valid: INTERVENTIONAL | OBSERVATIONAL | EXPANDED_ACCESS"
        ),
    )
    intervention_type: Optional[str] = Field(
        None,
        description=(
            "Intervention type filter (filter.advanced AREA[InterventionType]...). "
            "Valid: DRUG | DEVICE | PROCEDURE | BIOLOGICAL | BEHAVIORAL | OTHER"
        ),
    )
    country: Optional[str] = Field(
        None, description="Country to restrict results to (query.locn)"
    )
    city: Optional[str] = Field(
        None, description="City to restrict results to (query.locn, combined with country)"
    )
    start_year: Optional[int] = Field(
        None, description="Earliest trial start year (filter.advanced date range)"
    )
    end_year: Optional[int] = Field(
        None, description="Latest trial start year (filter.advanced date range)"
    )
    age_group: Optional[str] = Field(
        None,
        description=(
            "Age group filter (aggFilters=ageRange:...). "
            "Valid: child | adult | older"
        ),
    )
    sex: Optional[str] = Field(
        None,
        description="Participant sex filter (aggFilters=sex:...). Valid: female | male | all",
    )
    page_size: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Number of studies to return per page (1–1000, default 50)",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_overrides(body: QueryRequest) -> dict:
    """Translate explicit ``QueryRequest`` fields into a ``QueryParams`` override dict."""
    overrides: dict = {}

    if body.condition:
        overrides["query_cond"] = body.condition
    if body.drug_name:
        overrides["query_intr"] = body.drug_name
    if body.sponsor:
        overrides["query_spons"] = body.sponsor
    if body.trial_phase:
        overrides["filter_phase"] = [body.trial_phase.upper()]
    if body.recruitment_status:
        overrides["filter_overall_status"] = [body.recruitment_status.upper()]
    if body.study_type:
        overrides["filter_study_type"] = [body.study_type.upper()]
    if body.intervention_type:
        overrides["filter_intervention_type"] = body.intervention_type.upper()

    location_parts = [p for p in [body.city, body.country] if p]
    if location_parts:
        overrides["query_locn"] = " ".join(location_parts)

    if body.start_year is not None:
        overrides["filter_start_year"] = body.start_year
    if body.end_year is not None:
        overrides["filter_end_year"] = body.end_year
    if body.age_group:
        overrides["filter_age_range"] = [body.age_group.lower()]
    if body.sex:
        overrides["filter_sex"] = body.sex.lower()
    if body.page_size is not None:
        overrides["page_size"] = body.page_size

    return overrides


# ── Route ──────────────────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=PipelineResult,
    summary="Run the NL → visualization pipeline",
    response_description=(
        "Interpreted query params, data summary, and a visualization spec "
        "ready for the frontend to render."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    llm: LLMPort = Depends(get_llm),
    ct_api: ClinicalTrialsAPI = Depends(get_ct_api),
    max_pages: int = Depends(get_max_pages),
) -> PipelineResult:
    """Execute the four-stage pipeline and return a PipelineResult."""
    overrides = _build_overrides(body)
    try:
        return await run_pipeline(
            question=body.query,
            llm=llm,
            ct_api=ct_api,
            max_pages=max_pages,
            param_overrides=overrides or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}",
        ) from exc
