"""Core Pydantic types shared across all agent layers.

No internal project imports — this is the leaf of the dependency graph.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Query parameters ───────────────────────────────────────────────────────────


class QueryParams(BaseModel):
    """Validated parameters for the ClinicalTrials.gov v2 API.

    Field → API-param mapping
    ─────────────────────────
    query_cond              → query.cond
    query_term              → query.term
    query_intr              → query.intr
    filter_overall_status   → filter.overallStatus  (comma-joined)
    filter_phase            → aggFilters=phase:N,N  (numeric mapped)
    filter_funder_type      → aggFilters=funderType:V,V
    fields                  → fields                (comma-joined)
    page_size               → pageSize
    sort                    → sort
    """

    query_cond: Optional[str] = Field(
        None, description="Condition/disease search term (query.cond)"
    )
    query_term: Optional[str] = Field(
        None, description="General keyword search (query.term)"
    )
    query_intr: Optional[str] = Field(
        None, description="Intervention/treatment search (query.intr)"
    )
    filter_overall_status: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by overall status. "
            "Valid: RECRUITING | NOT_YET_RECRUITING | ACTIVE_NOT_RECRUITING | "
            "COMPLETED | SUSPENDED | TERMINATED | WITHDRAWN | "
            "AVAILABLE | NO_LONGER_AVAILABLE | TEMPORARILY_NOT_AVAILABLE | "
            "APPROVED_FOR_MARKETING | WITHHELD | UNKNOWN"
        ),
    )
    filter_phase: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by trial phase (aggFilters=phase:...). "
            "Valid: EARLY_PHASE1 | PHASE1 | PHASE2 | PHASE3 | PHASE4 | NA"
        ),
    )
    filter_funder_type: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by funder/sponsor type (aggFilters=funderType:...). "
            "Valid: NIH | OTHER_GOV | INDIV | INDUSTRY | OTHER | FED | NETWORK | UNKNOWN"
        ),
    )
    fields: Optional[List[str]] = Field(
        None, description="Specific protocol-section fields to return from the API"
    )
    page_size: int = Field(
        default=50, ge=1, le=1000, description="Studies per page (1–1 000)"
    )
    sort: Optional[str] = Field(
        None,
        description='Sort expression e.g. "@relevance" or "StartDate:desc"',
    )
    query_spons: Optional[str] = Field(
        None, description="Sponsor/organization search (query.spons)"
    )
    query_locn: Optional[str] = Field(
        None, description="Location name search — country, state, or city (query.locn)"
    )
    filter_study_type: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by study type (aggFilters=studyType:...). "
            "Valid: INTERVENTIONAL | OBSERVATIONAL | EXPANDED_ACCESS"
        ),
    )
    filter_sex: Optional[str] = Field(
        None,
        description="Participant sex filter (aggFilters=sex:...). Valid: female | male | all",
    )
    filter_age_range: Optional[List[str]] = Field(
        None,
        description=(
            "Age group filter (aggFilters=ageRange:...). Valid: child | adult | older"
        ),
    )
    filter_start_year: Optional[int] = Field(
        None, description="Earliest trial start year (mapped to filter.advanced date range)"
    )
    filter_end_year: Optional[int] = Field(
        None, description="Latest trial start year (mapped to filter.advanced date range)"
    )
    filter_intervention_type: Optional[str] = Field(
        None,
        description=(
            "Intervention type filter (filter.advanced AREA[InterventionType]...). "
            "Examples: DRUG | DEVICE | PROCEDURE | BIOLOGICAL | BEHAVIORAL | OTHER"
        ),
    )

    @property
    def has_any_query(self) -> bool:
        return any([self.query_cond, self.query_term, self.query_intr])


# ── Study data ─────────────────────────────────────────────────────────────────


class Study(BaseModel):
    """Normalised representation of a single clinical trial study."""

    nct_id: str
    brief_title: Optional[str] = None
    overall_status: Optional[str] = None
    phases: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    enrollment: Optional[int] = None
    sponsor: Optional[str] = None
    sponsor_class: Optional[str] = None  # NIH | INDUSTRY | OTHER | FED | …
    study_type: Optional[str] = None     # INTERVENTIONAL | OBSERVATIONAL | …

    @classmethod
    def from_api(cls, raw: dict) -> "Study":
        """Parse a study from a raw ClinicalTrials.gov v2 API response dict."""
        proto = raw.get("protocolSection", {})

        id_mod     = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        cond_mod   = proto.get("conditionsModule", {})
        arms_mod   = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        start_struct = status_mod.get("startDateStruct", {})
        comp_struct  = status_mod.get("completionDateStruct", {})

        interventions = [
            i.get("name", "")
            for i in arms_mod.get("interventions", [])
            if i.get("name")
        ]

        enroll_info = design_mod.get("enrollmentInfo", {})
        enrollment  = enroll_info.get("count")

        lead_sponsor = sponsor_mod.get("leadSponsor", {})

        return cls(
            nct_id=id_mod.get("nctId", ""),
            brief_title=id_mod.get("briefTitle"),
            overall_status=status_mod.get("overallStatus"),
            phases=design_mod.get("phases", []),
            start_date=start_struct.get("date"),
            completion_date=comp_struct.get("date"),
            conditions=cond_mod.get("conditions", []),
            interventions=interventions,
            enrollment=enrollment if isinstance(enrollment, int) else None,
            sponsor=lead_sponsor.get("name"),
            sponsor_class=lead_sponsor.get("class"),
            study_type=design_mod.get("studyType"),
        )


class StudyData(BaseModel):
    """Result of a ClinicalTrials.gov API search (one or more pages)."""

    studies: List[Study]
    total_count: int = 0
    next_page_token: Optional[str] = None


# ── Aggregated data ────────────────────────────────────────────────────────────


class AggregatedData(BaseModel):
    """Deterministic, pre-computed aggregations over a set of studies.

    The LLM reads these values but NEVER modifies or recomputes them.
    """

    by_phase: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_year: Dict[str, int] = Field(
        default_factory=dict, description="Keyed by 4-digit start year, sorted asc"
    )
    by_study_type: Dict[str, int] = Field(default_factory=dict)
    by_sponsor_class: Dict[str, int] = Field(default_factory=dict)
    enrollment_buckets: Dict[str, int] = Field(
        default_factory=dict,
        description='Keys: "1-100", "101-500", "501-1000", "1001+"',
    )
    top_conditions: Dict[str, int] = Field(
        default_factory=dict, description="Up to 10 most frequent conditions"
    )
    total_studies: int = 0


# ── Visualization spec ─────────────────────────────────────────────────────────


class ChartType(str, Enum):
    """Exhaustive set of supported chart types.

    To add a new type see README § Adding a New Visualization Type.
    """

    bar     = "bar"
    line    = "line"
    pie     = "pie"
    scatter = "scatter"
    table   = "table"
    none    = "none"


class DataPoint(BaseModel):
    """A single data point in the visualization series."""

    label: str  = Field(description="Category label (x-axis value)")
    value: float = Field(description="Numeric measure (y-axis value)")
    group: Optional[str] = Field(
        None, description="Optional grouping key for grouped/stacked charts"
    )


class VisualizationSpec(BaseModel):
    """Vega-Lite-inspired, library-agnostic chart specification."""

    chart_type: ChartType
    title: str
    description: Optional[str] = None
    x_field: Optional[str] = Field(None, description='DataPoint field for x-axis, e.g. "label"')
    y_field: Optional[str] = Field(None, description='DataPoint field for y-axis, e.g. "value"')
    series: List[DataPoint] = Field(default_factory=list)
    axis_labels: Dict[str, str] = Field(
        default_factory=dict,
        description='e.g. {"x": "Phase", "y": "Trials"}',
    )
    aggregation_applied: str = Field(
        description="Human-readable description of the aggregation used"
    )
    total_records: int = Field(0, description="Total studies in the underlying dataset")


# ── Pipeline result ────────────────────────────────────────────────────────────


class DataSummary(BaseModel):
    total_count: int = Field(description="Total matching studies reported by the API")
    retrieved:   int = Field(description="Studies actually in the response payload")
    has_more:    bool = Field(description="True when more pages exist beyond what was fetched")


class PipelineResult(BaseModel):
    """Top-level response envelope returned by POST /api/query."""

    interpreted_params: QueryParams
    data_summary:       DataSummary
    visualization_spec: VisualizationSpec
