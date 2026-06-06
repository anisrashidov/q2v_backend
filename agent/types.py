"""Core Pydantic types shared across all agent layers.

No internal project imports — this is the leaf of the dependency graph.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

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
    countries: List[str] = Field(default_factory=list)

    @classmethod
    def from_api(cls, raw: dict) -> "Study":
        """Parse a study from a raw ClinicalTrials.gov v2 API response dict."""
        proto = raw.get("protocolSection", {})

        id_mod      = proto.get("identificationModule", {})
        status_mod  = proto.get("statusModule", {})
        design_mod  = proto.get("designModule", {})
        cond_mod    = proto.get("conditionsModule", {})
        arms_mod    = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        loc_mod     = proto.get("contactsLocationsModule", {})

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

        countries = list({
            loc.get("country", "")
            for loc in loc_mod.get("locations", [])
            if loc.get("country")
        })

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
            countries=countries,
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
    """Supported chart types.

    bar_chart         → discrete category comparison
    time_series       → ordered time axis (year/month trends)
    scatter_plot      → two continuous axes
    histogram         → continuous value distribution (bin_continuous output)
    network_graph     → nodes and edges (co-occurrence / relationship data)
    choropleth_map    → geographic fill map (aggregate_by_country output)
    none              → single-value answer, no meaningful visual
    """

    bar_chart         = "bar_chart"
    time_series       = "time_series"
    scatter_plot      = "scatter_plot"
    histogram         = "histogram"
    network_graph     = "network_graph"
    choropleth_map    = "choropleth_map"
    none              = "none"
    table             = "table"          # table → ranked or multi-column tabular listing
    heatmap           = "heatmap"        # heatmap → two categorical axes with a numeric intensity


class DataPoint(BaseModel):
    """Kept for backward compatibility with existing test fixtures."""

    label: str  = Field(description="Category label")
    value: float = Field(description="Numeric measure")
    group: Optional[str] = Field(None, description="Optional grouping key")


class VisualizationSpec(BaseModel):
    """Library-agnostic chart specification.

    encoding       — maps semantic roles to field names, e.g. {"x": "label", "y": "value"}
    data           — list of records; shape depends on chart_type
    metadata       — provenance, warnings, annotations, filters applied
    sequence_index — 0-based position within a multi-chart response (auto-assigned by the loop)
    group          — semantic label for this chart in a dashboard, e.g. "trends", "distribution"
    """

    chart_type: ChartType
    title: str
    description: Optional[str] = None
    encoding: Dict[str, Any] = Field(default_factory=dict)
    data: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    total_records: int = Field(0, description="Total studies in the underlying dataset")
    sequence_index: int = Field(0, description="Position within a multi-chart response")
    group: Optional[str] = Field(None, description="Semantic label for this chart, e.g. 'trends'")


# ── Query plan (planning stage) ──────────────────────────────────────────────────


class PlannedChart(BaseModel):
    """One chart the planner intends to produce, before any tools run.

    chart_type is validated against ChartType, so the planner cannot propose a
    type the pipeline can't render. tool_sequence is advisory — execution may
    deviate when real tool output requires it.
    """

    chart_type: ChartType
    title: str = Field(description="Working title for the intended chart")
    rationale: str = Field(description="Why this chart answers the question")
    tool_sequence: List[str] = Field(
        default_factory=list,
        description="Planned tool call order, e.g. ['search_trials','aggregate_by','build_visualization']",
    )


class QueryPlan(BaseModel):
    """Structured 'thinking process' produced before the execution loop.

    The pipeline gates on needs_clarification (deterministic), not on whether
    charts were proposed — a single-value answer legitimately has no real chart.
    """

    is_clinical_trial_query: bool = Field(
        description="False if the question is not about clinical trials at all"
    )
    needs_clarification: bool = Field(
        description="True if the query lacks enough detail to search (no condition, drug, sponsor, etc.)"
    )
    clarification_message: Optional[str] = Field(
        None, description="If needs_clarification, the question to ask the user"
    )
    search_strategy: str = Field(
        description="How to search: entities, filters, and time window to apply"
    )
    charts: List[PlannedChart] = Field(
        default_factory=list,
        description=(
            "Charts the execution loop should aim to produce. Default to exactly ONE; "
            "include multiple only when the question explicitly asks for several or "
            "unambiguously implies a dashboard. Empty if needs_clarification is true."
        ),
    )


# ── Pipeline result ────────────────────────────────────────────────────────────


class DataSummary(BaseModel):
    total_count: int = Field(description="Total matching studies reported by the API")
    retrieved:   int = Field(description="Studies actually in the response payload")
    has_more:    bool = Field(description="True when more pages exist beyond what was fetched")


class PipelineResult(BaseModel):
    """Top-level response envelope returned by POST /api/query."""

    visualizations: List[VisualizationSpec]
    interpreted_params: Optional[QueryParams] = None
    data_summary:       Optional[DataSummary] = None
