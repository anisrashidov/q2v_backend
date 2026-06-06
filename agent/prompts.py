"""LLM prompt constants for every stage that calls an LLM.

Centralising prompts here means:
* All prompt text is visible and editable in one file.
* agent/interpret.py and agent/visualize.py stay focused on logic.
* Prompt schemas can be unit-tested without running the pipeline.
"""
from __future__ import annotations

# ── Stage 1: Interpret ─────────────────────────────────────────────────────────

INTERPRET_SYSTEM = """\
You are a clinical-trial search assistant.  Convert the user's natural-language
question (plus any optional structured hints) into search parameters for the
ClinicalTrials.gov v2 API.

Field guide
───────────
query_cond  – The medical condition or disease being studied.
              Examples: "type 2 diabetes", "non-small cell lung cancer".

query_term  – General keywords that don't fit condition or intervention.
              Use sparingly; prefer query_cond / query_intr when possible.

query_intr  – Specific intervention, treatment, or drug.
              Examples: "metformin", "CAR-T cell therapy", "radiation therapy".

filter_overall_status – Only set when the question EXPLICITLY asks about status.
  Valid values (use exactly as written):
  RECRUITING | NOT_YET_RECRUITING | ACTIVE_NOT_RECRUITING | COMPLETED |
  SUSPENDED | TERMINATED | WITHDRAWN | AVAILABLE | NO_LONGER_AVAILABLE |
  TEMPORARILY_NOT_AVAILABLE | APPROVED_FOR_MARKETING | WITHHELD | UNKNOWN

filter_phase – Only set when the question EXPLICITLY mentions a trial phase.
  Valid values (use exactly as written):
  EARLY_PHASE1 | PHASE1 | PHASE2 | PHASE3 | PHASE4 | NA

filter_funder_type – Only set when the question EXPLICITLY asks about the
  funding source or sponsor type.
  Valid values (use exactly as written):
  NIH | OTHER_GOV | INDIV | INDUSTRY | OTHER | FED | NETWORK | UNKNOWN

page_size   – Default 50.  Increase (up to 1 000) when the user asks for a
              broad overview or trend analysis.  Decrease for narrow lookups.

sort        – Omit unless the user asks for a specific order.
              Common values: "@relevance", "StartDate:desc", "EnrollmentCount:desc".

query_spons – Sponsor or organisation name to match.
              Examples: "Pfizer", "National Cancer Institute", "Johns Hopkins".

query_locn  – Location name (country, state, or city) to restrict results to.
              Examples: "United States", "Germany", "New York".

filter_study_type – Only set when the question EXPLICITLY mentions study type.
  Valid values (use exactly as written):
  INTERVENTIONAL | OBSERVATIONAL | EXPANDED_ACCESS

filter_sex  – Only set when the question EXPLICITLY mentions participant sex.
  Valid values (use exactly as written): female | male | all

filter_age_range – Only set when the question EXPLICITLY mentions age groups.
  Valid values (use exactly as written): child | adult | older
  Multiple values allowed (e.g. ["adult", "older"]).

filter_start_year – Integer year (e.g. 2020). Only when an explicit lower
  bound on trial start date is mentioned.

filter_end_year – Integer year (e.g. 2024). Only when an explicit upper
  bound on trial start date is mentioned.

Rules
─────
* Leave a field null / omit it if you cannot confidently infer it.
* Do NOT invent statuses or values not listed above.
* Multiple statuses go in the array (e.g. ["RECRUITING", "NOT_YET_RECRUITING"]).
"""

INTERPRET_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query_cond": {
            "type": "string",
            "description": "Condition or disease (query.cond)",
        },
        "query_term": {
            "type": "string",
            "description": "General keyword (query.term)",
        },
        "query_intr": {
            "type": "string",
            "description": "Intervention or drug (query.intr)",
        },
        "filter_overall_status": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "RECRUITING",
                    "NOT_YET_RECRUITING",
                    "ACTIVE_NOT_RECRUITING",
                    "COMPLETED",
                    "SUSPENDED",
                    "TERMINATED",
                    "WITHDRAWN",
                    "AVAILABLE",
                    "NO_LONGER_AVAILABLE",
                    "TEMPORARILY_NOT_AVAILABLE",
                    "APPROVED_FOR_MARKETING",
                    "WITHHELD",
                    "UNKNOWN",
                ],
            },
            "description": "Trial status filter (filter.overallStatus)",
        },
        "filter_phase": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"],
            },
            "description": "Trial phase filter (aggFilters=phase:...)",
        },
        "filter_funder_type": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["NIH", "OTHER_GOV", "INDIV", "INDUSTRY", "OTHER", "FED", "NETWORK", "UNKNOWN"],
            },
            "description": "Funder/sponsor type filter (aggFilters=funderType:...)",
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific protocol fields to request",
        },
        "page_size": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "description": "Number of studies to fetch (pageSize)",
        },
        "sort": {
            "type": "string",
            "description": "Sort expression (sort)",
        },
        "query_spons": {
            "type": "string",
            "description": "Sponsor or organisation name (query.spons)",
        },
        "query_locn": {
            "type": "string",
            "description": "Location name — country, state, or city (query.locn)",
        },
        "filter_study_type": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"],
            },
            "description": "Study type filter (aggFilters=studyType:...)",
        },
        "filter_sex": {
            "type": "string",
            "enum": ["female", "male", "all"],
            "description": "Participant sex filter (aggFilters=sex:...)",
        },
        "filter_age_range": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["child", "adult", "older"],
            },
            "description": "Age group filter (aggFilters=ageRange:...)",
        },
        "filter_start_year": {
            "type": "integer",
            "description": "Earliest trial start year (filter.advanced date range)",
        },
        "filter_end_year": {
            "type": "integer",
            "description": "Latest trial start year (filter.advanced date range)",
        },
    },
    "required": [],
    "additionalProperties": False,
}


# ── Stage 3b: Decide ───────────────────────────────────────────────────────────

DECISION_SYSTEM = """\
You are a data-visualisation expert working with aggregated clinical trial data.

You will receive:
1. The original user question.
2. Pre-computed aggregations (you CANNOT change any numbers in them).

Your job: choose the best chart type and which single aggregation to display.

Chart-type selection rules
──────────────────────────
bar     → comparing counts across ≤20 discrete categories (e.g. phases, statuses)
line    → data keyed by an ordered time dimension (year/month trend)
pie     → proportions / share; ONLY when ≤7 categories AND the question is
          about percentage or distribution
scatter → correlation between two numeric dimensions (rarely applicable here)
table   → user asks for a list of studies or wants to see individual records
none    → question cannot be meaningfully visualised (definitions, out-of-scope)

CRITICAL constraints
────────────────────
* You only CHOOSE the type and the aggregation_key.
* You NEVER invent, modify, or recompute any numbers.
* aggregation_key MUST be one of the exact keys listed in the schema.
"""

DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chart_type": {
            "type": "string",
            "enum": ["bar", "line", "pie", "scatter", "table", "none"],
            "description": "The visualisation type to render",
        },
        "aggregation_key": {
            "type": "string",
            "enum": [
                "by_phase",
                "by_status",
                "by_year",
                "by_study_type",
                "by_sponsor_class",
                "enrollment_buckets",
                "top_conditions",
            ],
            "description": "Which pre-computed AggregatedData field to display",
        },
        "title": {
            "type": "string",
            "description": "Concise, human-readable chart title",
        },
        "x_label": {
            "type": "string",
            "description": "X-axis label shown to the user",
        },
        "y_label": {
            "type": "string",
            "description": "Y-axis label shown to the user",
        },
        "description": {
            "type": "string",
            "description": "One sentence explaining the chart in plain language",
        },
    },
    "required": ["chart_type", "aggregation_key", "title", "x_label", "y_label"],
    "additionalProperties": False,
}
