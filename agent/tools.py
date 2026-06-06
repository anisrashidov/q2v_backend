"""Tool implementations and schemas for the agentic loop.

ToolRegistry is a context manager that holds per-request state (study_cache,
counter) and exposes tools via __getitem__ / __contains__.  Use it with the
`with` statement so the cache is always cleared on exit:

    with ToolRegistry(ct_api=ct_api, max_pages=max_pages) as registry:
        ...

TOOL_SCHEMAS is module-level and imported directly by agent/loop.py.

Adding a new tool requires three things:
1. Write the implementation (module-level for pure tools, method on ToolRegistry
   for stateful tools that need ct_api or study_cache).
2. Wire it in ToolRegistry._build_tools().
3. Append its schema to TOOL_SCHEMAS.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Optional

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.aggregate import aggregate_studies
from agent.types import ChartType, QueryParams, Study, VisualizationSpec

logger = logging.getLogger(__name__)

# ── Shared dimension helpers ───────────────────────────────────────────────────

# Used by aggregate_by (single-dim, via AggregatedData)
_DIMENSION_ATTR: dict[str, str] = {
    "phase": "by_phase",
    "status": "by_status",
    "year": "by_year",
    "study_type": "by_study_type",
    "sponsor": "by_sponsor_class",
    "enrollment": "enrollment_buckets",
    "condition": "top_conditions",
}

# Used by compute_co_occurrence
_ALL_DIMENSIONS = list(_DIMENSION_ATTR) + ["country", "intervention"]

# extract_field_values names fields in the plural (conditions, phases,
# interventions, countries); the dimension helpers below use the singular.
# The model frequently chains the two and carries the plural form over, so
# accept both interchangeably instead of failing the whole request.
_DIMENSION_ALIASES: dict[str, str] = {
    "phases": "phase",
    "statuses": "status",
    "years": "year",
    "study_types": "study_type",
    "sponsor_class": "sponsor",
    "sponsors": "sponsor",
    "conditions": "condition",
    "countries": "country",
    "interventions": "intervention",
}


def _normalize_dimension(dimension: str) -> str:
    """Map a plural/alias field name to its canonical singular dimension."""
    return _DIMENSION_ALIASES.get(dimension, dimension)


def _study_dim_values(study: Study, dimension: str) -> list[str]:
    """Return all values for *dimension* from a single study."""
    dimension = _normalize_dimension(dimension)
    if dimension == "phase":
        return study.phases if study.phases else ["N/A"]
    if dimension == "status":
        return [study.overall_status or "N/A"]
    if dimension == "year":
        if study.start_date and len(study.start_date) >= 4 and study.start_date[:4].isdigit():
            return [study.start_date[:4]]
        return ["N/A"]
    if dimension == "study_type":
        return [study.study_type or "N/A"]
    if dimension == "sponsor":
        return [study.sponsor_class or "N/A"]
    if dimension == "enrollment":
        e = study.enrollment
        if e is None:
            return ["Unknown"]
        if e <= 100:
            return ["1-100"]
        if e <= 500:
            return ["101-500"]
        if e <= 1000:
            return ["501-1000"]
        return ["1001+"]
    if dimension == "condition":
        return study.conditions[:5] if study.conditions else ["N/A"]
    if dimension == "country":
        return study.countries if study.countries else ["N/A"]
    if dimension == "intervention":
        return study.interventions[:5] if study.interventions else ["N/A"]
    raise ValueError(f"Unknown dimension {dimension!r}. Valid: {_ALL_DIMENSIONS}")


# ── Country → continent mapping ────────────────────────────────────────────────

_COUNTRY_TO_CONTINENT: dict[str, str] = {
    "United States": "North America", "Canada": "North America",
    "Mexico": "North America", "Brazil": "South America",
    "Argentina": "South America", "Chile": "South America",
    "Colombia": "South America", "Peru": "South America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Sweden": "Europe", "Denmark": "Europe", "Norway": "Europe",
    "Finland": "Europe", "Poland": "Europe", "Czech Republic": "Europe",
    "Hungary": "Europe", "Portugal": "Europe", "Greece": "Europe",
    "Romania": "Europe", "Ukraine": "Europe", "Russia": "Europe",
    "China": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "India": "Asia", "Taiwan": "Asia", "Singapore": "Asia",
    "Hong Kong": "Asia", "Thailand": "Asia", "Malaysia": "Asia",
    "Indonesia": "Asia", "Philippines": "Asia", "Vietnam": "Asia",
    "Israel": "Asia", "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia", "Turkey": "Asia",
    "Australia": "Oceania", "New Zealand": "Oceania",
    "South Africa": "Africa", "Egypt": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ethiopia": "Africa",
}

# ── Pure tool implementations ─────────────────────────────────────────────────

# ·· Transformation ·····························································


def sort_and_filter(
    data: dict[str, Any],
    sort_by: str = "count_desc",
    top_n: Optional[int] = None,
) -> dict[str, Any]:
    """Sort a label→value dict and optionally trim to top_n entries."""
    if sort_by == "count_desc":
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    elif sort_by == "count_asc":
        items = sorted(data.items(), key=lambda x: x[1])
    else:
        items = sorted(data.items(), key=lambda x: x[0])
    if top_n is not None:
        items = items[:top_n]
    return dict(items)


def normalize(data: dict, mode: str = "global") -> dict:
    """Convert raw counts to percentages.

    mode='global'  → each value as % of the total across all keys.
    mode='group'   → data is {group: {label: count}}; normalise within each group.
    """
    if mode == "global":
        total = sum(float(v) for v in data.values() if not isinstance(v, dict))
        if total == 0:
            return {k: 0.0 for k in data}
        return {k: round(float(v) / total * 100, 2) for k, v in data.items()}
    # group mode: data is {group: {label: count}}
    result: dict = {}
    for group, counts in data.items():
        group_total = sum(counts.values())
        if group_total == 0:
            result[group] = {k: 0.0 for k in counts}
        else:
            result[group] = {k: round(v / group_total * 100, 2) for k, v in counts.items()}
    return result


def compute_rolling_average(
    time_series_data: dict[str, float],
    window: int = 3,
) -> dict[str, float]:
    """Smooth a time series with a rolling window of *window* periods."""
    keys = sorted(time_series_data.keys())
    values = [float(time_series_data[k]) for k in keys]
    result: dict[str, float] = {}
    for i, key in enumerate(keys):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result[key] = round(sum(window_vals) / len(window_vals), 2)
    return result


def bin_continuous(
    values: list[float],
    bin_size: float,
) -> dict[str, int]:
    """Bucket continuous values into histogram bins of width *bin_size*."""
    if not values:
        return {}
    bins: Counter = Counter()
    for v in values:
        bucket_start = int(math.floor(float(v) / bin_size)) * bin_size
        label = f"{int(bucket_start)}-{int(bucket_start + bin_size - 1)}"
        bins[label] += 1
    return dict(sorted(bins.items(), key=lambda x: float(x[0].split("-")[0])))


def project_trend(
    time_series_data: dict[str, float],
    periods: int = 3,
    method: str = "linear",
) -> list[dict]:
    """Extrapolate a time series forward by *periods* steps.

    Historical points have projected=False; forecast points have projected=True.
    """
    keys = sorted(time_series_data.keys())
    values = [float(time_series_data[k]) for k in keys]
    result = [{"label": k, "value": v, "projected": False} for k, v in zip(keys, values)]

    if len(values) < 2:
        return result

    try:
        last_year = int(keys[-1])
    except ValueError:
        return result

    if method == "linear":
        n = len(values)
        x_vals = list(range(n))
        mean_x = sum(x_vals) / n
        mean_y = sum(values) / n
        num = sum((x_vals[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        den = sum((x_vals[i] - mean_x) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0
        intercept = mean_y - slope * mean_x
        for p in range(1, periods + 1):
            proj = max(0.0, intercept + slope * (n + p - 1))
            result.append({"label": str(last_year + p), "value": round(proj, 1), "projected": True})

    elif method == "exponential":
        last_val = values[-1]
        growth = (values[-1] / values[-2]) if values[-2] != 0 else 1.0
        for p in range(1, periods + 1):
            last_val = last_val * growth
            result.append({"label": str(last_year + p), "value": round(last_val, 1), "projected": True})

    return result


def merge_time_series(series_map: dict[str, dict]) -> list[dict]:
    """Align multiple {label: value} time series onto a shared x-axis.

    Missing years are zero-padded so all series cover the same span.
    """
    all_keys = sorted(set().union(*(d.keys() for d in series_map.values())))
    result = []
    for key in all_keys:
        record: dict[str, Any] = {"label": key}
        for name, data in series_map.items():
            record[name] = float(data.get(key, 0))
        result.append(record)
    return result


def count_values(values: list, top_n: Optional[int] = None) -> dict[str, int]:
    """Count occurrences in a list of categorical values → {label: count}.

    Use after extract_field_values for categorical fields (interventions,
    conditions, phases, countries) to turn the raw list into chartable counts.
    Returns the most frequent entries first; pass top_n to cap the result.
    """
    counter: Counter = Counter(
        str(v) for v in values if v is not None and str(v) != ""
    )
    items = counter.most_common(top_n) if top_n else counter.most_common()
    return dict(items)


# ·· Statistical ································································


def compute_average(yearly_counts: dict[str, int]) -> dict[str, float]:
    """Return average, total, and year count for a {year: count} dict."""
    if not yearly_counts:
        return {"average": 0.0, "total": 0, "years": 0}
    total = sum(yearly_counts.values())
    years = len(yearly_counts)
    return {"average": round(total / years, 2), "total": total, "years": years}


def compute_summary_stats(values: list[float]) -> dict:
    """Return mean, median, std-dev, min, max, p25, p75 for a list of values."""
    if not values:
        return {}
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((v - mean) ** 2 for v in sorted_vals) / n
    std_dev = math.sqrt(variance)
    mid = n // 2
    median = sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std_dev": round(std_dev, 2),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p25": sorted_vals[int(n * 0.25)],
        "p75": sorted_vals[int(n * 0.75)],
    }


def compute_growth_rate(time_series_data: dict[str, float]) -> list[dict]:
    """Compute year-over-year percentage change for a time series.

    Returns a list of {label, value, growth_rate} records.
    The first period has growth_rate=null.
    """
    keys = sorted(time_series_data.keys())
    result = []
    for i, key in enumerate(keys):
        curr = float(time_series_data[key])
        if i == 0:
            growth: Optional[float] = None
        else:
            prev = float(time_series_data[keys[i - 1]])
            growth = round((curr - prev) / prev * 100, 1) if prev != 0 else None
        result.append({"label": key, "value": curr, "growth_rate": growth})
    return result


def rank_entities(
    data: dict[str, float],
    metric: str = "count",
    top_n: Optional[int] = None,
) -> list[dict]:
    """Rank items by descending value.

    Returns [{rank, label, value}, …] sorted from highest to lowest.
    """
    items = sorted(data.items(), key=lambda x: float(x[1]), reverse=True)
    if top_n:
        items = items[:top_n]
    return [{"rank": i + 1, "label": k, "value": float(v)} for i, (k, v) in enumerate(items)]


# ·· Network / Relationship ·····················································


def build_network(nodes: list[dict], edges: list[dict]) -> dict:
    """Construct a network graph structure from node and edge lists.

    nodes: [{id, label, type?, size?}, …]
    edges: [{source, target, weight?}, …]
    """
    return {"nodes": nodes, "edges": edges}


def extract_network_from_co_occurrence(
    co_occurrence_matrix: dict[str, dict[str, int]],
    min_weight: float = 1.0,
) -> dict:
    """Convert a co-occurrence matrix into a network, dropping weak edges.

    co_occurrence_matrix: output of compute_co_occurrence.
    min_weight: edges below this value are filtered out.
    Returns {"nodes": […], "edges": […]}.
    """
    node_set: set[str] = set()
    edges = []
    seen: set[tuple] = set()
    for source, targets in co_occurrence_matrix.items():
        for target, weight in targets.items():
            if float(weight) < min_weight or source == target:
                continue
            key = (min(source, target), max(source, target))
            if key in seen:
                continue
            seen.add(key)
            node_set.add(source)
            node_set.add(target)
            edges.append({"source": source, "target": target, "weight": weight})
    nodes = [{"id": n, "label": n} for n in sorted(node_set)]
    return {"nodes": nodes, "edges": edges}


# ·· Output ·····································································


def add_annotation(text: str, target: dict) -> dict:
    """Return a formatted annotation object.

    Accumulate multiple calls, then pass them as metadata.annotations
    when calling build_visualization.
    """
    return {"text": text, "target": target}


def build_visualization(
    type: str,
    title: str,
    encoding: dict,
    data: list,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    total_records: int = 0,
    group: Optional[str] = None,
) -> VisualizationSpec:
    """Construct and return a VisualizationSpec.

    May be called multiple times per query to build a multi-chart response.
    sequence_index is auto-assigned by the loop; group is a semantic label
    the caller provides (e.g. 'trends', 'distribution').
    """
    chart_type = ChartType(type)
    return VisualizationSpec(
        chart_type=chart_type,
        title=title,
        description=description,
        encoding=encoding,
        data=[d if isinstance(d, dict) else dict(d) for d in data],
        metadata=metadata,
        total_records=total_records,
        group=group,
    )


# ── Stateful tool registry ────────────────────────────────────────────────────


class ToolRegistry:
    """Per-request tool registry with isolated study cache.

    Use as a context manager so the cache is always released on exit:

        with ToolRegistry(ct_api=ct_api, max_pages=max_pages) as registry:
            fn = registry["search_trials"]
    """

    def __init__(
        self,
        ct_api: ClinicalTrialsAPI,
        max_pages: int,
    ) -> None:
        self._ct_api = ct_api
        self._max_pages = max_pages
        self._study_cache: dict[str, list[Study]] = {}
        self._counter: int = 0
        self._tools: dict[str, Callable] = self._build_tools()

    # ── context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "ToolRegistry":
        return self

    def __exit__(self, *_: object) -> None:
        self._study_cache.clear()

    # ── dict-like access ───────────────────────────────────────────────────────

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Callable:
        return self._tools[name]

    # ── tool wiring ────────────────────────────────────────────────────────────

    def _build_tools(self) -> dict[str, Callable]:
        return {
            # Data retrieval
            "search_trials":           self._search_trials,
            "search_trials_by_nct":    self._search_trials_by_nct,
            "get_trial_details":       self._get_trial_details,
            # Aggregation
            "list_studies":            self._list_studies,
            "aggregate_by":            self._aggregate_by,
            "aggregate_by_multi":      self._aggregate_by_multi,
            "compare_groups":          self._compare_groups,
            "extract_field_values":    self._extract_field_values,
            # Transformation (pure — module-level)
            "sort_and_filter":         sort_and_filter,
            "normalize":               normalize,
            "compute_rolling_average": compute_rolling_average,
            "bin_continuous":          bin_continuous,
            "project_trend":           project_trend,
            "compute_co_occurrence":   self._compute_co_occurrence,
            "merge_time_series":       merge_time_series,
            "count_values":            count_values,
            # Statistical (pure — module-level)
            "compute_average":         compute_average,
            "compute_summary_stats":   compute_summary_stats,
            "compute_growth_rate":     compute_growth_rate,
            "rank_entities":           rank_entities,
            # Network / Relationship (pure — module-level)
            "build_network":                      build_network,
            "extract_network_from_co_occurrence": extract_network_from_co_occurrence,
            # Geographic (stateful)
            "aggregate_by_country":    self._aggregate_by_country,
            "aggregate_by_region":     self._aggregate_by_region,
            # Output (pure — module-level)
            "add_annotation":          add_annotation,
            "build_visualization":     build_visualization,
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    def _next_search_id(self) -> str:
        self._counter += 1
        return f"search_{self._counter}"

    def _require_search(self, search_id: str) -> list[Study]:
        studies = self._study_cache.get(search_id)
        if studies is None:
            raise ValueError(f"Unknown search_id {search_id!r}. Call search_trials first.")
        return studies

    # ── data retrieval ─────────────────────────────────────────────────────────

    async def _search_trials(
        self,
        condition: Optional[str] = None,
        drug: Optional[str] = None,
        sponsor: Optional[str] = None,
        phase: Optional[list[str]] = None,
        status: Optional[list[str]] = None,
        study_type: Optional[str] = None,
        intervention_type: Optional[str] = None,
        location: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        sex: Optional[str] = None,
        age_group: Optional[list[str]] = None,
        page_size: int = 50,
    ) -> dict:
        search_id = self._next_search_id()

        kwargs: dict = {"page_size": page_size}
        if condition:
            kwargs["query_cond"] = condition
        if drug:
            kwargs["query_intr"] = drug
        if sponsor:
            kwargs["query_spons"] = sponsor
        if phase:
            kwargs["filter_phase"] = phase
        if status:
            kwargs["filter_overall_status"] = status
        if study_type:
            kwargs["filter_study_type"] = [study_type]
        if intervention_type:
            kwargs["filter_intervention_type"] = intervention_type
        if location:
            kwargs["query_locn"] = location
        if start_year:
            kwargs["filter_start_year"] = start_year
        if end_year:
            kwargs["filter_end_year"] = end_year
        if sex:
            kwargs["filter_sex"] = sex
        if age_group:
            kwargs["filter_age_range"] = age_group

        params = QueryParams(**kwargs)
        study_data = await self._ct_api.search_all_pages(params, max_pages=self._max_pages)
        self._study_cache[search_id] = study_data.studies

        logger.debug(
            "search_trials | id=%s total=%d retrieved=%d",
            search_id, study_data.total_count, len(study_data.studies),
        )
        return {
            "search_id": search_id,
            "total_count": study_data.total_count,
            "retrieved": len(study_data.studies),
            "has_more": study_data.next_page_token is not None,
        }

    async def _search_trials_by_nct(self, nct_ids: list[str]) -> dict:
        search_id = self._next_search_id()
        study_data = await self._ct_api.fetch_by_nct_ids(nct_ids)
        self._study_cache[search_id] = study_data.studies
        logger.debug("search_trials_by_nct | id=%s retrieved=%d", search_id, len(study_data.studies))
        return {
            "search_id": search_id,
            "retrieved": len(study_data.studies),
            "not_found": [nid for nid in nct_ids if not any(s.nct_id == nid for s in study_data.studies)],
        }

    async def _get_trial_details(self, nct_id: str) -> dict:
        return await self._ct_api.fetch_study_details(nct_id)

    # ── aggregation ────────────────────────────────────────────────────────────

    def _aggregate_by(self, search_id: str, dimension: str) -> dict[str, int]:
        studies = self._require_search(search_id)
        attr = _DIMENSION_ATTR.get(_normalize_dimension(dimension))
        if attr is None:
            raise ValueError(f"Unknown dimension {dimension!r}. Valid: {list(_DIMENSION_ATTR)}")
        return dict(getattr(aggregate_studies(studies), attr))

    def _aggregate_by_multi(
        self, search_id: str, dimensions: list[str], top_k: int = 12
    ) -> list[dict]:
        if len(dimensions) != 2:
            raise ValueError("aggregate_by_multi requires exactly 2 dimensions")
        studies = self._require_search(search_id)
        dim1, dim2 = dimensions
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for study in studies:
            for v1 in _study_dim_values(study, dim1):
                for v2 in _study_dim_values(study, dim2):
                    matrix[v1][v2] += 1
        # Cap cardinality: keep the top_k rows and top_k cols by marginal total,
        # so high-cardinality dimensions can't return a huge matrix.
        row_totals = {r: sum(inner.values()) for r, inner in matrix.items()}
        col_totals: Counter = Counter()
        for inner in matrix.values():
            col_totals.update(inner)
        keep_rows = {r for r, _ in Counter(row_totals).most_common(top_k)}
        keep_cols = {c for c, _ in col_totals.most_common(top_k)}
        return [
            {"row": v1, "col": v2, "value": count}
            for v1, inner in matrix.items() if v1 in keep_rows
            for v2, count in inner.items() if v2 in keep_cols
        ]

    def _compare_groups(self, search_ids: dict[str, str], dimension: str) -> list[dict]:
        """Aggregate several searches by the same dimension for side-by-side comparison.

        search_ids maps a group label → search_id (e.g. {"Drug A": "search_1"}).
        Returns flattened [{label, value, group}] records, zero-padded so every
        group covers the same set of labels. Render with build_visualization(
        type="bar_chart", encoding={x:"label", y:"value", group:"group"}).
        """
        per_group: dict[str, Counter] = {}
        all_labels: set[str] = set()
        for group_label, sid in search_ids.items():
            counts: Counter = Counter()
            for study in self._require_search(sid):
                for value in _study_dim_values(study, dimension):
                    counts[value] += 1
            per_group[group_label] = counts
            all_labels |= set(counts)
        return [
            {"label": label, "value": counts.get(label, 0), "group": group_label}
            for group_label, counts in per_group.items()
            for label in sorted(all_labels)
        ]

    def _extract_field_values(self, search_id: str, field: str) -> list:
        studies = self._require_search(search_id)
        _extractors: dict[str, Any] = {
            "enrollment": lambda s: [s.enrollment] if s.enrollment is not None else [],
            "year": lambda s: (
                [int(s.start_date[:4])]
                if s.start_date and len(s.start_date) >= 4 and s.start_date[:4].isdigit()
                else []
            ),
            "conditions":    lambda s: s.conditions,
            "phases":        lambda s: s.phases,
            "interventions": lambda s: s.interventions,
            "countries":     lambda s: s.countries,
            "sponsor":       lambda s: [s.sponsor] if s.sponsor else [],
            "sponsor_class": lambda s: [s.sponsor_class] if s.sponsor_class else [],
        }
        extractor = _extractors.get(field)
        if extractor is None:
            raise ValueError(f"Unknown field {field!r}. Valid: {list(_extractors)}")
        result: list = []
        for study in studies:
            result.extend(extractor(study))
        return result

    def _list_studies(
        self,
        search_id: str,
        fields: Optional[list[str]] = None,
        top_n: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> list[dict]:
        studies = self._require_search(search_id)
        limit = top_n if top_n is not None else page_size
        if limit is not None:
            studies = studies[:limit]
        _all_fields = ["nct_id", "brief_title", "overall_status", "phases", "sponsor", "start_date", "conditions", "countries", "enrollment"]
        selected = fields if fields else _all_fields
        rows = []
        for s in studies:
            row: dict[str, Any] = {}
            if "nct_id" in selected:
                row["nct_id"] = s.nct_id
            if "brief_title" in selected:
                row["brief_title"] = s.brief_title
            if "overall_status" in selected:
                row["overall_status"] = s.overall_status
            if "phases" in selected:
                row["phases"] = ", ".join(s.phases) if s.phases else None
            if "sponsor" in selected:
                row["sponsor"] = s.sponsor
            if "start_date" in selected:
                row["start_date"] = s.start_date
            if "conditions" in selected:
                row["conditions"] = ", ".join(s.conditions[:3]) if s.conditions else None
            if "countries" in selected:
                row["countries"] = ", ".join(sorted(s.countries)[:3]) if s.countries else None
            if "enrollment" in selected:
                row["enrollment"] = s.enrollment
            rows.append(row)
        return rows

    def _compute_co_occurrence(
        self, search_id: str, field_a: str, field_b: str, top_k: int = 15
    ) -> dict:
        studies = self._require_search(search_id)
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for study in studies:
            for a in _study_dim_values(study, field_a):
                for b in _study_dim_values(study, field_b):
                    matrix[a][b] += 1
        # Cap cardinality so high-cardinality fields (interventions, conditions)
        # can't return a matrix too large to fit back into the message context.
        row_totals = {r: sum(inner.values()) for r, inner in matrix.items()}
        col_totals: Counter = Counter()
        for inner in matrix.values():
            col_totals.update(inner)
        keep_rows = {r for r, _ in Counter(row_totals).most_common(top_k)}
        keep_cols = {c for c, _ in col_totals.most_common(top_k)}
        return {
            r: {c: v for c, v in inner.items() if c in keep_cols}
            for r, inner in matrix.items() if r in keep_rows
        }

    # ── geographic ─────────────────────────────────────────────────────────────

    def _aggregate_by_country(self, search_id: str, normalize_counts: bool = False) -> list[dict]:
        studies = self._require_search(search_id)
        counter: Counter = Counter()
        for study in studies:
            for country in study.countries:
                counter[country] += 1
        total = sum(counter.values()) or 1
        result = []
        for country, count in counter.most_common():
            record: dict[str, Any] = {"country_name": country, "country_code": None, "count": count}
            if normalize_counts:
                record["percentage"] = round(count / total * 100, 2)
            result.append(record)
        return result

    def _aggregate_by_region(self, search_id: str, region_level: str = "continent") -> dict[str, int]:
        if region_level != "continent":
            raise ValueError(f"Unsupported region_level {region_level!r}. Only 'continent' is supported.")
        studies = self._require_search(search_id)
        counter: Counter = Counter()
        for study in studies:
            for country in study.countries:
                region = _COUNTRY_TO_CONTINENT.get(country, "Other")
                counter[region] += 1
        return dict(counter)


# ── OpenAI tool schemas ────────────────────────────────────────────────────────

_PHASE_ENUM = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
_STATUS_ENUM = [
    "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "COMPLETED", "SUSPENDED", "TERMINATED", "WITHDRAWN",
]
_DIMENSION_ENUM = ["phase", "status", "year", "study_type", "sponsor", "enrollment", "condition"]
_DIMENSION_ENUM_WITH_COUNTRY = _DIMENSION_ENUM + ["country", "intervention"]

TOOL_SCHEMAS: list[dict] = [
    # ── Data Retrieval ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_trials",
            "description": (
                "Search ClinicalTrials.gov for studies. "
                "Call once per distinct entity when comparing two things. "
                "Returns a search_id to reference in aggregation tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "Medical condition or disease"},
                    "drug": {"type": "string", "description": "Drug or intervention name"},
                    "sponsor": {"type": "string", "description": "Sponsor or organization name"},
                    "phase": {"type": "array", "items": {"type": "string", "enum": _PHASE_ENUM}},
                    "status": {"type": "array", "items": {"type": "string", "enum": _STATUS_ENUM}},
                    "study_type": {"type": "string", "enum": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]},
                    "intervention_type": {"type": "string", "description": "e.g. DRUG, DEVICE, BIOLOGICAL"},
                    "location": {"type": "string", "description": "Country, state, or city"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                    "sex": {"type": "string", "enum": ["female", "male", "all"]},
                    "age_group": {"type": "array", "items": {"type": "string", "enum": ["child", "adult", "older"]}},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Default 50"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trials_by_nct",
            "description": "Fetch specific trials by NCT ID. Use when the user references known trial IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nct_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of NCT IDs, e.g. ['NCT01234567', 'NCT07654321']",
                    },
                },
                "required": ["nct_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trial_details",
            "description": (
                "Fetch full detail for a single trial — outcomes, eligibility, arms, sites. "
                "Use when the question asks about a specific trial's contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "Single NCT ID, e.g. 'NCT01234567'"},
                },
                "required": ["nct_id"],
                "additionalProperties": False,
            },
        },
    },
    # ── Aggregation ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "list_studies",
            "description": (
                "Return study records as table rows. "
                "Use before build_visualization(type='table') to surface individual trials. "
                "Pass the result directly as data=."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["nct_id", "brief_title", "overall_status", "phases", "sponsor", "start_date", "conditions", "countries", "enrollment"],
                        },
                        "description": "Columns to include. Defaults to all fields when omitted.",
                    },
                    "top_n": {"type": "integer", "description": "Limit to first N studies (default: all)"},
                    "page_size": {"type": "integer", "description": "Alias for top_n"},
                },
                "required": ["search_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_by",
            "description": (
                "Group studies by a single dimension and return {label: count}. "
                "The dimension must be DIFFERENT from any filter already applied in search_trials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "dimension": {"type": "string", "enum": _DIMENSION_ENUM},
                },
                "required": ["search_id", "dimension"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_by_multi",
            "description": (
                "Cross-tabulate studies by exactly 2 dimensions. "
                "Returns [{row, col, value}] — one record per cell. "
                "Pass directly as data= to build_visualization(type='heatmap')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string", "enum": _DIMENSION_ENUM_WITH_COUNTRY},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Exactly 2 dimensions to cross-tabulate, e.g. ['phase', 'status']",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Keep only the top K rows and columns by frequency (default 12)",
                    },
                },
                "required": ["search_id", "dimensions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_groups",
            "description": (
                "Aggregate multiple searches by the same dimension for side-by-side comparison. "
                "Pass {group_label: search_id} (one search per entity, e.g. {'Drug A': 'search_1', "
                "'Drug B': 'search_2'}). Returns flattened [{label, value, group}], zero-padded so "
                "every group shares the same labels. Render with build_visualization(type='bar_chart', "
                "encoding={x:'label', y:'value', group:'group'})."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_ids": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Mapping of group label → search_id",
                    },
                    "dimension": {
                        "type": "string",
                        "enum": _DIMENSION_ENUM_WITH_COUNTRY,
                    },
                },
                "required": ["search_ids", "dimension"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_field_values",
            "description": (
                "Pull raw values of a single field across all studies in a search result. "
                "Use before bin_continuous (enrollment, year) or compute_summary_stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "field": {
                        "type": "string",
                        "enum": ["enrollment", "year", "conditions", "phases", "interventions", "countries", "sponsor", "sponsor_class"],
                    },
                },
                "required": ["search_id", "field"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_values",
            "description": (
                "Count occurrences in a list of categorical values, returning {label: count}. "
                "Use after extract_field_values on a categorical field "
                "(interventions, conditions, phases, countries) to build bar_chart data. "
                "This is the only way to chart interventions/drugs by frequency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of categorical values from extract_field_values",
                    },
                    "top_n": {"type": "integer", "description": "Keep only the N most frequent (default: all)"},
                },
                "required": ["values"],
                "additionalProperties": False,
            },
        },
    },
    # ── Transformation ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "sort_and_filter",
            "description": "Sort a {label: count} dict and optionally keep only the top N entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "object", "additionalProperties": {"type": "number"}},
                    "sort_by": {"type": "string", "enum": ["count_desc", "count_asc", "alphabetical"]},
                    "top_n": {"type": "integer", "description": "Keep only the top N entries after sorting"},
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize",
            "description": (
                "Convert raw counts to percentages. "
                "mode='global': each value as % of total. "
                "mode='group': data is {group: {label: count}}, normalised within each group."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Either {label: count} or {group: {label: count}} for group mode",
                    },
                    "mode": {"type": "string", "enum": ["global", "group"], "description": "Default global"},
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_rolling_average",
            "description": "Smooth a {label: value} time series with a rolling window of N periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_series_data": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Year/period → count mapping (e.g. from aggregate_by dimension='year')",
                    },
                    "window": {"type": "integer", "minimum": 2, "description": "Window size in periods (default 3)"},
                },
                "required": ["time_series_data"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bin_continuous",
            "description": (
                "Bucket continuous values into histogram bins. "
                "Use after extract_field_values(field='enrollment') or field='year' to build a histogram."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric values from extract_field_values",
                    },
                    "bin_size": {"type": "number", "description": "Width of each bin"},
                },
                "required": ["values", "bin_size"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_trend",
            "description": (
                "Extrapolate a time series forward by N periods. "
                "Projected points are marked with projected=true so the frontend can style them differently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_series_data": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "periods": {"type": "integer", "minimum": 1, "description": "Number of periods to forecast (default 3)"},
                    "method": {"type": "string", "enum": ["linear", "exponential"], "description": "Default linear"},
                },
                "required": ["time_series_data"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_co_occurrence",
            "description": (
                "Count how often values of field_a and field_b appear together in the same study. "
                "Returns {a_value: {b_value: count}}. "
                "Pass result to extract_network_from_co_occurrence to build a network graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "field_a": {"type": "string", "enum": _DIMENSION_ENUM_WITH_COUNTRY},
                    "field_b": {"type": "string", "enum": _DIMENSION_ENUM_WITH_COUNTRY},
                    "top_k": {
                        "type": "integer",
                        "description": "Keep only the top K rows and columns by frequency (default 15)",
                    },
                },
                "required": ["search_id", "field_a", "field_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_time_series",
            "description": (
                "Align multiple {label: value} time series onto the same x-axis, zero-padding missing years. "
                "Returns [{label, series1_name, series2_name, …}, …]. "
                "Use before building a multi-line time_series chart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_map": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "description": "Mapping of series_name → {year: count}",
                    },
                },
                "required": ["series_map"],
                "additionalProperties": False,
            },
        },
    },
    # ── Statistical ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "compute_average",
            "description": "Compute average, total, and year count from a {year: count} dict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "yearly_counts": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
                "required": ["yearly_counts"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_summary_stats",
            "description": "Return mean, median, std-dev, min, max, p25, p75 for a list of numeric values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric values (e.g. from extract_field_values field='enrollment')",
                    },
                },
                "required": ["values"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_growth_rate",
            "description": "Compute year-over-year percentage change. Returns [{label, value, growth_rate}, …].",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_series_data": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": ["time_series_data"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_entities",
            "description": "Rank items by descending value. Returns [{rank, label, value}, …].",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "object", "additionalProperties": {"type": "number"}},
                    "metric": {"type": "string", "description": "Label for the metric being ranked (default 'count')"},
                    "top_n": {"type": "integer", "description": "Limit to top N"},
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        },
    },
    # ── Network / Relationship ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "build_network",
            "description": (
                "Construct a network graph structure. "
                "Pass result as data=[{nodes:[…], edges:[…]}] to build_visualization with type='network_graph'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Node list: [{id, label, type?, size?}, …]",
                    },
                    "edges": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Edge list: [{source, target, weight?}, …]",
                    },
                },
                "required": ["nodes", "edges"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_network_from_co_occurrence",
            "description": (
                "Convert a co-occurrence matrix (output of compute_co_occurrence) into "
                "a network of nodes and edges, filtering out weak connections below min_weight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "co_occurrence_matrix": {
                        "type": "object",
                        "description": "Output of compute_co_occurrence",
                    },
                    "min_weight": {
                        "type": "number",
                        "description": "Minimum edge weight to include (default 1)",
                    },
                },
                "required": ["co_occurrence_matrix"],
                "additionalProperties": False,
            },
        },
    },
    # ── Geographic ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "aggregate_by_country",
            "description": (
                "Aggregate studies by country. "
                "Returns [{country_name, country_code, count, (percentage?)}]. "
                "Use with type='choropleth_map' in build_visualization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "normalize_counts": {
                        "type": "boolean",
                        "description": "If true, adds a percentage field to each record",
                    },
                },
                "required": ["search_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_by_region",
            "description": "Group countries into continents and sum counts. Returns {region_name: count}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_id": {"type": "string"},
                    "region_level": {
                        "type": "string",
                        "enum": ["continent"],
                        "description": "Grouping level (currently only 'continent')",
                    },
                },
                "required": ["search_id"],
                "additionalProperties": False,
            },
        },
    },
    # ── Output ─────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "add_annotation",
            "description": (
                "Return a formatted annotation object. "
                "Accumulate results from multiple calls, then pass them as "
                "metadata.annotations=[…] when calling build_visualization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Annotation callout text"},
                    "target": {
                        "type": "object",
                        "description": 'Point or region to annotate, e.g. {"label": "2020"} or {"index": 5}',
                    },
                },
                "required": ["text", "target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_visualization",
            "description": (
                "Construct and return the final visualization specification. "
                "This MUST be the last tool called."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [t.value for t in ChartType],
                        "description": (
                            "bar_chart=discrete categories | "
                            "time_series=ordered time axis | "
                            "scatter_plot=two continuous axes | "
                            "histogram=value distribution (bin_continuous) | "
                            "network_graph=nodes+edges (build_network) | "
                            "choropleth_map=geographic fill (aggregate_by_country) | "
                            "none=single answer | "
                            "table=multi-column rows | "
                            "heatmap=two categorical axes with numeric intensity (aggregate_by_multi)"
                        ),
                    },
                    "title": {"type": "string"},
                    "encoding": {
                        "type": "object",
                        "description": (
                            "Field-to-channel mapping. Examples: "
                            "bar_chart → {x:'label',y:'value'} | "
                            "time_series → {x:'label',y:'value'} | "
                            "choropleth_map → {location:'country_name',color:'count'} | "
                            "network_graph → {node_id:'id',node_label:'label',edge_source:'source',edge_target:'target',edge_weight:'weight'} | "
                            "table → {columns:['col1',…]} | "
                            "heatmap → {x:'col',y:'row',color:'value'}"
                        ),
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Data records. Shape depends on chart_type: "
                            "bar/time/scatter/histogram → [{label,value,…}] | "
                            "grouped bar (comparison) → bar_chart with [{label,value,group}] (compare_groups output) | "
                            "choropleth_map → [{country_name,country_code,count}] | "
                            "network_graph → [{nodes:[…],edges:[…]}] (single element) | "
                            "table → [{col1,col2,…}] | "
                            "heatmap → [{row,col,value}] (aggregate_by_multi output)"
                        ),
                    },
                    "description": {"type": "string", "description": "One-sentence plain-language summary"},
                    "metadata": {
                        "type": "object",
                        "description": "Optional: {filters_applied, warnings, annotations:[…], data_provenance}",
                    },
                    "total_records": {"type": "integer"},
                    "group": {
                        "type": "string",
                        "description": (
                            "Semantic label for this chart within a multi-chart response, "
                            "e.g. 'trends', 'distribution', 'comparison', 'geographic'. "
                            "Required when calling build_visualization more than once."
                        ),
                    },
                },
                "required": ["type", "title", "encoding", "data"],
                "additionalProperties": False,
            },
        },
    },
]
