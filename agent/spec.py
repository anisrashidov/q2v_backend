"""Stage 4 — SPEC.

Pure function: converts the LLM decision dict and pre-computed AggregatedData
into a fully-validated VisualizationSpec.

No LLM, no I/O — deterministic and trivially unit-testable.
"""
from __future__ import annotations

from agent.types import AggregatedData, ChartType, DataPoint, VisualizationSpec


def build_spec(decision: dict, aggregated: AggregatedData) -> VisualizationSpec:
    """Build a :class:`VisualizationSpec` from an LLM decision + aggregated data.

    Parameters
    ----------
    decision:
        Dict produced by :func:`agent.visualize.decide_visualization`.
        Required keys: ``chart_type``, ``aggregation_key``, ``title``,
        ``x_label``, ``y_label``.  Optional: ``description``.
    aggregated:
        Pre-computed aggregations from :func:`agent.aggregate.aggregate_studies`.

    Returns
    -------
    VisualizationSpec
        Fully validated spec ready for serialisation.
    """
    chart_type = ChartType(decision["chart_type"])
    agg_key: str = decision.get("aggregation_key", "by_status")

    # Retrieve the aggregated dict; fall back gracefully to empty
    agg_data: dict = getattr(aggregated, agg_key, None) or {}

    series = [
        DataPoint(label=str(label), value=float(count))
        for label, count in agg_data.items()
    ]

    x_label = decision.get("x_label") or agg_key.replace("_", " ").title()
    y_label = decision.get("y_label") or "Count"

    return VisualizationSpec(
        chart_type=chart_type,
        title=decision.get("title", "Clinical Trial Overview"),
        description=decision.get("description"),
        x_field="label",
        y_field="value",
        series=series,
        axis_labels={"x": x_label, "y": y_label},
        aggregation_applied=f"Grouped by {agg_key.replace('_', ' ')}",
        total_records=aggregated.total_studies,
    )
