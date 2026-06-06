"""Streamlit demo for the Q2V Agent API.

Two modes:
  Gallery — browse cached benchmark results (benchmark/results.jsonl) with no API call
  Live    — submit a new query to the running backend

Run from the project root:
    streamlit run demo/app.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import date

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_PATH = Path(__file__).parent.parent / "benchmark" / "results.jsonl"
DEFAULT_API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Trials Q2V Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Q2V Agent")
    st.caption("Query-to-Visualization · Clinical Trials")
    st.divider()

    st.subheader("Backend URL")
    api_url = st.text_input("API URL (for Live tab)", value=DEFAULT_API_URL, label_visibility="collapsed")

    st.divider()
    st.subheader("Supported chart types")
    for ct, desc in {
        "bar_chart": "Categorical distributions & comparisons",
        "time_series": "Trends and rolling averages over time",
        "scatter_plot": "Two continuous axes",
        "histogram": "Continuous value distribution",
        "heatmap": "Two-category intensity grid",
        "choropleth_map": "Geographic fill map",
        "network_graph": "Co-occurrence / relationship networks",
        "table": "Multi-column tabular listing",
        "none": "Single scalar answer",
    }.items():
        st.markdown(f"**`{ct}`** — {desc}")

    st.divider()
    st.caption(
        "Data: [ClinicalTrials.gov v2](https://clinicaltrials.gov/data-api/api)  \n"
        "LLM: OpenAI via the Q2V Agent backend"
    )

# ── Load cached results ────────────────────────────────────────────────────────


@st.cache_data
def load_cached_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    entries = []
    with RESULTS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = obj.get("response", {})
            # Keep only clean 200 responses with at least one visualization
            if (
                isinstance(resp, dict)
                and resp.get("code") == 200
                and resp.get("data")
                and resp["data"].get("visualizations")
            ):
                entries.append(obj)
    return entries


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _df(spec: dict) -> pd.DataFrame:
    return pd.DataFrame(spec.get("data", []))


def _col(enc: dict, role: str, fallback: str) -> str:
    """Return the field name for an encoding role, with a fallback."""
    val = enc.get(role, fallback)
    return val if isinstance(val, str) else fallback


# Presentation labels for ClinicalTrials.gov controlled-vocabulary codes.
# Only exact matches are rewritten — free-text labels (drug names, conditions,
# countries, years) pass through untouched. "NA" (the API's value for a
# non-phased study) and "N/A" (the backend's placeholder for a trial that
# reported no phase at all) both collapse into one readable "Not Applicable".
_LABEL_OVERRIDES: dict[str, str] = {
    # Phase
    "NA": "Not Applicable",
    "N/A": "Not Applicable",
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    # Overall status
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not Yet Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, Not Recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by Invitation",
    "COMPLETED": "Completed",
    "SUSPENDED": "Suspended",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "AVAILABLE": "Available",
    "NO_LONGER_AVAILABLE": "No Longer Available",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily Not Available",
    "APPROVED_FOR_MARKETING": "Approved for Marketing",
    "WITHHELD": "Withheld",
    "UNKNOWN": "Unknown",
    # Study type
    "INTERVENTIONAL": "Interventional",
    "OBSERVATIONAL": "Observational",
    "EXPANDED_ACCESS": "Expanded Access",
    # Sponsor / funder class ("NIH" is already presentable)
    "OTHER_GOV": "Other Government",
    "INDIV": "Individual",
    "INDUSTRY": "Industry",
    "NETWORK": "Network",
    "FED": "Federal",
    "OTHER": "Other",
}


def _humanize_labels(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Rewrite category codes to readable labels and merge rows that now share
    a label (e.g. NA + N/A → 'Not Applicable', summing their values).

    sort=False preserves the backend's original ordering. Columns other than
    the label/value/group keys are dropped, which is fine for categorical bars.
    """
    if label_col not in df.columns or value_col not in df.columns:
        return df
    df = df.copy()
    df[label_col] = df[label_col].map(
        lambda v: _LABEL_OVERRIDES.get(v, v) if isinstance(v, str) else v
    )
    keys = [label_col]
    if group_col and group_col in df.columns:
        keys.append(group_col)
    return df.groupby(keys, as_index=False, sort=False)[value_col].sum()


def render_bar_chart(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    x = _col(enc, "x", "label")
    y_raw = enc.get("y", "value")
    # y can be a list (e.g. ["value", "growth_rate"]) — pick the first present column
    if isinstance(y_raw, list):
        y = next((c for c in y_raw if c in df.columns), y_raw[0])
    else:
        y = y_raw
    group = enc.get("group")

    # If y column is missing, fall back to any numeric column
    if y not in df.columns:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        y = numeric_cols[0] if numeric_cols else df.columns[-1]

    # Rewrite category codes to readable labels and merge NA + N/A into one bar
    df = _humanize_labels(df, x, y, group if (group and group in df.columns) else None)

    if group and group in df.columns:
        fig = px.bar(
            df, x=x, y=y, color=group, barmode="group",
            title=spec["title"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.bar(
            df, x=x, y=y, title=spec["title"],
            color=y, color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(xaxis_tickangle=-30, margin=dict(t=50, b=80))
    st.plotly_chart(fig, use_container_width=True)


def render_time_series(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    x = _col(enc, "x", "label")
    y_raw = enc.get("y", "value")

    # y can be a list → plot all listed columns as separate series
    if isinstance(y_raw, list):
        y_cols = [c for c in y_raw if c in df.columns]
    else:
        y_cols = [y_raw] if y_raw in df.columns else []

    if not y_cols:
        numeric = df.select_dtypes("number").columns.tolist()
        y_cols = numeric[:1] if numeric else []

    if not y_cols:
        st.info("No numeric column found for time series.")
        return

    # Handle projected flag: dashed line for projected points
    has_projected = "projected" in df.columns
    if has_projected and len(y_cols) == 1:
        y = y_cols[0]
        actual = df[df["projected"] == False].copy()   # noqa: E712
        proj   = df[df["projected"] == True].copy()    # noqa: E712
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual[x], y=actual[y],
            mode="lines+markers", name="Historical",
            line=dict(color="#2563eb"), marker=dict(color="#2563eb"),
        ))
        if not proj.empty:
            fig.add_trace(go.Scatter(
                x=proj[x], y=proj[y],
                mode="lines+markers", name="Projected",
                line=dict(color="#f59e0b", dash="dash"),
                marker=dict(color="#f59e0b"),
            ))
        fig.update_layout(title=spec["title"], margin=dict(t=50))
    elif len(y_cols) > 1:
        # Multiple y series — melt to long format
        df_melt = df.melt(id_vars=[x], value_vars=y_cols, var_name="series", value_name="_val")
        fig = px.line(df_melt, x=x, y="_val", color="series", title=spec["title"], markers=True)
    else:
        y = y_cols[0]
        fig = px.line(df, x=x, y=y, title=spec["title"], markers=True, line_shape="spline")
        fig.update_traces(line_color="#2563eb", marker_color="#2563eb")

    fig.update_layout(margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    x = _col(enc, "x", "label")
    y = _col(enc, "y", "value")

    fig = px.scatter(
        df, x=x, y=y, title=spec["title"],
        trendline="ols" if len(df) > 3 else None,
        hover_name=x if x in df.columns else None,
    )
    fig.update_traces(marker=dict(size=8, color="#2563eb", opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)


def render_histogram(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    x = _col(enc, "x", "label")
    y = _col(enc, "y", "count")
    if y not in df.columns:
        numeric = df.select_dtypes("number").columns.tolist()
        y = numeric[0] if numeric else df.columns[-1]

    df = _humanize_labels(df, x, y)

    fig = px.bar(
        df, x=x, y=y, title=spec["title"],
        labels={x: x.replace("_", " ").title(), y: "Count"},
        color=y, color_continuous_scale="Purples",
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    x_col = _col(enc, "x", "col")
    y_col = _col(enc, "y", "row")
    color_col = _col(enc, "color", "value")

    try:
        pivot = df.pivot(index=y_col, columns=x_col, values=color_col).fillna(0)
        fig = px.imshow(
            pivot, title=spec["title"],
            color_continuous_scale="Blues",
            labels={"color": color_col.replace("_", " ").title()},
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)


def render_choropleth(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data returned for this map.")
        return

    enc = spec.get("encoding", {})
    loc_col  = _col(enc, "location", "country_name")
    color_col = _col(enc, "color", "count")

    # Normalize known country-name quirks for Plotly's locationmode="country names"
    name_fixes = {
        "Turkey (Türkiye)": "Turkey",
        "Czechia": "Czech Republic",
    }
    if loc_col in df.columns:
        df[loc_col] = df[loc_col].replace(name_fixes)

    use_iso = "country_code" in df.columns and df["country_code"].notna().any()
    if use_iso:
        fig = px.choropleth(
            df, locations="country_code", color=color_col,
            hover_name=loc_col, title=spec["title"],
            color_continuous_scale="Blues", projection="natural earth",
        )
    else:
        fig = px.choropleth(
            df, locations=loc_col, locationmode="country names",
            color=color_col, hover_name=loc_col, title=spec["title"],
            color_continuous_scale="Blues", projection="natural earth",
        )

    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_network_graph(spec: dict) -> None:
    data = spec.get("data", [])
    if not data:
        st.info("No network data returned.")
        return

    record = data[0] if isinstance(data, list) else data
    nodes: list[dict] = record.get("nodes", [])
    edges: list[dict] = record.get("edges", [])

    if not nodes:
        st.info("No nodes in the network.")
        return

    G = nx.Graph()
    for node in nodes:
        G.add_node(node["id"], label=node.get("label", node["id"]))
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

    k = max(1.5 / (len(nodes) ** 0.5 + 1e-6), 0.3)
    pos = nx.spring_layout(G, seed=42, k=k)

    edge_traces = []
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
    for u, v, data_e in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data_e.get("weight", 1)
        width = 1 + 4 * (w / max_w)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color="rgba(100,130,200,0.4)"),
            hoverinfo="none", showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]
    degrees = [d for _, d in G.degree()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=[10 + d * 3 for d in degrees],
            color=degrees,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Edges", thickness=12, len=0.6),
            line=dict(width=1, color="white"),
        ),
        hovertext=[f"<b>{lbl}</b><br>Connections: {d}" for lbl, d in zip(node_labels, degrees)],
        hoverinfo="text",
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=spec["title"],
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            margin=dict(t=60, b=20, l=20, r=20),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_table(spec: dict) -> None:
    df = _df(spec)
    if df.empty:
        st.info("No data to display.")
        return

    enc = spec.get("encoding", {})
    cols = enc.get("columns")
    if cols and isinstance(cols, list):
        present = [c for c in cols if c in df.columns]
        if present:
            df = df[present]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_none(spec: dict) -> None:
    desc = spec.get("description") or ""
    # Try to pull the first number (possibly with %) from the description
    nums = re.findall(r"[\d,]+(?:\.\d+)?%?", desc)
    if nums:
        st.metric(label=spec["title"], value=nums[0])
        rest = desc.replace(nums[0], "", 1).strip(" .,")
        if rest:
            st.caption(rest)
    else:
        st.metric(label=spec["title"], value=desc or "—")


RENDERERS: dict = {
    "bar_chart":      render_bar_chart,
    "time_series":    render_time_series,
    "scatter_plot":   render_scatter_plot,
    "histogram":      render_histogram,
    "heatmap":        render_heatmap,
    "choropleth_map": render_choropleth,
    "network_graph":  render_network_graph,
    "table":          render_table,
    "none":           render_none,
}

CHART_ICONS: dict = {
    "bar_chart":      "📊",
    "time_series":    "📈",
    "scatter_plot":   "⚬",
    "histogram":      "▦",
    "heatmap":        "🟦",
    "choropleth_map": "🗺️",
    "network_graph":  "🕸️",
    "table":          "🗂️",
    "none":           "🔢",
}


def render_specs(visualizations: list[dict]) -> None:
    """Render all VisualizationSpec objects in sequence_index order."""
    for spec in sorted(visualizations, key=lambda s: s.get("sequence_index", 0)):
        chart_type = spec.get("chart_type", "none")
        icon = CHART_ICONS.get(chart_type, "📌")

        with st.container(border=True):
            if spec.get("description"):
                st.caption(spec["description"])

            renderer = RENDERERS.get(chart_type)
            if renderer:
                renderer(spec)
            else:
                st.warning(f"Unknown chart type: `{chart_type}`")
                st.json(spec)

            meta = st.columns(4)
            with meta[0]:
                st.caption(f"{icon} `{chart_type}`")
            with meta[1]:
                if spec.get("total_records"):
                    st.caption(f"Records: {spec['total_records']:,}")
            with meta[2]:
                if spec.get("group"):
                    st.caption(f"Group: {spec['group']}")
            with meta[3]:
                st.caption(f"Index: {spec.get('sequence_index', 0)}")


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("Clinical Trials Query-to-Visualization")
st.markdown(
    "Ask a natural-language question about clinical trials. "
    "The AI agent retrieves real data from **ClinicalTrials.gov** "
    "and returns the best visualization for your question."
)

tab_gallery, tab_live = st.tabs(["Gallery (cached results)", "Live Query"])

# ── Tab 1: Gallery ────────────────────────────────────────────────────────────

with tab_gallery:
    st.markdown(
        "Browse pre-computed results from the benchmark run. "
        "No backend required — all data is loaded from `benchmark/results.jsonl`."
    )

    cached = load_cached_results()

    if not cached:
        st.warning(
            f"No cached results found at `{RESULTS_PATH}`. "
            "Run the benchmark first: `python benchmark/run_eval.py`"
        )
    else:
        # Build display options
        chart_type_of = {}
        labels = []
        for entry in cached:
            vizs = entry["response"]["data"]["visualizations"]
            types = ", ".join(
                f"{CHART_ICONS.get(v['chart_type'], '')} {v['chart_type']}"
                for v in vizs
            )
            label = f"[{types}]  {entry['query']}"
            labels.append(label)
            chart_type_of[label] = types

        # Filter by chart type
        all_types = sorted({
            v["chart_type"]
            for e in cached
            for v in e["response"]["data"]["visualizations"]
        })
        filter_col, _ = st.columns([2, 3])
        with filter_col:
            selected_type = st.selectbox(
                "Filter by chart type",
                ["All"] + all_types,
                format_func=lambda t: f"{CHART_ICONS.get(t, '')} {t}" if t != "All" else "All chart types",
            )

        filtered = [
            (label, entry)
            for label, entry in zip(labels, cached)
            if selected_type == "All"
            or any(
                v["chart_type"] == selected_type
                for v in entry["response"]["data"]["visualizations"]
            )
        ]

        if not filtered:
            st.info(f"No cached results for chart type `{selected_type}`.")
        else:
            filtered_labels = [lbl for lbl, _ in filtered]
            chosen_label = st.selectbox("Select a query", filtered_labels)
            chosen_entry = next(e for lbl, e in filtered if lbl == chosen_label)

            dur = chosen_entry.get("duration_ms", 0)
            st.caption(f"Cached response · {dur / 1000:.1f}s agent runtime")

            vizs = chosen_entry["response"]["data"]["visualizations"]
            render_specs(vizs)

            with st.expander("Raw response JSON"):
                st.json(chosen_entry["response"])

# ── Tab 2: Live Query ─────────────────────────────────────────────────────────

with tab_live:
    st.markdown(
        "Submit a new query to the running backend. "
        "Start the server with `python -m uvicorn app.main:app --reload` before querying."
    )

    # Pre-fill from Gallery tab is not wired, but provide example picker
    cached_queries = [e["query"] for e in (load_cached_results() or [])]
    example_opts = ["— type your own query —"] + cached_queries
    pick = st.selectbox("Start from a benchmark query (optional)", example_opts, key="live_pick")
    prefill = "" if pick == example_opts[0] else pick

    query = st.text_area(
        "Your query",
        value=prefill,
        height=80,
        placeholder="e.g. How are diabetes trials distributed across phases?",
        key="live_query",
    )

    use_tp = st.checkbox("Restrict to a time period", key="live_tp")
    time_period: list[str] | None = None
    if use_tp:
        tp1, tp2 = st.columns(2)
        with tp1:
            start = st.date_input("Start date", value=date(2015, 1, 1), key="tp_start")
        with tp2:
            end = st.date_input("End date", value=date(2023, 12, 31), key="tp_end")
        time_period = [str(start), str(end)]

    run_btn = st.button("Run Query ▶", type="primary", use_container_width=True)

    if run_btn:
        if not query.strip():
            st.warning("Please enter a query first.")
            st.stop()

        payload: dict = {"query": query.strip()}
        if time_period:
            payload["time_period"] = time_period

        with st.spinner("Running the agent… this may take 10–30 seconds."):
            try:
                resp = requests.post(
                    f"{api_url.rstrip('/')}/api/query",
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                result: dict = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the API at **{api_url}**. "
                    "Start the backend with: `python -m uvicorn app.main:app --reload`"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out after 3 minutes.")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        code    = result.get("code", 200)
        message = result.get("message", "")

        if code == 400:
            st.warning(f"Query rejected (400): {message}")
            st.stop()
        elif code == 401:
            st.error("Authentication error (401): invalid or missing OpenAI API key.")
            st.stop()
        elif code == 429:
            st.error(f"Rate limit (429): {message}")
            st.stop()
        elif code != 200:
            st.error(f"Error {code}: {message}")
            st.stop()

        data = result.get("data") or {}
        vizs: list[dict] = data.get("visualizations", [])

        if not vizs:
            st.info("The agent returned no visualizations for this query.")
        else:
            n = len(vizs)
            st.success(f"Got {n} visualization{'s' if n > 1 else ''}.")
            render_specs(vizs)

        with st.expander("Raw API response (JSON)"):
            st.json(result)
