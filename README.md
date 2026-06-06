# Query-to-Visualization Clinical Trials Agent

A FastAPI backend that transforms natural-language questions about clinical trials into rich, library-agnostic visualization specifications. Powered by the [ClinicalTrials.gov v2 API](https://clinicaltrials.gov/data-api/api) and Anthropic Claude.

---

## Architecture

```
POST /api/query  {query, fields?}
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1  INTERPRET   agent/interpret.py                │
│           LLM converts NL query → QueryParams (LLM)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2  RETRIEVE    agent/retrieve.py                 │
│           QueryParams → StudyData                       │
│           httpx · pagination · retry/back-off (API)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3a ANALYZE     agent/visualize.py                │
│           StudyData → AggregatedData  (pure Python)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3b DECIDE      agent/visualize.py                │
│           question + AggregatedData → chart choice (LLM)│
│           LLM reads numbers — NEVER computes them        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4  SPEC        agent/visualize.py                │
│           decision + AggregatedData → VisualizationSpec │
│           (pure, fully validated Pydantic model)        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
        {interpreted_params, data_summary, visualization_spec}
```

### Ports & Adapters

The `agent/` package depends **only** on `ports.py`. Concrete implementations live in `adapters/`.

| Interface (`ports.py`) | Adapter (`adapters/`) | Notes                       |
| ---------------------- | --------------------- | --------------------------- |
| `LLMPort`              | `OpenAILLM`           | Tool-use forced JSON output |
| `ClinicalTrialsPort`   | `ClinicalTrialsAPI`   | httpx, retry, pagination    |

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

All other settings have sensible defaults (see `.env.example`).

### Run

```bash
uv run uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

---

## Usage

```bash
curl -s -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many phase 3 cancer trials are currently recruiting?"}' \
  | python -m json.tool
```

Example response:

```json
{
	"interpreted_params": {
		"query_cond": "cancer",
		"filter_overall_status": ["RECRUITING"],
		"page_size": 50
	},
	"data_summary": {
		"total_count": 4821,
		"retrieved": 50,
		"has_more": true
	},
	"visualization_spec": {
		"chart_type": "bar",
		"title": "Phase 3 Cancer Trials Currently Recruiting",
		"x_field": "label",
		"y_field": "value",
		"series": [{ "label": "PHASE3", "value": 47.0, "group": null }],
		"axis_labels": { "x": "Phase", "y": "Count" },
		"aggregation_applied": "Grouped by phase",
		"total_records": 50
	}
}
```

The `visualization_spec` is intentionally library-agnostic. Pass it directly to Vega-Lite, Recharts, Chart.js, or any other renderer.

---

## Tests

```bash
uv run pytest
# or with verbose output:
uv run pytest -v
```

| File                        | What it covers                                    |
| --------------------------- | ------------------------------------------------- |
| `tests/test_interpret.py`   | NL → QueryParams with FakeLLM                     |
| `tests/test_aggregation.py` | Deterministic aggregation, exact counts           |
| `tests/test_visualize.py`   | Chart-type selection + VisualizationSpec assembly |
| `tests/test_api.py`         | TestClient smoke tests on POST /api/query         |

---

## Schema Reference

### QueryParams

Maps to ClinicalTrials.gov v2 API query parameters.

| Field                   | API param              | Type            | Default |
| ----------------------- | ---------------------- | --------------- | ------- |
| `query_cond`            | `query.cond`           | `str \| null`   | `null`  |
| `query_term`            | `query.term`           | `str \| null`   | `null`  |
| `query_intr`            | `query.intr`           | `str \| null`   | `null`  |
| `filter_overall_status` | `filter.overallStatus` | `str[] \| null` | `null`  |
| `fields`                | `fields`               | `str[] \| null` | `null`  |
| `page_size`             | `pageSize`             | `int` 1–1000    | `50`    |
| `sort`                  | `sort`                 | `str \| null`   | `null`  |

Valid `filter_overall_status` values: `RECRUITING`, `NOT_YET_RECRUITING`, `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `SUSPENDED`, `TERMINATED`, `WITHDRAWN`, `AVAILABLE`, `UNKNOWN` (and several rarer values).

### VisualizationSpec

Vega-Lite-inspired, fully library-agnostic.

| Field                 | Type          | Description                                           |
| --------------------- | ------------- | ----------------------------------------------------- |
| `chart_type`          | `ChartType`   | `bar` / `line` / `pie` / `scatter` / `table` / `none` |
| `title`               | `str`         | Human-readable chart title                            |
| `description`         | `str \| null` | One-sentence explanation                              |
| `x_field`             | `str \| null` | DataPoint key for x-axis (`"label"`)                  |
| `y_field`             | `str \| null` | DataPoint key for y-axis (`"value"`)                  |
| `series`              | `DataPoint[]` | Aggregated data: `{label, value, group?}`             |
| `axis_labels`         | `{x, y}`      | Human-readable axis label strings                     |
| `aggregation_applied` | `str`         | Plain description of the aggregation                  |
| `total_records`       | `int`         | Total studies in the underlying dataset               |

---

## Adding a New Visualization Type

Follow these five steps to add, say, a `heatmap` type.

### Step 1 — Add to `ChartType` (`models.py`)

```python
class ChartType(str, Enum):
    bar     = "bar"
    line    = "line"
    pie     = "pie"
    scatter = "scatter"
    table   = "table"
    none    = "none"
    heatmap = "heatmap"   # ← new
```

### Step 2 — Extend the LLM decision schema (`agent/visualize.py`)

In `_DECISION_SCHEMA`, add the type to the `chart_type` enum:

```python
"chart_type": {
    "type": "string",
    "enum": ["bar", "line", "pie", "scatter", "table", "none", "heatmap"],
},
```

### Step 3 — Document the selection rule (`agent/visualize.py`)

In the `_DECISION_SYSTEM` prompt, add a rule under "Chart-type selection rules":

```
heatmap → two categorical dimensions with a numeric value; best for
           showing density patterns (e.g. phase × sponsor class counts)
```

### Step 4 — (Optional) Add a new aggregation

If the chart needs data not in `AggregatedData`:

1. Add the field to `AggregatedData` in `models.py`:

    ```python
    by_phase_sponsor: Dict[str, int] = Field(default_factory=dict)
    ```

2. Compute it in `aggregate_studies()` in `agent/visualize.py`.

3. Add the new key to `aggregation_key` in `_DECISION_SCHEMA`:

    ```python
    "enum": [..., "by_phase_sponsor"],
    ```

### Step 5 — Write a test (`tests/test_visualize.py`)

```python
@pytest.mark.asyncio
async def test_chart_type_heatmap():
    llm = FakeLLM({
        "chart_type": "heatmap",
        "aggregation_key": "by_phase",
        "title": "Phase Heatmap",
        "x_label": "Phase",
        "y_label": "Count",
    })
    spec = await decide_and_build_spec("phase heatmap", make_aggregated(), llm)
    assert spec.chart_type == ChartType.heatmap
```

That is all. The frontend renderer is responsible for interpreting the new `chart_type` value — the backend remains unchanged beyond these five steps.

---

## Project Structure

```
q2v_agent/
├── models.py                  # Core Pydantic models (shared, no internal deps)
├── ports.py                   # Abstract port interfaces (LLMPort, ClinicalTrialsPort)
├── agent/
│   ├── interpret.py           # Stage 1: NL → QueryParams
│   ├── retrieve.py            # Stage 2: QueryParams → StudyData
│   ├── visualize.py           # Stages 3+4: Aggregate → Decide → VisualizationSpec
│   └── pipeline.py            # Orchestrates all stages
├── adapters/
│   ├── anthropic_llm.py       # LLMPort via Anthropic tool-use
│   └── clinicaltrials_api.py  # ClinicalTrialsPort via httpx
├── app/
│   ├── config.py              # pydantic-settings (.env)
│   ├── main.py                # FastAPI app + lifespan
│   └── routers/
│       └── query.py           # POST /api/query
├── tests/
│   ├── conftest.py            # Sets dummy env var before imports
│   ├── test_interpret.py
│   ├── test_aggregation.py
│   ├── test_visualize.py
│   └── test_api.py
├── pyproject.toml
└── .env.example
```

