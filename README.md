# Query-to-Visualization Clinical Trials Agent

A FastAPI backend that accepts a natural-language question about clinical trials, autonomously retrieves and analyses data from the [ClinicalTrials.gov v2 API](https://clinicaltrials.gov/data-api/api), and returns a library-agnostic visualization specification ready for any frontend renderer.

---

## How It Works

```
POST /api/query  {"query": "...", "time_period": [...]}
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  agent/loop.py  — agentic tool-calling loop             │
│                                                         │
│  1. Builds a messages list with the system prompt and   │
│     the user question.                                  │
│  2. Calls the OpenAI Chat Completions API with a full   │
│     catalogue of 24 tools.                              │
│  3. Executes every tool call returned by the model,     │
│     feeding results back into the conversation.         │
│  4. Repeats until the model calls build_visualization   │
│     and then stops (finish_reason = "stop").            │
│  5. Returns all accumulated VisualizationSpec objects.  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ToolRegistry — 24 tools in four categories             │
│                                                         │
│  Data retrieval   search_trials, search_trials_by_nct,  │
│                   get_trial_details                     │
│  Aggregation      aggregate_by,                         │
│                   compare_groups, extract_field_values, │
│                   aggregate_by_country,                 │
│                   aggregate_by_region,                  │
│                   compute_co_occurrence                 │
│  Transformation   sort_and_filter, normalize,           │
│  & statistics     compute_rolling_average, bin_continuous│
│                   project_trend, merge_time_series,     │
│                   compute_average, compute_summary_stats│
│                   compute_growth_rate, rank_entities    │
│  Network          build_network,                        │
│                   extract_network_from_co_occurrence    │
│  Annotation       add_annotation                        │
│  Output           build_visualization                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  adapters/clinicaltrials.py — ClinicalTrials.gov v2     │
│  aiohttp · pagination · exponential back-off retry      │
└─────────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# For development / testing
pip install -r requirements-dev.txt
```

### Configure

```bash
cp .env.example .env
# Open .env and fill in at minimum:
#   OPENAI_API_KEY=sk-...
```

| Variable         | Default                             | Description                              |
| ---------------- | ----------------------------------- | ---------------------------------------- |
| `OPENAI_API_KEY` | —                                   | **Required.** OpenAI API key             |
| `OPENAI_MODEL`   | `gpt-4.1`                           | Model used for all tool-calling          |
| `CT_BASE_URL`    | `https://clinicaltrials.gov/api/v2` | ClinicalTrials.gov v2 base URL           |
| `CT_MAX_PAGES`   | `5`                                 | Max pagination depth per search          |
| `CT_TIMEOUT`     | `30.0`                              | HTTP timeout in seconds                  |
| `CT_MAX_RETRIES` | `3`                                 | Retries on transient / rate-limit errors |
| `DEBUG`          | `false`                             | Enable FastAPI debug mode                |

### Run

```bash
python -m uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

---

## API Reference

### `POST /api/query`

#### Request body

```json
{
	"query": "How many phase 3 cancer trials are currently recruiting?",
	"time_period": ["2018-01-01", "2023-12-31"]
}
```

| Field         | Type                   | Required | Description                                                                                                                                           |
| ------------- | ---------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query`       | `string` (min 3 chars) | Yes      | Natural-language question about clinical trials                                                                                                       |
| `time_period` | `[date, date]`         | No       | Inclusive `[start, end]` date range to restrict trial start dates. Acts as a default — if the query itself mentions a date range, that takes priority |

#### Response envelope

Every response uses the same `BaseResponse` wrapper. The HTTP status is **always 200** for client-facing errors; `500` is reserved for unexpected server failures.

```json
{
  "code": 200,
  "message": "OK",
  "data": { ... }
}
```

| `code` | Meaning                              | `data`                                      |
| ------ | ------------------------------------ | ------------------------------------------- |
| `200`  | Success                              | `PipelineResult` object                     |
| `400`  | Bad or insufficient query            | `null` — `message` explains what is missing |
| `401`  | Invalid OpenAI API key               | `null`                                      |
| `429`  | OpenAI rate limit or budget exceeded | `null`                                      |

HTTP `500` (no envelope) for unexpected server errors.

#### `PipelineResult` schema

```json
{
	"visualizations": [
		{
			"chart_type": "bar_chart",
			"title": "Phase 3 Cancer Trials by Status",
			"description": "Distribution of recruiting statuses across 847 phase 3 cancer trials",
			"encoding": { "x": "label", "y": "value" },
			"data": [
				{ "label": "RECRUITING", "value": 412 },
				{ "label": "COMPLETED", "value": 305 }
			],
			"metadata": null,
			"total_records": 847,
			"sequence_index": 0,
			"group": null
		}
	]
}
```

| Field            | Type             | Description                                                                 |
| ---------------- | ---------------- | --------------------------------------------------------------------------- |
| `chart_type`     | `ChartType`      | One of the values below                                                     |
| `title`          | `string`         | Human-readable chart title                                                  |
| `description`    | `string \| null` | One-sentence plain-language summary                                         |
| `encoding`       | `object`         | Field-to-channel mapping (see table below)                                  |
| `data`           | `object[]`       | Data records shaped to match `chart_type`                                   |
| `metadata`       | `object \| null` | Optional: filters applied, warnings, annotations                            |
| `total_records`  | `int`            | Total studies in the underlying dataset                                     |
| `sequence_index` | `int`            | Position within a multi-chart response (0-based)                            |
| `group`          | `string \| null` | Semantic label for multi-chart grouping (e.g. `"trends"`, `"distribution"`) |

#### `ChartType` values and expected encoding / data

| `chart_type`     | `encoding`                                                                                                  | `data` record shape                                           |
| ---------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `bar_chart`      | `{x: "label", y: "value"}`                                                                                  | `{label, value}`                                              |
| `time_series`    | `{x: "label", y: "value"}`                                                                                  | `{label, value}`                                              |
| `scatter_plot`   | `{x: "label", y: "value"}`                                                                                  | `{label, value}`                                              |
| `histogram`      | `{x: "label", y: "count"}`                                                                                  | `{label, count}`                                              |
| `network_graph`  | `{node_id: "id", node_label: "label", edge_source: "source", edge_target: "target", edge_weight: "weight"}` | `{nodes: [...], edges: [...]}` (single element)               |
| `choropleth_map` | `{location: "country_name", color: "count"}`                                                                | `{country_name, country_code, count}`                         |
| `none`           | `{}`                                                                                                        | Single numeric answer — use `description` to convey the value |
| `table`          | `{columns: ["col1", "col2"]}`                                                                               | `{col1, col2, …}` (one row per record)                        |

The `encoding` object is intentionally Vega-Lite-inspired but not tied to it. Pass it directly to Vega-Lite, Recharts, Chart.js, D3, or any other renderer.

#### Error response example

```json
{
	"code": 400,
	"message": "Your query does not contain enough information to search clinical trials. Please include at least one filter such as a condition, drug name, sponsor, phase, status, or location.",
	"data": null
}
```

---

## Design Decisions and Tradeoffs

### Agentic tool-calling loop instead of a fixed pipeline

Earlier iterations used a fixed multi-stage pipeline (interpret → retrieve → aggregate → visualise). This was replaced with a single agentic loop where the model drives all decisions via tool calls. The benefit is flexibility: the model can call `search_trials` multiple times for comparisons, chain aggregation and transformation tools arbitrarily, and decide independently whether a bar chart or choropleth map best answers the question. The tradeoff is non-determinism — the same query may take different paths on different runs, making debugging harder.

### All search parameters extracted from natural language

The API accepts only `query` (plus an optional `time_period`). There are no structured fields for condition, phase, status, etc. The model extracts these from the query text. This simplifies the API surface and lets callers express nuanced constraints naturally ("phase 2 or 3 trials recruiting women with HER2-positive breast cancer in Europe since 2019"). The tradeoff is that the model may occasionally misread ambiguous phrasing.

### `time_period` as an explicit structured field

Date ranges are the one exception to natural-language-only input because they are frequently system-generated (e.g. a frontend date picker) and must be reliably honoured. When provided, the date range is injected into the user message as a default that the model uses unless the query text mentions a different range.

### `ToolRegistry` as a context manager

Each request gets its own `ToolRegistry` instance with an isolated `study_cache` (a `dict[search_id → List[Study]]`). The model receives opaque `search_id` handles rather than raw study objects, which keeps tool call payloads small. The context manager guarantees the cache is cleared on exit — including on exceptions — so study data never leaks across requests.

### Library-agnostic `VisualizationSpec`

The response schema deliberately avoids any rendering-library concepts. `encoding` is a generic field-to-channel mapping; `data` is a list of plain dicts. This means the same backend can serve a Vega-Lite renderer, a Recharts dashboard, and a table renderer without any adapter layer.

### `BaseResponse` envelope with HTTP 200 for client errors

Client-visible errors (bad query, wrong API key, rate limit) return HTTP 200 with a structured `code` / `message` body. This prevents frontend error-handling logic from diverging across HTTP status codes and makes it easy to display a human-readable message directly from `message`. True server errors still surface as HTTP 500 without the envelope so they are not silently swallowed.

### Multi-chart support with economical defaults

`build_visualization` can be called multiple times in one request, producing an ordered list of `VisualizationSpec` objects. The system prompt instructs the model to default to a single chart and only produce multiple when the question explicitly requests it ("show me both the trend and the breakdown"). `sequence_index` is assigned by the loop (not the model) to guarantee stable ordering.

---

## Limitations and What Would Be Improved with More Time

### Data coverage

The backend only queries ClinicalTrials.gov. Other major registries (EU Clinical Trials Register, WHO ICTRP, ISRCTN) are not integrated. A registry-agnostic adapter layer and a query fan-out strategy would significantly improve completeness.

### In-memory study cache

`study_cache` lives in the `ToolRegistry` object for the lifetime of one request and is then discarded. For large paginated result sets this means the data is re-fetched on every request. A short-lived per-request cache (e.g. keyed by a hash of the query parameters) backed by Redis would reduce API load and latency on repeated or similar queries.

### Year-level date filtering

`search_trials` accepts `start_year` and `end_year` as integers because the ClinicalTrials.gov API exposes year-level filters. Full ISO-date filtering (e.g. "trials that started after 2022-06-01") is not supported at the retrieval layer and would require post-fetch filtering in Python, which reduces result set accuracy when `max_pages` is low.

### No streaming

The entire agentic loop completes before any bytes are sent to the client. For complex multi-tool queries (5–10 tool calls, large paginated result sets) this can mean 10–30 seconds of silence. Server-Sent Events or WebSocket streaming of intermediate tool results would improve perceived responsiveness.

### No request authentication or per-user rate limiting

The API is open. In production it would need an auth layer (API keys or JWT) and per-key rate limiting to prevent abuse and to attribute OpenAI costs to individual callers.

### Continent-only geographic grouping

`aggregate_by_region` only supports `region_level="continent"`. Finer-grained groupings (sub-region, WHO region, income group) would require an enriched country-to-region mapping and additional enum values in the tool schema.

### Determinism and testability

Because the LLM path is non-deterministic, integration tests mock `run_pipeline` rather than running the real loop. A replay/record mechanism (capturing OpenAI responses and replaying them deterministically) would allow true end-to-end regression tests without live API calls.

---

## Project Structure

```
q2v_agent/
├── agent/
│   ├── aggregate.py       # Deterministic study aggregation (counts by phase, status, etc.)
│   ├── loop.py            # Agentic tool-calling loop (run_agent)
│   ├── pipeline.py        # Thin orchestrator: run_agent → PipelineResult
│   ├── prompts.py         # System prompt for the agentic loop
│   ├── tools.py           # ToolRegistry class + 24 tool implementations + TOOL_SCHEMAS
│   └── types.py           # Pydantic models: Study, VisualizationSpec, PipelineResult, etc.
├── adapters/
│   └── clinicaltrials.py  # ClinicalTrials.gov v2 HTTP adapter (aiohttp, retry, pagination)
├── app/
│   ├── config.py          # pydantic-settings — reads .env
│   ├── main.py            # FastAPI app + lifespan (OpenAI client, CT API client)
│   ├── dependencies.py    # FastAPI Depends providers
│   └── routers/
│       ├── query.py       # POST /api/query
│       ├── meta.py        # GET /health etc.
│       └── schemas.py     # QueryRequest, BaseResponse
├── tests/
│   ├── conftest.py
│   └── test_api.py        # TestClient smoke tests
├── .env.example
├── requirements.txt
└── requirements-dev.txt
```

