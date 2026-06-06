"""Benchmark runner.

Reads queries from tests.txt (one per line), POSTs each to the q2v_agent API,
and appends the full query + response to results.jsonl.

Usage:
    python benchmark/run_eval.py [--api-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BENCHMARK_DIR = Path(__file__).parent
TESTS_FILE = BENCHMARK_DIR / "tests.txt"
RESULTS_FILE = BENCHMARK_DIR / "results.jsonl"

DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 180.0  # seconds — LLM calls can be slow


async def _run_query(client: httpx.AsyncClient, query: str, api_url: str) -> dict:
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{api_url}/api/query",
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
        response_body = resp.json()
    except Exception as exc:
        response_body = {"error": str(exc)}
    duration_ms = int((time.monotonic() - start) * 1000)
    return {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "response": response_body,
    }


async def main(api_url: str) -> None:
    if not TESTS_FILE.exists():
        print(f"tests.txt not found at {TESTS_FILE}", file=sys.stderr)
        sys.exit(1)

    queries = [q.strip() for q in TESTS_FILE.read_text(encoding="utf-8").splitlines() if q.strip()]
    if not queries:
        print("tests.txt is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(queries)} queries against {api_url}")
    print(f"Results -> {RESULTS_FILE}\n")

    async with httpx.AsyncClient() as client:
        with RESULTS_FILE.open("w", encoding="utf-8") as out:
            for i, query in enumerate(queries, 1):
                print(f"[{i}/{len(queries)}] {query[:90]}")
                result = await _run_query(client, query, api_url)
                code = result["response"].get("code", "err")
                charts = len((result["response"].get("data") or {}).get("visualizations", []))
                print(f"         code={code}  charts={charts}  {result['duration_ms']}ms")
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

    print(f"\nDone. {len(queries)} results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark queries against the q2v_agent API.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the running server")
    args = parser.parse_args()
    asyncio.run(main(args.api_url))
