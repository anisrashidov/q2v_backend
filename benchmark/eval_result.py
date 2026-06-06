"""Benchmark evaluator.

Reads results.jsonl produced by run_eval.py, calls an LLM to score each
query/response pair on completeness and soundness (0.0–1.0), and writes
scores to scores.jsonl with a summary printed at the end.

Usage:
    python benchmark/eval_result.py [--model gpt-4.1-mini] [--results results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv('.env')

import openai
from pydantic import BaseModel, Field

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_RESULTS_FILE = BENCHMARK_DIR / "results.jsonl"
SCORES_FILE = BENCHMARK_DIR / "scores.jsonl"

DEFAULT_MODEL = "gpt-4.1-mini"

EVAL_SYSTEM_PROMPT = """\
You are evaluating an AI assistant that converts natural-language questions about
clinical trials into library-agnostic visualization specifications (JSON).

Given a user query and a compact summary of the assistant's response, rate the
response on COMPLETENESS and SOUNDNESS from 0.0 to 1.0:

  1.0  — Fully answers the question. Appropriate chart type(s), meaningful data
          (non-empty, correct domain), title matches intent.
  0.75 — Mostly correct; minor issues (slightly wrong chart type, thin data, etc.).
  0.5  — Partially answers; one major gap (wrong aggregation, missing dimension, etc.).
  0.25 — Attempts an answer but substantially fails (empty data, wrong domain, etc.).
  0.0  — Complete failure: error, no visualization, or refused a clearly valid question.

Special case — rejection scoring:
  If the query is NOT about clinical trials or is genuinely unanswerable (too vague),
  a rejection (code=400) is CORRECT and should score 1.0.
  If the query is a valid clinical-trial question and was rejected, score 0.0.

Return a score (float, exactly 0.0–1.0) and one or two sentences of reasoning.
"""


class EvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _response_summary(response: dict) -> str:
    """Compact the API response to a token-efficient string for the evaluator."""
    code = response.get("code", "?")
    message = response.get("message", "")
    data = response.get("data")
    error = response.get("error")

    if error:
        return f"ERROR: {error}"
    if data is None:
        return f"code={code}  message={message!r}  data=null"

    vizs = data.get("visualizations", [])
    lines = [f"code={code}  charts={len(vizs)}"]
    for v in vizs[:4]:
        data_pts = len(v.get("data", []))
        lines.append(
            f"  chart_type={v.get('chart_type')}  "
            f"title={v.get('title')!r}  "
            f"data_points={data_pts}"
        )
    return "\n".join(lines)


def _score_entry(client: openai.OpenAI, entry: dict, model: str) -> dict:
    query = entry.get("query", "")
    summary = _response_summary(entry.get("response", {}))

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\nResponse summary:\n{summary}",
            },
        ],
        response_format=EvalScore,
        temperature=0.0,
    )

    parsed: Optional[EvalScore] = completion.choices[0].message.parsed
    if parsed is None:
        return {"query": query, "score": None, "reasoning": "Evaluator returned no parsable output."}
    return {"query": query, "score": round(parsed.score, 4), "reasoning": parsed.reasoning}


def main(results_file: Path, model: str) -> None:
    if not results_file.exists():
        print(f"results.jsonl not found at {results_file}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    entries = [
        json.loads(line)
        for line in results_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        print("results.jsonl is empty.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    print(f"Evaluating {len(entries)} results with {model}")
    print(f"Scores -> {SCORES_FILE}\n")

    scored: list[dict] = []
    with SCORES_FILE.open("w", encoding="utf-8") as out:
        for i, entry in enumerate(entries, 1):
            query_preview = entry.get("query", "")[:80]
            print(f"[{i}/{len(entries)}] {query_preview}")
            result = _score_entry(client, entry, model)
            score_str = f"{result['score']:.2f}" if result["score"] is not None else "N/A"
            print(f"         score={score_str}  {result['reasoning'][:100]}")
            scored.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    valid = [s["score"] for s in scored if s["score"] is not None]
    if valid:
        avg = sum(valid) / len(valid)
        low = min(valid)
        high = max(valid)
        print(f"\n{'─' * 50}")
        print(f"Evaluated : {len(valid)}/{len(scored)}")
        print(f"Average   : {avg:.3f}")
        print(f"Range     : {low:.3f} – {high:.3f}")
        print(f"{'─' * 50}")

    print(f"\nScores saved to {SCORES_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate q2v_agent benchmark results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to results.jsonl (default: benchmark/results.jsonl)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for evaluation (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    main(args.results, args.model)
