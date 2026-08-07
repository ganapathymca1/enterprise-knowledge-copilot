"""Offline evaluation harness.

Runs the *whole* pipeline (screening, rewriting, routing, retrieval, tools,
generation, verification) against a hand-labelled set and prints the metrics
quoted in docs/ACCURACY_AND_LIMITATIONS.md. It calls the orchestrator directly
rather than over HTTP so a run needs no server and no network.

    python eval/run_eval.py                     # offline, deterministic, free
    python eval/run_eval.py --provider ollama   # measure a real LLM
    python eval/run_eval.py --retrieval-only    # retrieval metrics only, fast
    python eval/run_eval.py --json out.json     # machine-readable results

Metrics
-------
retrieval    hit@1 / hit@5 / MRR at *document* level (a chunk is a hit if its
             document is one of the labelled sources)
routing      share of turns whose answer type matches the label; tool accuracy
abstention   share of unanswerable questions that were declined or explicitly
             flagged with a caution
fidelity     share of `must_include` facts that survived into the answer
grounding    mean post-generation grounding score
latency      p50 / p95 wall clock per turn
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.models import ChatRequest  # noqa: E402
from backend.app.rag import KnowledgeIndex  # noqa: E402
from backend.app.services import ConversationStore, FeedbackStore  # noqa: E402
from backend.app.services.orchestrator import build_orchestrator  # noqa: E402
from backend.app.tools import ToolRegistry, load_records  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.json"


def _score_retrieval(row: dict, retrieved: list[str], expected: set[str]) -> None:
    """Document-level hit@1 / hit@k / reciprocal rank for one case."""
    row["retrieved_docs"] = retrieved
    row["hit@1"] = bool(retrieved) and retrieved[0] in expected
    row["hit@k"] = any(doc in expected for doc in retrieved)
    rank = next((i + 1 for i, doc in enumerate(retrieved) if doc in expected), 0)
    row["rr"] = 1.0 / rank if rank else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def evaluate(provider: str | None, retrieval_only: bool, top_k: int) -> dict:
    settings = get_settings()
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    settings.top_k = top_k

    index = KnowledgeIndex.from_directory(
        settings.knowledge_base_dir,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))["cases"]
    chunk_text = {chunk.chunk_id: chunk.text for chunk in index.chunks}

    orchestrator = None
    if not retrieval_only:
        tools = ToolRegistry(load_records(settings.hr_records_dir))
        orchestrator = await build_orchestrator(
            settings,
            index,
            tools,
            ConversationStore(ttl_minutes=settings.session_ttl_minutes),
            FeedbackStore(settings.state_dir / "eval.sqlite3"),
        )

    rows: list[dict] = []
    for case in cases:
        expected_docs = set(case.get("expected_docs", []))
        row: dict = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_docs": sorted(expected_docs),
        }

        # ---- retrieval ----------------------------------------------------
        # In retrieval-only mode the raw question is searched directly. When the
        # full pipeline runs, retrieval is scored on the citations the *user*
        # actually received, so multi-turn cases are judged on the rewritten
        # query rather than on a standalone search that never happens in
        # production.
        if expected_docs and orchestrator is None:
            hits = index.search(
                case["question"],
                top_k=top_k,
                candidate_k=settings.candidate_k,
                min_score=settings.min_retrieval_score,
                mmr_lambda=settings.mmr_lambda,
            )
            _score_retrieval(row, [hit.chunk.doc_id for hit in hits], expected_docs)

        # ---- full pipeline ------------------------------------------------
        if orchestrator is not None:
            session_id = None
            for previous in case.get("history", []):
                warmup = await orchestrator.answer(
                    ChatRequest(message=previous, session_id=session_id)
                )
                session_id = warmup.session_id

            started = time.perf_counter()
            response = await orchestrator.answer(
                ChatRequest(message=case["question"], session_id=session_id)
            )
            row["latency_ms"] = int((time.perf_counter() - started) * 1000)

            if expected_docs:
                _score_retrieval(
                    row, [citation.doc_id for citation in response.citations], expected_docs
                )
            if response.rewritten_query:
                row["rewritten"] = response.rewritten_query

            answer_lower = response.answer.lower()
            expected_type = case.get("expected_answer_type", "any")
            row["answer_type"] = response.answer_type.value
            row["type_ok"] = expected_type in ("any", response.answer_type.value)
            row["confidence"] = response.confidence.value
            row["grounding"] = response.grounding_score
            row["tools"] = [call.name for call in response.tool_calls if call.ok]
            if case.get("expected_tool"):
                row["tool_ok"] = case["expected_tool"] in row["tools"]

            required = case.get("must_include", [])
            if required:
                found = [fact for fact in required if fact.lower() in answer_lower]
                row["facts_found"] = len(found)
                row["facts_expected"] = len(required)
                row["missing_facts"] = [f for f in required if f not in found]
                # Was the fact even *available* to the model? This separates a
                # retrieval failure from a generation failure — without it, a
                # single "fact recall" number blames the model for both.
                context_text = (
                    " ".join(
                        chunk_text.get(citation.chunk_id, "") for citation in response.citations
                    )
                    + " "
                    + " ".join(call.summary for call in response.tool_calls)
                ).lower()
                row["facts_in_context"] = sum(
                    1 for fact in required if fact.lower() in context_text
                )

            forbidden = case.get("must_not_include", [])
            if forbidden:
                row["leaked"] = [f for f in forbidden if f.lower() in answer_lower]

            if case.get("expects_caution"):
                # Either a hard abstention, or an explicit caution the user sees.
                row["cautioned"] = (
                    response.answer_type.value == "abstained"
                    or response.confidence.value == "low"
                    or any("not appear anywhere" in notice for notice in response.notices)
                )
            row["answer"] = response.answer[:220]

        rows.append(row)

    return {"rows": rows, "summary": summarise(rows), "top_k": top_k}


def summarise(rows: list[dict]) -> dict:
    retrieval_rows = [row for row in rows if "hit@k" in row]
    typed_rows = [row for row in rows if "type_ok" in row]
    tool_rows = [row for row in rows if "tool_ok" in row]
    fact_rows = [row for row in rows if "facts_expected" in row]
    caution_rows = [row for row in rows if "cautioned" in row]
    latencies = [row["latency_ms"] for row in rows if "latency_ms" in row]
    # Grounding is only meaningful for turns that actually made policy claims:
    # a refusal or an abstention is fixed text with nothing to attribute.
    groundings = [
        row["grounding"]
        for row in rows
        if "grounding" in row and row.get("answer_type") in {"grounded", "tool", "hybrid"}
    ]
    leaks = [row for row in rows if row.get("leaked")]

    def share(items: list[dict], key: str) -> float | None:
        return round(sum(1 for row in items if row[key]) / len(items), 3) if items else None

    return {
        "cases": len(rows),
        "retrieval_cases": len(retrieval_rows),
        "hit@1": share(retrieval_rows, "hit@1"),
        "hit@k": share(retrieval_rows, "hit@k"),
        "mrr": round(
            sum(row["rr"] for row in retrieval_rows) / len(retrieval_rows), 3
        ) if retrieval_rows else None,
        "answer_type_accuracy": share(typed_rows, "type_ok"),
        "tool_selection_accuracy": share(tool_rows, "tool_ok"),
        "fact_in_context": round(
            sum(row.get("facts_in_context", 0) for row in fact_rows)
            / sum(row["facts_expected"] for row in fact_rows),
            3,
        ) if fact_rows else None,
        "fact_recall": round(
            sum(row["facts_found"] for row in fact_rows)
            / sum(row["facts_expected"] for row in fact_rows),
            3,
        ) if fact_rows else None,
        "caution_rate_on_unsupported": share(caution_rows, "cautioned"),
        "access_control_leaks": len(leaks),
        "mean_grounding": round(statistics.fmean(groundings), 3) if groundings else None,
        "latency_p50_ms": int(percentile(latencies, 0.50)) if latencies else None,
        "latency_p95_ms": int(percentile(latencies, 0.95)) if latencies else None,
    }


def print_report(result: dict, verbose: bool) -> None:
    summary = result["summary"]
    print("\n" + "=" * 78)
    print(f"Enterprise Knowledge Copilot — evaluation ({summary['cases']} cases, "
          f"top_k={result['top_k']})")
    print("=" * 78)
    labels = {
        "hit@1": "Retrieval hit@1 (document level)",
        "hit@k": "Retrieval hit@k (document level)",
        "mrr": "Mean reciprocal rank",
        "answer_type_accuracy": "Answer-type routing accuracy",
        "tool_selection_accuracy": "Tool selection accuracy",
        "fact_in_context": "Key fact present in retrieved context",
        "fact_recall": "Key-fact recall in answers",
        "caution_rate_on_unsupported": "Abstained or warned on unsupported",
        "access_control_leaks": "Access-control leaks (want 0)",
        "mean_grounding": "Mean grounding score",
        "latency_p50_ms": "Latency p50 (ms)",
        "latency_p95_ms": "Latency p95 (ms)",
    }
    for key, label in labels.items():
        value = summary.get(key)
        if value is None:
            continue
        print(f"  {label:<42} {value}")

    failures = [
        row
        for row in result["rows"]
        if row.get("hit@k") is False
        or row.get("type_ok") is False
        or row.get("tool_ok") is False
        or row.get("cautioned") is False
        or row.get("leaked")
        or row.get("missing_facts")
    ]
    if failures:
        print(f"\n  {len(failures)} case(s) need attention:")
        for row in failures:
            reasons = []
            if row.get("hit@k") is False:
                reasons.append(f"retrieval miss (got {row.get('retrieved_docs')})")
            if row.get("type_ok") is False:
                reasons.append(f"answer type {row.get('answer_type')}")
            if row.get("tool_ok") is False:
                reasons.append(f"tools {row.get('tools')}")
            if row.get("cautioned") is False:
                reasons.append("no caution on an unsupported question")
            if row.get("missing_facts"):
                reasons.append(f"missing {row['missing_facts']}")
            if row.get("leaked"):
                reasons.append(f"LEAKED {row['leaked']}")
            print(f"    - {row['id']:<14} {'; '.join(reasons)}")
            if verbose:
                print(f"        Q: {row['question']}")
                print(f"        A: {row.get('answer', '')[:180]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the copilot pipeline.")
    parser.add_argument("--provider", help="Override COPILOT_LLM_PROVIDER for this run")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", dest="json_out", help="Write full results to this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(evaluate(args.provider, args.retrieval_only, args.top_k))
    print_report(result, args.verbose)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  full results written to {args.json_out}\n")


if __name__ == "__main__":
    main()
