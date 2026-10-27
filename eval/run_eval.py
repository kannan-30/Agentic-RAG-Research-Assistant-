"""
eval/run_eval.py — CLI to run evaluation experiments against the LangSmith dataset.

Usage:
    python -m eval.run_eval --target baseline --experiment-prefix phase0-baseline
    python -m eval.run_eval --target agent   --experiment-prefix phase2-agent
    python -m eval.run_eval --target agent   --experiment-prefix sweep-chunk500 --limit 10

The script:
1. Loads examples from LangSmith (or falls back to local dataset.jsonl)
2. Runs each example through the target pipeline
3. Scores with the evaluators in evaluators.py
4. Saves results to eval/results/<experiment>.json
5. Prints a summary table (mean per metric, per category)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_DATASET_NAME = "agentic-rag-eval-v1"


# ── Target pipeline runners ───────────────────────────────────────────────────

def run_baseline(question: str, expected_sources: list[str], category: str) -> dict[str, Any]:
    """Run the baseline linear RAG pipeline."""
    from rag.baseline import baseline_query  # noqa: PLC0415
    t0 = time.perf_counter()
    result = baseline_query(question, top_k=5)
    latency = time.perf_counter() - t0
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "context": result.get("context", ""),
        "latency_s": round(latency, 3),
        "pipeline": "baseline",
        "llm_calls": 1,  # baseline makes exactly 1 LLM call
    }


def run_agent(question: str, expected_sources: list[str], category: str) -> dict[str, Any]:
    """Run the LangGraph agentic pipeline."""
    import uuid  # noqa: PLC0415
    from rag.graph.builder import run_agent as _run  # noqa: PLC0415
    t0 = time.perf_counter()
    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    result = _run(question=question, thread_id=thread_id)
    latency = time.perf_counter() - t0
    steps = result.get("steps", [])
    # Estimate LLM calls from trace_notes (each node with LLM appears in steps)
    llm_node_keywords = ["route", "grade", "rewrite_query", "generated", "grounding", "regenerated", "decompose"]
    llm_calls = sum(1 for s in steps if any(k in s for k in llm_node_keywords))
    retrieved_chunks = result.get("retrieved", [])
    # For retrieval_hit: use all retrieved document sources
    retrieved_sources = [c.get("source", "") for c in retrieved_chunks if isinstance(c, dict) and c.get("source")]
    if not retrieved_sources:
        retrieved_sources = [c.get("source", "") if isinstance(c, dict) else str(c) for c in result.get("citations", [])]

    # Supply retrieved context for groundedness evaluation
    context_text = "\n\n".join(c.get("text", "") for c in retrieved_chunks if isinstance(c, dict))

    return {
        "answer": result.get("answer", ""),
        "sources": retrieved_sources,
        "context": context_text,
        "latency_s": round(latency, 3),
        "pipeline": "agent",
        "llm_calls": max(llm_calls, 1),
        "route": result.get("route", ""),
        "confidence": result.get("confidence", 0.0),
        "steps": steps,
    }


RUNNERS = {"baseline": run_baseline, "agent": run_agent}


# ── Evaluate one example ──────────────────────────────────────────────────────

def evaluate_example(
    example: dict,
    runner_fn,
    evaluators: list,
) -> dict[str, Any]:
    """Run one example through the pipeline and all evaluators."""
    question = example["question"]
    expected_sources = example.get("expected_sources", [])
    reference_answer = example.get("reference_answer", "")
    category = example.get("category", "factual")

    # Run the pipeline
    try:
        output = runner_fn(question, expected_sources, category)
    except Exception as exc:
        logger.error("Pipeline error for %r: %s", question[:60], exc)
        output = {"answer": f"ERROR: {exc}", "sources": [], "context": "", "latency_s": 0.0, "llm_calls": 0}

    # Score with each evaluator
    scores = {}
    for ev in evaluators:
        try:
            # Adapt to evaluator's expected interface (run, example dicts)
            class FakeRun:
                def __init__(self, outputs):
                    self.outputs = outputs

            class FakeExample:
                def __init__(self, inputs, outputs):
                    self.inputs = inputs
                    self.outputs = outputs

            fake_run = FakeRun(output)
            fake_example = FakeExample(
                inputs={"question": question, "expected_sources": expected_sources, "category": category},
                outputs={"reference_answer": reference_answer},
            )
            result = ev(fake_run, fake_example)
            scores[result["key"]] = result["score"]
        except Exception as exc:
            logger.warning("Evaluator %s failed: %s", ev.__name__, exc)
            scores[ev.__name__] = None

    return {
        "question": question,
        "category": category,
        "answer": output.get("answer", ""),
        "latency_s": output.get("latency_s", 0.0),
        "llm_calls": output.get("llm_calls", 0),
        "route": output.get("route", ""),
        "confidence": output.get("confidence", 0.0),
        **scores,
    }


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(results: list[dict], experiment: str) -> None:
    """Print a formatted summary table of evaluation results."""
    metrics = ["correctness", "groundedness", "retrieval_hit", "refusal_accuracy"]
    categories = list({r["category"] for r in results})

    print(f"\n{'='*70}")
    print(f"  Experiment: {experiment}")
    print(f"  N = {len(results)} examples")
    print(f"{'='*70}")

    # Overall means
    print("\nOverall metrics:")
    for m in metrics:
        scores = [r.get(m) for r in results if r.get(m) is not None]
        if scores:
            print(f"  {m:22s}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")

    # Latency and LLM calls
    latencies = [r["latency_s"] for r in results if r.get("latency_s")]
    calls = [r["llm_calls"] for r in results if r.get("llm_calls")]
    if latencies:
        print(f"\n  {'latency_s (mean)':22s}: {sum(latencies)/len(latencies):.2f}s")
    if calls:
        print(f"  {'llm_calls (mean)':22s}: {sum(calls)/len(calls):.1f}")

    # Per-category correctness
    print("\nCorrectness by category:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_scores = [r.get("correctness") for r in cat_results if r.get("correctness") is not None]
        if cat_scores:
            print(f"  {cat:18s}: {sum(cat_scores)/len(cat_scores):.3f}  (n={len(cat_scores)})")

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments.")
    parser.add_argument(
        "--target",
        choices=["baseline", "agent"],
        required=True,
        help="Which pipeline to evaluate.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="experiment",
        help="Prefix for the experiment name (timestamp appended).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N examples (for quick smoke tests).",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent evaluations (keep at 1 for free-tier rate limits).",
    )
    args = parser.parse_args()

    experiment_name = f"{args.experiment_prefix}-{int(time.time())}"
    logger.info("Starting experiment: %s (target=%s)", experiment_name, args.target)

    # Load examples from local dataset file (supports JSON array or JSONL)
    examples = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        # JSON array format
        examples = json.loads(raw)
    else:
        # JSONL format (one JSON object per line)
        for line in raw.splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.limit:
        examples = examples[: args.limit]
        logger.info("Limited to %d examples.", args.limit)

    logger.info("Loaded %d examples.", len(examples))

    runner_fn = RUNNERS[args.target]

    from eval.evaluators import ALL_EVALUATORS  # noqa: PLC0415

    results = []
    for i, example in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] {example['question'][:70]}", flush=True)
        row = evaluate_example(example, runner_fn, ALL_EVALUATORS)
        results.append(row)
        # Polite delay to avoid rate-limiting the judge LLM
        if i < len(examples):
            time.sleep(1.5)

    # Save results
    output_path = RESULTS_DIR / f"{experiment_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "target": args.target,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_examples": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved to %s", output_path)

    print_summary(results, experiment_name)

    # Upload to LangSmith if tracing is enabled
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith import Client  # noqa: PLC0415
            client = Client()
            logger.info("LangSmith client available — traces should appear in project: %s",
                        os.getenv("LANGSMITH_PROJECT", "agentic-rag"))
        except Exception as exc:
            logger.warning("LangSmith upload skipped: %s", exc)


if __name__ == "__main__":
    main()
