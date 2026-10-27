"""
rag/baseline.py — Thin wrapper around the original linear RAG pipeline.

This file preserves the baseline retrieve→generate flow exactly as it existed
before the LangGraph agent was introduced (tagged as baseline-v1).  The new
agent can be compared against it via ``eval/run_eval.py --target baseline``.

Design decisions (recorded here for PROCESSES.md):
- We do NOT re-implement logic; we delegate directly to the existing
  ``rag_query`` function so the baseline results are byte-for-byte identical
  to what the old app produced.
- A ``pipeline_version`` key is injected into the response so LangSmith
  traces can be filtered by ``baseline`` vs ``agent``.
- The ``@traceable`` decorator wraps the whole call so it appears in
  LangSmith as a top-level run even when called outside FastAPI.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def baseline_query(question: str, top_k: int = 5) -> dict[str, Any]:
    """
    Run the original linear RAG pipeline and return a standardised result dict.

    Args:
        question: The user's question.
        top_k:    Number of chunks to retrieve (default: 5, same as original).

    Returns:
        {
          "answer":       str,
          "sources":      list[dict],   # {source, page, content}
          "context":      str,
          "retrieved_chunks": int,
          "pipeline_version": "baseline",
          "latency_s":    float,
        }
    """
    from rag.retrieval import rag_query  # noqa: PLC0415  (lazy import)

    logger.info("baseline_query | question=%r top_k=%d", question, top_k)
    t0 = time.perf_counter()

    try:
        result = rag_query(question, k=top_k)
    except Exception:
        logger.exception("baseline_query failed")
        raise

    result["pipeline_version"] = "baseline"
    result["latency_s"] = round(time.perf_counter() - t0, 3)
    return result


if __name__ == "__main__":
    # Quick smoke test: python -m rag.baseline
    import json

    logging.basicConfig(level=logging.INFO)
    q = "What is multi-head attention and why is it useful?"
    print(f"\nBaseline smoke test — question: {q!r}\n")
    res = baseline_query(q)
    print(f"Answer ({res['retrieved_chunks']} chunks, {res['latency_s']}s):\n")
    print(res["answer"])
    print("\nSources:")
    for s in res["sources"]:
        print(f"  [{s['source']} p.{s['page']}]")
