"""
rag/graph/builder.py — Assemble and compile the LangGraph agent graph.

The graph is built once and compiled with an optional checkpointer for
persistent multi-turn memory.  The compiled graph is cached as a module-level
singleton so subsequent requests (both FastAPI and Gradio) reuse it.

Checkpointer strategy (documented design decision):
- Locally: ``SqliteSaver`` (langgraph-checkpoint-sqlite) backed by
  ``checkpoints.sqlite`` in the project root.
- If the file system is read-only (HuggingFace Spaces without persistent
  storage): fall back to ``MemorySaver`` and log a warning.
- Tests can pass ``checkpointer=None`` to get a non-checkpointed graph that
  is faster to compile and doesn't touch the filesystem.

Thread isolation: every conversation gets a unique ``thread_id`` so state
never leaks between users.

Recursion limit: set to 25 steps to enforce a hard cap on the graph's total
step budget, preventing any accidental infinite loops.
"""

from __future__ import annotations

import logging
import os

from langgraph.graph import StateGraph, START, END

from rag.graph.state import AgentState
from rag.graph.nodes import (
    rewrite_with_history,
    classify_route,
    decompose,
    retrieve,
    retrieve_parallel,
    grade_docs,
    rewrite_query,
    generate,
    check_grounding,
    regenerate,
    tool_agent,
    fallback,
    finalize,
)
from rag.graph.edges import (
    route_after_classify,
    route_after_grade,
    route_after_grounding,
)

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 25
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoints.sqlite")

# ── Module-level compiled graph singleton ─────────────────────────────────────
_compiled_graph = None


def _make_checkpointer():
    """
    Create a checkpointer appropriate for the current environment.

    Returns a SqliteSaver if the filesystem is writable, else MemorySaver.
    """
    try:
        import sqlite3  # noqa: PLC0415
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415
        conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as exc:
        logger.warning(
            "SqliteSaver unavailable (%s); falling back to MemorySaver. "
            "Conversation history will NOT persist across process restarts.",
            exc,
        )
        from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
        return MemorySaver()


def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph agent.

    Args:
        checkpointer: A LangGraph checkpointer instance.  If None, creates
                      the environment-appropriate one via ``_make_checkpointer``.
                      Pass ``False`` to disable checkpointing (useful in tests).

    Returns:
        A compiled LangGraph runnable.
    """
    if checkpointer is None:
        checkpointer = _make_checkpointer()

    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("rewrite_with_history", rewrite_with_history)
    builder.add_node("classify_route", classify_route)
    builder.add_node("decompose", decompose)
    builder.add_node("retrieve", retrieve)
    builder.add_node("retrieve_parallel", retrieve_parallel)
    builder.add_node("grade_docs", grade_docs)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("check_grounding", check_grounding)
    builder.add_node("regenerate", regenerate)
    builder.add_node("tool_agent", tool_agent)
    builder.add_node("fallback", fallback)
    builder.add_node("finalize", finalize)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "rewrite_with_history")
    builder.add_edge("rewrite_with_history", "classify_route")

    # ── Routing from classifier ───────────────────────────────────────────────
    builder.add_conditional_edges(
        "classify_route",
        route_after_classify,
        {
            "fallback": "fallback",
            "decompose": "decompose",
            "tool_agent": "tool_agent",
            "retrieve": "retrieve",
        },
    )

    # ── Compare sub-pipeline ──────────────────────────────────────────────────
    builder.add_edge("decompose", "retrieve_parallel")
    builder.add_edge("retrieve_parallel", "grade_docs")

    # ── RAG sub-pipeline ──────────────────────────────────────────────────────
    builder.add_edge("retrieve", "grade_docs")

    # ── Corrective retrieval loop ─────────────────────────────────────────────
    builder.add_conditional_edges(
        "grade_docs",
        route_after_grade,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
            "fallback": "fallback",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")

    # ── Generation + grounding check loop ────────────────────────────────────
    builder.add_edge("generate", "check_grounding")
    builder.add_conditional_edges(
        "check_grounding",
        route_after_grounding,
        {
            "regenerate": "regenerate",
            "finalize": "finalize",
        },
    )
    builder.add_edge("regenerate", "check_grounding")

    # ── Terminal nodes ────────────────────────────────────────────────────────
    builder.add_edge("tool_agent", "finalize")
    builder.add_edge("fallback", END)
    builder.add_edge("finalize", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_kwargs: dict = {}
    if checkpointer is not False:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)
    logger.info("LangGraph agent compiled (nodes=%d, recursion_limit=%d)", 13, RECURSION_LIMIT)
    return graph


def get_graph():
    """
    Return the module-level compiled graph singleton (built on first call).

    All FastAPI endpoints and Gradio handlers should call this rather than
    calling ``build_graph()`` directly, to avoid recompiling on every request.
    """
    global _compiled_graph  # noqa: PLW0603
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(question: str, thread_id: str, history: list | None = None) -> dict:
    """
    Convenience wrapper: run one turn of the agent and return the result dict.

    Args:
        question:  The user's question for this turn.
        thread_id: Conversation identifier (persisted in the checkpointer).
        history:   Optional list of prior messages (for stateless callers).

    Returns:
        {
          "answer":     str,
          "citations":  list[dict],
          "confidence": float,
          "route":      str,
          "steps":      list[str],   # trace_notes
          "thread_id":  str,
        }
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    initial_state: AgentState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "retries": 0,
        "regenerations": 0,
        "trace_notes": [],
        "retrieved": [],
        "sub_questions": [],
        "citations": [],
    }

    if history:
        initial_state["messages"] = history + [HumanMessage(content=question)]

    final_state = graph.invoke(initial_state, config=config)

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "retrieved": final_state.get("retrieved", []),
        "confidence": final_state.get("confidence", 0.0),
        "route": final_state.get("route", "rag"),
        "steps": final_state.get("trace_notes", []),
        "thread_id": thread_id,
    }
