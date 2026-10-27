"""
rag/graph/state.py — AgentState TypedDict and reducers.

All inter-node communication happens through this single state dict.
LangGraph merges state updates returned by each node.

Design decision: we use ``add_messages`` from langgraph.graph.message for the
``messages`` field so that multi-turn history accumulates correctly (the
reducer appends rather than replaces).  All other fields use the default
``last-write-wins`` reducer because nodes overwrite them explicitly.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Mutable state threaded through every node of the LangGraph agent.

    Fields marked ``Annotated[..., add_messages]`` use the reducer that
    *appends* new messages to the list rather than replacing it.  All other
    fields use last-write-wins (the langgraph default).
    """

    # ── Conversation history ──────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    """Full multi-turn message history including the current human turn."""

    # ── Current turn inputs ───────────────────────────────────────────────────
    question: str
    """Raw question as submitted by the user this turn."""

    standalone_question: str
    """
    Standalone version of the question after history-aware rewriting.
    If there is no prior history the rewrite node passes the question through
    unchanged (skipping the LLM call).
    """

    # ── Routing ───────────────────────────────────────────────────────────────
    route: Literal["rag", "compare", "tool", "off_topic"]
    """
    Route chosen by the classifier node:
    - ``rag``       – single-paper or general lookup
    - ``compare``   – cross-paper comparison (decomposed into sub-questions)
    - ``tool``      – question best answered by a tool (calculator, arXiv, etc.)
    - ``off_topic`` – clearly unrelated to the document corpus
    """

    route_reason: str
    """One-line explanation of why the classifier chose this route (for tracing)."""

    # ── Decomposition (compare route) ─────────────────────────────────────────
    sub_questions: list[str]
    """2–4 focused sub-questions decomposed from the original comparison query."""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved: list[dict]
    """
    Retrieved chunks: each entry is
    ``{text: str, score: float, source: str, page: int, chunk_id: str}``.
    """

    # ── Grading ───────────────────────────────────────────────────────────────
    relevance: Literal["relevant", "partial", "irrelevant"]
    """
    Grader's verdict on whether the retrieved context can answer the question.
    ``partial`` means enough to attempt an answer but with caveats.
    """

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    """The final generated answer string (with inline citations like [source p.N])."""

    citations: list[dict]
    """De-duplicated list of cited sources: ``{source: str, page: int}``."""

    grounded: bool
    """
    True when the grounding-check node confirms every claim is supported by the
    retrieved context.  False triggers one regeneration attempt.
    """

    unsupported_claims: list[str]
    """Claims flagged as unsupported by the grounding check (used in regeneration prompt)."""

    # ── Loop counters (safety caps) ───────────────────────────────────────────
    retries: int
    """Number of query-rewrite retries attempted (hard cap: 2)."""

    regenerations: int
    """Number of answer regenerations attempted (hard cap: 1)."""

    # ── Confidence heuristic ──────────────────────────────────────────────────
    confidence: float
    """
    0–1 heuristic combining:
      0.4 × top_chunk_score  (normalised cosine)
    + 0.3 × relevance_score  (relevant=1, partial=0.5, irrelevant=0)
    + 0.2 × grounded_score   (grounded=1, else 0)
    + 0.1 × retry_penalty    (1 − retries/2)
    See docs/PROCESSES.md §"Confidence heuristic" for rationale and caveats.
    This is explicitly a heuristic, not a calibrated probability.
    """

    # ── Observability ─────────────────────────────────────────────────────────
    trace_notes: list[str]
    """
    Human-readable log of major steps taken this turn (shown in the Gradio UI
    and recorded in the LangSmith trace).  Each node appends one line.
    """
