"""
rag/graph/edges.py — Conditional edge (routing) functions for the LangGraph agent.

Each function receives AgentState and returns the name of the next node
(a string that matches a node key in the graph).  LangGraph uses these to
choose which branch to follow.

Edge functions are kept here (separate from nodes.py) so the routing logic can
be tested independently from the node logic.
"""

from __future__ import annotations

from rag.graph.state import AgentState
from rag.graph.nodes import MAX_RETRIES, MAX_REGENERATIONS


def route_after_classify(state: AgentState) -> str:
    """
    After classify_route, branch to the appropriate sub-pipeline.

    Returns:
        "fallback"         – off_topic questions
        "decompose"        – compare route (needs query decomposition)
        "tool_agent"       – tool route (calculator, arXiv, list docs)
        "retrieve"         – rag route (standard retrieval)
    """
    route = state.get("route", "rag")
    if route == "off_topic":
        return "fallback"
    if route == "compare":
        return "decompose"
    if route == "tool":
        return "tool_agent"
    return "retrieve"


def route_after_grade(state: AgentState) -> str:
    """
    After grade_docs, decide whether to generate, rewrite, or fall back.

    Logic:
    - If irrelevant AND retries < MAX_RETRIES → rewrite_query (corrective loop)
    - If irrelevant AND retries >= MAX_RETRIES → fallback (honest "not found")
    - If relevant or partial → generate
    """
    relevance = state.get("relevance", "relevant")
    retries = state.get("retries", 0)

    if relevance == "irrelevant":
        if retries < MAX_RETRIES:
            return "rewrite_query"
        return "fallback"

    return "generate"


def route_after_grounding(state: AgentState) -> str:
    """
    After check_grounding, decide whether to regenerate or finalize.

    Logic:
    - If not grounded AND regenerations < MAX_REGENERATIONS → regenerate
    - Otherwise → finalize (with a caveat note if still ungrounded)
    """
    grounded = state.get("grounded", True)
    regenerations = state.get("regenerations", 0)

    if not grounded and regenerations < MAX_REGENERATIONS:
        return "regenerate"

    return "finalize"
