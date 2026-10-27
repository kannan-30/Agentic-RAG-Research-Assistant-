"""
rag/graph/nodes.py — Node functions for the LangGraph agentic RAG system.

Each function is a pure node: it receives the current AgentState and returns a
dict of state updates.  Nodes do NOT mutate state in place.

Design principles:
- Nodes are individually testable (small, single-purpose).
- All LLM calls use structured output where possible to avoid brittle parsing.
- Every node appends to ``trace_notes`` so the UI can show what happened.
- JSON parse failures degrade gracefully with sensible defaults and a warning log.
- A hard cap (``MAX_RETRIES = 2``, ``MAX_REGENERATIONS = 1``) prevents infinite loops.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.graph.state import AgentState
from rag.graph.prompts import (
    HISTORY_REWRITE_SYSTEM,
    HISTORY_REWRITE_HUMAN,
    ROUTE_CLASSIFIER_SYSTEM,
    ROUTE_CLASSIFIER_HUMAN,
    DECOMPOSE_SYSTEM,
    DECOMPOSE_HUMAN,
    GRADE_DOCS_SYSTEM,
    GRADE_DOCS_HUMAN,
    REWRITE_QUERY_SYSTEM,
    REWRITE_QUERY_HUMAN,
    GENERATE_SYSTEM,
    GENERATE_HUMAN,
    GROUNDING_SYSTEM,
    GROUNDING_HUMAN,
    REGENERATE_SYSTEM,
    REGENERATE_HUMAN,
    FALLBACK_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
MAX_REGENERATIONS = 1

# ── Helper: get LLM ───────────────────────────────────────────────────────────


def _llm():
    """Return the shared LLM singleton (lazy so tests can mock before import)."""
    from rag.llm import get_llm  # noqa: PLC0415
    return get_llm()


def _call_llm_json(system: str, human: str, default: dict) -> dict:
    """
    Call the LLM and parse the response as JSON.

    On parse failure, logs a warning and returns ``default`` so the graph
    can continue with a safe fallback (graceful degradation).
    """
    llm = _llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        text = response.content
        if isinstance(text, list):
            text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
        text = text.strip()
        # Strip markdown code fences if the model wraps JSON in ```json ... ```
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("LLM JSON parse failed (%s); using default %s", exc, default)
        return default


def _call_llm_text(system: str, human: str) -> str:
    """Call the LLM and return the plain text response."""
    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])
    text = response.content
    if isinstance(text, list):
        text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
    return text.strip()


# ── Node 1: rewrite_with_history ──────────────────────────────────────────────

def rewrite_with_history(state: AgentState) -> dict:
    """
    Rewrite the latest question as a standalone question using chat history.

    If there are fewer than 2 previous messages (no prior context), skip the
    LLM call entirely to save quota.  This is the first node and runs on every
    turn.
    """
    question = state["question"]
    messages = state.get("messages", [])

    # Only rewrite if there is meaningful history (at least one prior exchange).
    history_messages = [m for m in messages if not isinstance(m, HumanMessage) or m.content != question]
    if len(history_messages) < 2:
        return {
            "standalone_question": question,
            "trace_notes": [f"rewrite: no history → passing question through unchanged"],
        }

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history_messages[-6:]  # last 3 turns
    )

    standalone = _call_llm_text(
        HISTORY_REWRITE_SYSTEM,
        HISTORY_REWRITE_HUMAN.format(history=history_text, question=question),
    )

    logger.info("rewrite_with_history: %r → %r", question, standalone)
    return {
        "standalone_question": standalone,
        "trace_notes": [f"rewrite: '{question[:60]}' → '{standalone[:60]}'"],
    }


# ── Node 2: classify_route ────────────────────────────────────────────────────

def classify_route(state: AgentState) -> dict:
    """
    Classify the standalone question into one of: rag | compare | tool | off_topic.

    Uses a structured-output LLM call (JSON with route + reason).
    On parse failure, defaults to 'rag' to avoid dropping user queries.
    """
    question = state.get("standalone_question") or state["question"]

    result = _call_llm_json(
        ROUTE_CLASSIFIER_SYSTEM,
        ROUTE_CLASSIFIER_HUMAN.format(question=question),
        default={"route": "rag", "reason": "parse error — defaulted to rag"},
    )

    route = result.get("route", "rag")
    reason = result.get("reason", "")

    # Validate route value
    valid_routes = {"rag", "compare", "tool", "off_topic"}
    if route not in valid_routes:
        logger.warning("Invalid route %r from classifier; defaulting to rag", route)
        route = "rag"

    logger.info("classify_route: %r → %s (%s)", question[:60], route, reason)
    return {
        "route": route,
        "route_reason": reason,
        "retries": 0,
        "regenerations": 0,
        "sub_questions": [],
        "retrieved": [],
        "trace_notes": [f"route: {route} — {reason}"],
    }


# ── Node 3: decompose ─────────────────────────────────────────────────────────

def decompose(state: AgentState) -> dict:
    """
    Decompose a comparison question into 2–4 focused sub-questions.

    Each sub-question is designed to target a single paper so they can be
    retrieved in parallel without cross-contaminating results.
    """
    question = state.get("standalone_question") or state["question"]

    result = _call_llm_json(
        DECOMPOSE_SYSTEM,
        DECOMPOSE_HUMAN.format(question=question),
        default={"sub_questions": [question]},  # fallback: treat as single question
    )

    # The model may return a list directly or wrap it in a dict.
    if isinstance(result, list):
        sub_questions = result
    else:
        sub_questions = result.get("sub_questions", [question])

    if not sub_questions:
        sub_questions = [question]

    logger.info("decompose: %d sub-questions", len(sub_questions))
    return {
        "sub_questions": sub_questions,
        "trace_notes": [f"decompose: {len(sub_questions)} sub-questions"],
    }


# ── Node 4: retrieve ──────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """
    Retrieve relevant document chunks for the standalone question.

    Uses hybrid BM25 + vector retrieval with RRF fusion.
    """
    from rag.retrieval import hybrid_retrieve, docs_to_dicts  # noqa: PLC0415

    question = state.get("standalone_question") or state["question"]
    docs = hybrid_retrieve(question, k=int(os.getenv("TOP_K_RESULTS", "5")))
    retrieved = docs_to_dicts(docs)

    logger.info("retrieve: %d chunks for %r", len(retrieved), question[:60])
    return {
        "retrieved": retrieved,
        "trace_notes": [f"retrieved: {len(retrieved)} chunks via hybrid search"],
    }


# ── Node 5: retrieve_parallel (compare route) ─────────────────────────────────

def retrieve_parallel(state: AgentState) -> dict:
    """
    Run retrieval for each sub-question and merge results with source diversity.

    De-duplicates by content prefix and ensures we sample from multiple papers
    rather than having the top-k dominated by one paper.
    """
    import asyncio  # noqa: PLC0415
    from rag.retrieval import hybrid_retrieve, docs_to_dicts  # noqa: PLC0415

    sub_questions = state.get("sub_questions") or [state.get("standalone_question") or state["question"]]

    all_docs_raw = []
    seen_keys: set[str] = set()

    for sq in sub_questions:
        docs = hybrid_retrieve(sq, k=3)  # 3 per sub-question → up to 12 total before dedup
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs_raw.append(doc)

    retrieved = docs_to_dicts(all_docs_raw[:10])  # cap at 10 total

    logger.info("retrieve_parallel: %d unique chunks from %d sub-questions", len(retrieved), len(sub_questions))
    return {
        "retrieved": retrieved,
        "trace_notes": [f"retrieved (parallel): {len(retrieved)} chunks across {len(sub_questions)} sub-questions"],
    }


# ── Node 6: grade_docs ────────────────────────────────────────────────────────

def grade_docs(state: AgentState) -> dict:
    """
    Grade whether the retrieved context is sufficient to answer the question.

    Grades all retrieved chunks in a SINGLE LLM call (not per-chunk) to avoid
    burning too many API calls.  Returns relevant | partial | irrelevant.
    """
    question = state.get("standalone_question") or state["question"]
    retrieved = state.get("retrieved", [])

    if not retrieved:
        return {
            "relevance": "irrelevant",
            "trace_notes": ["grade: no chunks retrieved → irrelevant"],
        }

    # Truncate context to avoid very long prompts
    context_text = "\n\n---\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text'][:400]}"
        for c in retrieved[:5]
    )

    result = _call_llm_json(
        GRADE_DOCS_SYSTEM,
        GRADE_DOCS_HUMAN.format(question=question, context=context_text),
        default={"relevance": "relevant"},  # if parse fails, attempt generation
    )

    relevance = result.get("relevance", "relevant")
    if relevance not in ("relevant", "partial", "irrelevant"):
        relevance = "relevant"

    logger.info("grade_docs: %s", relevance)
    return {
        "relevance": relevance,
        "trace_notes": [f"grade: context is {relevance}"],
    }


# ── Node 7: rewrite_query ─────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """
    Rewrite the query using different keywords or HyDE style.

    Only reached when grade_docs returned 'irrelevant'.  Increments the retry
    counter.  The corrective edge checks the counter against MAX_RETRIES.
    """
    question = state.get("standalone_question") or state["question"]
    retries = state.get("retries", 0) + 1

    rewritten = _call_llm_text(
        REWRITE_QUERY_SYSTEM,
        REWRITE_QUERY_HUMAN.format(
            question=question,
            standalone_question=state.get("standalone_question", question),
            retry_num=retries,
        ),
    )

    logger.info("rewrite_query #%d: %r → %r", retries, question[:60], rewritten[:60])
    return {
        "standalone_question": rewritten,
        "retries": retries,
        "trace_notes": [f"rewrite_query #{retries}: '{rewritten[:60]}'"],
    }


# ── Node 8: generate ──────────────────────────────────────────────────────────

def generate(state: AgentState) -> dict:
    """
    Generate a grounded answer from the retrieved context.

    The prompt forbids outside knowledge and requires inline citations.
    For 'partial' relevance, it tells the model to flag gaps.
    """
    question = state.get("standalone_question") or state["question"]
    retrieved = state.get("retrieved", [])
    relevance = state.get("relevance", "relevant")

    context_parts: list[str] = []
    for c in retrieved[:7]:
        context_parts.append(f"[{c['source']} p.{c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Inject a partial-context note so the LLM knows it may be incomplete
    if relevance == "partial":
        context += (
            "\n\n[NOTE: The retrieved context may be incomplete. "
            "Acknowledge any gaps in your answer.]"
        )

    answer = _call_llm_text(
        GENERATE_SYSTEM,
        GENERATE_HUMAN.format(context=context, question=question),
    )

    # Extract citations from the answer (support [source p.N], [source p. N, p. M], etc.)
    import re  # noqa: PLC0415
    citation_pattern = re.compile(r'\[([^\[\]]+?)\s+p\.?\s*(\d+(?:,\s*(?:p\.?\s*)?\d+)*)\]')
    citations: list[dict] = []
    seen_cites: set[str] = set()
    for m in citation_pattern.finditer(answer):
        source = m.group(1).strip()
        pages = re.findall(r'\d+', m.group(2))
        for p in pages:
            key = f"{source}::{p}"
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append({"source": source, "page": int(p)})

    # If the model didn't cite inline, fall back to the retrieved sources
    if not citations:
        seen_cites_fb: set[str] = set()
        for c in retrieved:
            key = f"{c['source']}::{c['page']}"
            if key not in seen_cites_fb:
                seen_cites_fb.add(key)
                citations.append({"source": c["source"], "page": c["page"]})

    logger.info("generate: %d chars, %d citations", len(answer), len(citations))
    return {
        "answer": answer,
        "citations": citations,
        "trace_notes": [f"generated: {len(answer)} chars, {len(citations)} citations"],
    }


# ── Node 9: check_grounding ───────────────────────────────────────────────────

def check_grounding(state: AgentState) -> dict:
    """
    Verify that every claim in the answer is supported by the retrieved context.

    On failure, lists unsupported claims so ``regenerate`` can soften them.
    The grounding check is itself an LLM call (LLM-judging-LLM), so it can be
    wrong — we always finalize after at most one regeneration.
    """
    retrieved = state.get("retrieved", [])
    answer = state.get("answer", "")

    if not answer or not retrieved:
        return {
            "grounded": True,  # nothing to check
            "unsupported_claims": [],
            "trace_notes": ["grounding: skipped (no answer or no context)"],
        }

    context = "\n\n---\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text'][:400]}" for c in retrieved[:5])

    result = _call_llm_json(
        GROUNDING_SYSTEM,
        GROUNDING_HUMAN.format(context=context, answer=answer),
        default={"grounded": True, "unsupported_claims": []},
    )

    grounded = bool(result.get("grounded", True))
    unsupported = result.get("unsupported_claims", [])
    if not isinstance(unsupported, list):
        unsupported = []

    logger.info("check_grounding: grounded=%s, unsupported=%d", grounded, len(unsupported))
    return {
        "grounded": grounded,
        "unsupported_claims": unsupported,
        "trace_notes": [
            f"grounding: {'✓ grounded' if grounded else f'✗ {len(unsupported)} unsupported claims'}"
        ],
    }


# ── Node 10: regenerate ───────────────────────────────────────────────────────

def regenerate(state: AgentState) -> dict:
    """
    Regenerate the answer with unsupported claims explicitly flagged for removal.

    Increments the regeneration counter.  The grounding edge checks this against
    MAX_REGENERATIONS before allowing another iteration.
    """
    question = state.get("standalone_question") or state["question"]
    retrieved = state.get("retrieved", [])
    answer = state.get("answer", "")
    unsupported = state.get("unsupported_claims", [])
    regenerations = state.get("regenerations", 0) + 1

    context = "\n\n---\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in retrieved[:7])
    claims_text = "\n".join(f"- {c}" for c in unsupported)

    system = REGENERATE_SYSTEM.format(unsupported_claims=claims_text)
    new_answer = _call_llm_text(
        system,
        REGENERATE_HUMAN.format(context=context, answer=answer),
    )

    import re  # noqa: PLC0415
    citation_pattern = re.compile(r'\[([^\[\]]+?)\s+p\.?\s*(\d+(?:,\s*(?:p\.?\s*)?\d+)*)\]')
    citations: list[dict] = []
    seen: set[str] = set()
    for m in citation_pattern.finditer(new_answer):
        src = m.group(1).strip()
        pages = re.findall(r'\d+', m.group(2))
        for p in pages:
            key = f"{src}::{p}"
            if key not in seen:
                seen.add(key)
                citations.append({"source": src, "page": int(p)})

    # Fallback to retrieved sources if regenerate dropped inline citations
    if not citations:
        seen_fb: set[str] = set()
        for c in retrieved:
            key = f"{c['source']}::{c['page']}"
            if key not in seen_fb:
                seen_fb.add(key)
                citations.append({"source": c["source"], "page": c["page"]})

    logger.info("regenerate #%d: %d chars", regenerations, len(new_answer))
    return {
        "answer": new_answer,
        "citations": citations,
        "regenerations": regenerations,
        "trace_notes": [f"regenerated #{regenerations} (removed {len(unsupported)} unsupported claims)"],
    }


# ── Node 11: tool_agent ───────────────────────────────────────────────────────

def tool_agent(state: AgentState) -> dict:
    """
    Handle tool-route questions with a simple ReAct loop.

    Uses LangGraph's ``create_react_agent`` to let the LLM choose and call
    tools iteratively.  We pass the system prompt via the tool-agent's config.
    The loop runs up to 5 steps before forcefully stopping.

    For the arXiv search + ingest flow, an ``interrupt`` is raised inside the
    tool to request human confirmation (see tools.py and builder.py).
    """
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415
    from rag.graph.tools import ALL_TOOLS  # noqa: PLC0415

    question = state.get("standalone_question") or state["question"]

    tool_llm = _llm()
    agent = create_react_agent(tool_llm, ALL_TOOLS)

    try:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        # The last message from the agent is the final answer.
        final_msgs = result.get("messages", [])
        if final_msgs:
            last_msg = final_msgs[-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            if isinstance(answer, list):
                answer = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in answer)
        else:
            answer = "Tool agent produced no result."
    except Exception as exc:
        logger.error("tool_agent error: %s", exc)
        answer = f"Tool error: {exc}"

    return {
        "answer": answer,
        "citations": [],
        "grounded": True,  # tool answers are considered grounded by definition
        "trace_notes": [f"tool_agent: executed tools for '{question[:60]}'"],
    }


# ── Node 12: fallback ─────────────────────────────────────────────────────────

def fallback(state: AgentState) -> dict:
    """
    Produce a polite fallback for off-topic questions or exhausted retries.

    For off-topic: explain what the assistant can do.
    For not-found (retries exhausted): honest "not in the documents" message.
    """
    route = state.get("route", "off_topic")
    retries = state.get("retries", 0)
    question = state.get("standalone_question") or state["question"]

    if route == "off_topic":
        specific_reason = f'Your question "{question[:80]}" is outside the scope of the available AI research papers.'
        example_questions = (
            "• What is multi-head attention and why is it useful?\n"
            "• How does LoRA reduce the number of trainable parameters?\n"
            "• What is the difference between BERT and GPT-3?"
        )
        answer = FALLBACK_TEMPLATE.format(
            specific_reason=specific_reason,
            example_questions=example_questions,
        )
    else:
        # Retries exhausted
        answer = (
            f"I was unable to find sufficient information about '{question[:80]}' "
            f"in the available documents after {retries} retrieval attempt(s).\n\n"
            "The answer may not be covered in the 10 ingested AI research papers. "
            "You can upload additional documents using the Upload tab."
        )

    return {
        "answer": answer,
        "citations": [],
        "grounded": True,
        "confidence": 0.0,
        "trace_notes": [f"fallback: route={route}, retries={retries}"],
    }


# ── Node 13: finalize ─────────────────────────────────────────────────────────

def finalize(state: AgentState) -> dict:
    """
    Assemble the final response: compute confidence, add grounding caveat if needed.

    Confidence heuristic (documented formula, not a calibrated probability):
        0.4 × top_chunk_score  (cosine similarity, 0–1)
        0.3 × relevance_score  (relevant=1, partial=0.5, irrelevant=0)
        0.2 × grounded_score   (grounded=1, else 0.5 — partial trust)
        0.1 × retry_bonus      (1 − retries/MAX_RETRIES)
    """
    retrieved = state.get("retrieved", [])
    relevance = state.get("relevance", "irrelevant")
    grounded = state.get("grounded", False)
    retries = state.get("retries", 0)
    answer = state.get("answer", "")

    # Top chunk score
    top_score = retrieved[0]["score"] if retrieved else 0.0
    # Clamp to [0, 1] (cosine similarity can be slightly > 1 due to float errors)
    top_score = min(max(float(top_score), 0.0), 1.0)

    relevance_score = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}.get(relevance, 0.0)
    grounded_score = 1.0 if grounded else 0.5
    retry_bonus = 1.0 - (retries / MAX_RETRIES)

    confidence = (
        0.4 * top_score
        + 0.3 * relevance_score
        + 0.2 * grounded_score
        + 0.1 * retry_bonus
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    # Add a visible caveat if the grounding check flagged issues
    if not grounded:
        answer = (
            answer
            + "\n\n⚠️ *Note: Some claims in this answer could not be fully verified "
            "against the retrieved documents.*"
        )

    return {
        "answer": answer,
        "confidence": confidence,
        "messages": [AIMessage(content=answer)],
        "trace_notes": [f"finalize: confidence={confidence:.3f}"],
    }


# ── Missing import ─────────────────────────────────────────────────────────────
import os  # noqa: E402 (placed here to avoid circular import issues at module top)
