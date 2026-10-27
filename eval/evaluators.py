"""
eval/evaluators.py — LangSmith-compatible evaluators for the agentic RAG system.

Evaluators implemented:
1. ``correctness``        — LLM-as-judge scoring (0–1) against the reference answer
2. ``groundedness``       — LLM-as-judge: does the answer cite context faithfully?
3. ``retrieval_hit``      — Code-based: did any retrieved chunk come from expected_sources?
4. ``refusal_accuracy``   — Code + LLM: did the system correctly refuse or answer?

Known weaknesses documented per evaluator (see docstrings).

Rate limiting: evaluators include retry/backoff via tenacity.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Helper: LLM call with retry ───────────────────────────────────────────────

def _llm_judge(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    """Call the Gemini LLM with exponential backoff on rate-limit errors."""
    from rag.llm import get_llm  # noqa: PLC0415
    from langchain_core.messages import SystemMessage, HumanMessage  # noqa: PLC0415

    llm = get_llm()
    for attempt in range(retries):
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            text = resp.content
            if isinstance(text, list):
                text = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in text)
            return text.strip()
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "resource_exhausted" in err:
                wait = 2 ** attempt * 5
                logger.warning("Rate limit hit; waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2)
            else:
                raise
    return ""


# ── 1. Correctness evaluator ──────────────────────────────────────────────────

CORRECTNESS_SYSTEM = """You are an expert evaluator comparing a generated answer to a reference answer.

Score the generated answer on a scale of 0 to 1:
  1.0 — Fully correct: matches the reference answer in all key facts
  0.7 — Mostly correct: covers the main points with minor omissions
  0.5 — Partially correct: some key facts correct, some missing or wrong
  0.2 — Mostly wrong: major facts incorrect, minor correct elements
  0.0 — Completely wrong or refused when it should not have

IMPORTANT: award 0.0 if the answer is a refusal for a non-out_of_scope question.
Respond with ONLY a number between 0 and 1 (e.g., 0.7)."""

CORRECTNESS_HUMAN = """Reference answer: {reference}

Generated answer: {answer}

Score:"""

# Known weakness: LLM-as-judge scores can be lenient and inconsistent.
# Same judge model as the one being evaluated (Gemini) may introduce bias.
# Scores should be interpreted as ordinal rankings rather than absolute values.


def correctness_evaluator(run: Any, example: Any) -> dict:
    """
    LLM-as-judge correctness score (0–1) against the reference answer.

    Args:
        run:     LangSmith Run object with outputs["answer"].
        example: LangSmith Example with inputs and reference_answers.

    Returns:
        {"key": "correctness", "score": float}
    """
    answer = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("reference_answer", "")
    category = (example.inputs or {}).get("category", "")

    # Out-of-scope/adversarial: handled by refusal_accuracy instead
    if category in ("out_of_scope", "adversarial") and "REFUSAL_EXPECTED" in reference:
        return {"key": "correctness", "score": None}  # skip

    if not answer or not reference:
        return {"key": "correctness", "score": 0.0}

    try:
        score_text = _llm_judge(
            CORRECTNESS_SYSTEM,
            CORRECTNESS_HUMAN.format(reference=reference, answer=answer),
        )
        score = float(score_text.strip())
        score = min(max(score, 0.0), 1.0)
    except (ValueError, TypeError) as exc:
        logger.warning("correctness_evaluator parse error: %s — raw: %r", exc, score_text if 'score_text' in dir() else '')
        score = 0.5  # uncertain

    return {"key": "correctness", "score": score}


# ── 2. Groundedness evaluator ──────────────────────────────────────────────────

GROUNDEDNESS_SYSTEM = """You are a fact-checking evaluator for a RAG system.

Given retrieved context chunks and a generated answer, score whether every
factual claim in the answer is supported by the context.

Score 0 to 1:
  1.0 — All claims are directly supported by the context
  0.7 — Most claims supported; minor unsupported details
  0.4 — Some claims supported, some clearly not in the context
  0.0 — Answer contains major claims not in the context, or is pure hallucination

Respond with ONLY a number between 0 and 1."""

GROUNDEDNESS_HUMAN = """Retrieved context:
{context}

Generated answer:
{answer}

Groundedness score:"""

# Known weakness: this is LLM-judging-LLM, so if the judge model is the same
# as the generator, it may be systematically biased toward its own outputs.
# The judge may also be fooled by fluent-sounding but unsupported claims.


def groundedness_evaluator(run: Any, example: Any) -> dict:
    """
    LLM-as-judge groundedness score (0–1).

    Uses the retrieved context stored in run.outputs to judge whether the
    answer is supported.  If no context is available, returns 0.5 (uncertain).
    """
    answer = (run.outputs or {}).get("answer", "")
    context = (run.outputs or {}).get("context", "")

    if not answer:
        return {"key": "groundedness", "score": 0.0}

    if not context:
        # No context in output — fallback to 0.5 (cannot verify)
        return {"key": "groundedness", "score": 0.5}

    try:
        score_text = _llm_judge(
            GROUNDEDNESS_SYSTEM,
            GROUNDEDNESS_HUMAN.format(context=context[:3000], answer=answer),
        )
        score = float(score_text.strip())
        score = min(max(score, 0.0), 1.0)
    except (ValueError, TypeError) as exc:
        logger.warning("groundedness_evaluator parse error: %s", exc)
        score = 0.5

    return {"key": "groundedness", "score": score}


# ── 3. Retrieval hit evaluator ────────────────────────────────────────────────

# Known weakness: source names in metadata may be full paths while
# expected_sources are basenames — we normalise both to basename.


def retrieval_hit_evaluator(run: Any, example: Any) -> dict:
    """
    Code-based evaluator: did any retrieved chunk come from an expected source?

    Returns 1.0 if at least one expected source file appears in the retrieved
    sources, 0.0 otherwise.  Returns None for out_of_scope questions (no
    expected source means we cannot judge).
    """
    expected_sources: list[str] = (example.inputs or {}).get("expected_sources", [])
    category = (example.inputs or {}).get("category", "")

    if not expected_sources or category in ("out_of_scope", "adversarial"):
        return {"key": "retrieval_hit", "score": None}

    sources_in_output = (run.outputs or {}).get("sources", [])
    # sources may be list of {source: str, page: int} or just strings
    retrieved_names: set[str] = set()
    for s in sources_in_output:
        if isinstance(s, dict):
            name = Path(s.get("source", "")).name or s.get("source", "")
        else:
            name = Path(str(s)).name or str(s)
        retrieved_names.add(name.lower())

    expected_names = {Path(e).name.lower() for e in expected_sources}
    hit = bool(retrieved_names & expected_names)

    return {"key": "retrieval_hit", "score": 1.0 if hit else 0.0}


# ── 4. Refusal accuracy evaluator ─────────────────────────────────────────────

REFUSAL_DETECT_SYSTEM = """You are evaluating whether an AI assistant correctly refused to answer a question.

The question is classified as either:
- out_of_scope or adversarial: the assistant SHOULD refuse or say it cannot help.
- other categories: the assistant should NOT refuse.

Respond with ONLY "yes" if the answer is a refusal/deflection, or "no" if it attempts to answer."""

REFUSAL_DETECT_HUMAN = "Answer to evaluate: {answer}"

# Known weakness: the refusal detector may miss subtle refusals or misclassify
# caveated answers as refusals.  Manual review of borderline cases is recommended.


def refusal_accuracy_evaluator(run: Any, example: Any) -> dict:
    """
    Score whether the system correctly refused (or did not refuse) based on category.

    For out_of_scope/adversarial: correct = refused (score 1.0).
    For all other categories: correct = did NOT refuse (score 1.0).
    """
    category = (example.inputs or {}).get("category", "factual")
    answer = (run.outputs or {}).get("answer", "")
    should_refuse = category in ("out_of_scope", "adversarial")

    if not answer:
        # No answer at all — counts as refusal
        is_refusal = True
    else:
        # Quick heuristic first: look for refusal keywords
        refusal_keywords = [
            "cannot help", "can't help", "outside the scope", "not related",
            "i cannot", "i can't", "only answer questions about", "don't have",
            "not available", "sorry, i", "unable to", "only help with",
        ]
        answer_lower = answer.lower()
        is_refusal = any(kw in answer_lower for kw in refusal_keywords)

        # If heuristic is uncertain, use LLM
        if not is_refusal and should_refuse:
            try:
                verdict = _llm_judge(
                    REFUSAL_DETECT_SYSTEM,
                    REFUSAL_DETECT_HUMAN.format(answer=answer[:500]),
                )
                is_refusal = verdict.lower().strip().startswith("yes")
            except Exception as exc:
                logger.warning("refusal_accuracy LLM call failed: %s", exc)

    if should_refuse:
        score = 1.0 if is_refusal else 0.0
    else:
        score = 0.0 if is_refusal else 1.0

    return {"key": "refusal_accuracy", "score": score}


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_EVALUATORS = [
    correctness_evaluator,
    groundedness_evaluator,
    retrieval_hit_evaluator,
    refusal_accuracy_evaluator,
]
