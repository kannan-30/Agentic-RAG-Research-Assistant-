"""
tests/test_routing.py — Unit tests for graph routing logic.

All tests use mocked LLMs and retrievers so they run offline and fast.
No API keys needed.
"""

from __future__ import annotations

import pytest

# ── Test edge routing functions (no LLM needed) ───────────────────────────────

class TestRouteAfterClassify:
    def test_off_topic_goes_to_fallback(self):
        from rag.graph.edges import route_after_classify
        state = {"route": "off_topic"}
        assert route_after_classify(state) == "fallback"

    def test_compare_goes_to_decompose(self):
        from rag.graph.edges import route_after_classify
        state = {"route": "compare"}
        assert route_after_classify(state) == "decompose"

    def test_tool_goes_to_tool_agent(self):
        from rag.graph.edges import route_after_classify
        state = {"route": "tool"}
        assert route_after_classify(state) == "tool_agent"

    def test_rag_goes_to_retrieve(self):
        from rag.graph.edges import route_after_classify
        state = {"route": "rag"}
        assert route_after_classify(state) == "retrieve"


class TestRouteAfterGrade:
    def test_irrelevant_with_retries_available_rewrites(self):
        from rag.graph.edges import route_after_grade
        state = {"relevance": "irrelevant", "retries": 0}
        assert route_after_grade(state) == "rewrite_query"

    def test_irrelevant_retries_exhausted_falls_back(self):
        from rag.graph.edges import route_after_grade
        from rag.graph.nodes import MAX_RETRIES
        state = {"relevance": "irrelevant", "retries": MAX_RETRIES}
        assert route_after_grade(state) == "fallback"

    def test_relevant_goes_to_generate(self):
        from rag.graph.edges import route_after_grade
        state = {"relevance": "relevant", "retries": 0}
        assert route_after_grade(state) == "generate"

    def test_partial_goes_to_generate(self):
        from rag.graph.edges import route_after_grade
        state = {"relevance": "partial", "retries": 0}
        assert route_after_grade(state) == "generate"

    def test_retry_cap_is_exactly_max_retries(self):
        """Retry cap must be exactly MAX_RETRIES, not MAX_RETRIES - 1."""
        from rag.graph.edges import route_after_grade
        from rag.graph.nodes import MAX_RETRIES
        # One below the cap: should still rewrite
        state_below = {"relevance": "irrelevant", "retries": MAX_RETRIES - 1}
        assert route_after_grade(state_below) == "rewrite_query"
        # At the cap: should fallback
        state_at = {"relevance": "irrelevant", "retries": MAX_RETRIES}
        assert route_after_grade(state_at) == "fallback"


class TestRouteAfterGrounding:
    def test_not_grounded_first_attempt_regenerates(self):
        from rag.graph.edges import route_after_grounding
        state = {"grounded": False, "regenerations": 0}
        assert route_after_grounding(state) == "regenerate"

    def test_not_grounded_after_max_regenerations_finalizes(self):
        from rag.graph.edges import route_after_grounding
        from rag.graph.nodes import MAX_REGENERATIONS
        state = {"grounded": False, "regenerations": MAX_REGENERATIONS}
        assert route_after_grounding(state) == "finalize"

    def test_grounded_finalizes(self):
        from rag.graph.edges import route_after_grounding
        state = {"grounded": True, "regenerations": 0}
        assert route_after_grounding(state) == "finalize"
