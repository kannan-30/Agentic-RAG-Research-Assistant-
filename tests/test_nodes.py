"""
tests/test_nodes.py — Unit tests for graph nodes with mocked LLMs.

Tests verify node behaviour without any API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFinalizeConfidence:
    """Test the confidence heuristic in the finalize node."""

    def test_high_confidence_all_good(self):
        from rag.graph.nodes import finalize  # noqa: PLC0415

        state = {
            "retrieved": [{"score": 0.9, "source": "a.pdf", "page": 1, "text": "t", "chunk_id": ""}],
            "relevance": "relevant",
            "grounded": True,
            "retries": 0,
            "answer": "Test answer.",
            "citations": [],
            "messages": [],
            "trace_notes": [],
        }
        result = finalize(state)
        # 0.4*0.9 + 0.3*1.0 + 0.2*1.0 + 0.1*1.0 = 0.36 + 0.30 + 0.20 + 0.10 = 0.96
        assert result["confidence"] == pytest.approx(0.96, abs=0.01)

    def test_low_confidence_irrelevant_ungrounded(self):
        from rag.graph.nodes import finalize  # noqa: PLC0415

        state = {
            "retrieved": [{"score": 0.1, "source": "a.pdf", "page": 1, "text": "t", "chunk_id": ""}],
            "relevance": "irrelevant",
            "grounded": False,
            "retries": 2,
            "answer": "Test answer.",
            "citations": [],
            "messages": [],
            "trace_notes": [],
        }
        result = finalize(state)
        # 0.4*0.1 + 0.3*0.0 + 0.2*0.5 + 0.1*0.0 = 0.04 + 0 + 0.10 + 0 = 0.14
        assert result["confidence"] == pytest.approx(0.14, abs=0.01)

    def test_confidence_clamped_to_0_1(self):
        from rag.graph.nodes import finalize  # noqa: PLC0415

        state = {
            "retrieved": [{"score": 2.0, "source": "a.pdf", "page": 1, "text": "t", "chunk_id": ""}],  # over 1.0
            "relevance": "relevant",
            "grounded": True,
            "retries": 0,
            "answer": "Test.",
            "citations": [],
            "messages": [],
            "trace_notes": [],
        }
        result = finalize(state)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_ungrounded_answer_gets_caveat(self):
        from rag.graph.nodes import finalize  # noqa: PLC0415

        state = {
            "retrieved": [{"score": 0.5, "source": "a.pdf", "page": 1, "text": "t", "chunk_id": ""}],
            "relevance": "relevant",
            "grounded": False,
            "retries": 0,
            "answer": "Some answer.",
            "citations": [],
            "messages": [],
            "trace_notes": [],
        }
        result = finalize(state)
        assert "⚠️" in result["answer"] or "Note" in result["answer"]


import pytest  # noqa: E402 (imported after class definitions for clarity)


class TestFallbackNode:
    def test_off_topic_fallback_mentions_papers(self):
        from rag.graph.nodes import fallback  # noqa: PLC0415

        state = {
            "route": "off_topic",
            "retries": 0,
            "question": "Who won the World Cup?",
            "standalone_question": "Who won the World Cup?",
            "trace_notes": [],
        }
        result = fallback(state)
        answer = result["answer"]
        # Should mention the corpus topics
        assert any(word in answer.lower() for word in ["attention", "bert", "rag", "research", "scope"])

    def test_not_found_fallback_mentions_retry_count(self):
        from rag.graph.nodes import fallback  # noqa: PLC0415

        state = {
            "route": "rag",
            "retries": 2,
            "question": "What is quantum entanglement?",
            "standalone_question": "What is quantum entanglement?",
            "trace_notes": [],
        }
        result = fallback(state)
        assert "2" in result["answer"]  # retry count mentioned


class TestRewriteWithHistory:
    def test_no_history_passes_through(self):
        from rag.graph.nodes import rewrite_with_history  # noqa: PLC0415
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        state = {
            "question": "What is BERT?",
            "messages": [HumanMessage(content="What is BERT?")],
            "trace_notes": [],
        }
        result = rewrite_with_history(state)
        assert result["standalone_question"] == "What is BERT?"

    def test_no_history_skips_llm_call(self):
        """With no prior history, no LLM call should be made."""
        from rag.graph.nodes import rewrite_with_history  # noqa: PLC0415
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        with patch("rag.graph.nodes._call_llm_text") as mock_llm:
            state = {
                "question": "What is BERT?",
                "messages": [HumanMessage(content="What is BERT?")],
                "trace_notes": [],
            }
            rewrite_with_history(state)
            mock_llm.assert_not_called()
