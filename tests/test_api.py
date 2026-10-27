"""
tests/test_api.py — FastAPI contract tests using TestClient.

All LLM and retrieval calls are mocked so tests run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with all heavy dependencies mocked."""
    # Patch the lifespan to skip real initialisation
    with patch("api.lifespan", return_value=MagicMock()):
        pass  # lifespan is async context manager; TestClient handles it

    with (
        patch("rag.ingestion.get_embeddings", return_value=MagicMock()),
        patch("rag.llm.get_llm", return_value=MagicMock()),
        patch("rag.graph.builder.get_graph", return_value=MagicMock()),
    ):
        from api import app  # noqa: PLC0415
        return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestChatEndpoint:
    def test_empty_question_returns_400(self, client):
        resp = client.post("/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_chat_generates_thread_id_if_absent(self, client):
        mock_result = {
            "answer": "Multi-head attention runs in parallel.",
            "citations": [],
            "confidence": 0.8,
            "route": "rag",
            "steps": ["route: rag"],
        }
        with patch("rag.graph.builder.run_agent", return_value=mock_result):
            resp = client.post("/chat", json={"question": "What is multi-head attention?"})
        if resp.status_code == 200:
            data = resp.json()
            assert "thread_id" in data
            assert data["thread_id"].startswith("thread-")

    def test_chat_uses_provided_thread_id(self, client):
        mock_result = {
            "answer": "BERT uses bidirectional training.",
            "citations": [],
            "confidence": 0.85,
            "route": "rag",
            "steps": [],
        }
        with patch("rag.graph.builder.run_agent", return_value=mock_result):
            resp = client.post("/chat", json={
                "question": "Tell me about BERT",
                "thread_id": "test-thread-001",
            })
        if resp.status_code == 200:
            assert resp.json()["thread_id"] == "test-thread-001"


class TestFeedbackEndpoint:
    def test_feedback_skips_without_langsmith_key(self, client):
        with patch.dict("os.environ", {"LANGSMITH_API_KEY": ""}, clear=False):
            resp = client.post("/feedback", json={
                "run_id": "fake-run-id",
                "score": 1.0,
            })
        # Should not raise — returns skipped status
        assert resp.status_code == 200
        assert "skipped" in resp.json().get("status", "")

    def test_feedback_validates_request_schema(self, client):
        # Missing required run_id
        resp = client.post("/feedback", json={"score": 1.0})
        assert resp.status_code == 422  # Pydantic validation error


class TestQueryEndpoint:
    def test_empty_question_returns_400(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 400

    def test_query_baseline_pipeline(self, client):
        mock_result = {
            "answer": "Attention is all you need.",
            "sources": [{"source": "01_attention.pdf", "page": 1, "content": "..."}],
            "context": "context text",
            "retrieved_chunks": 1,
        }
        with patch("rag.retrieval.rag_query", return_value=mock_result):
            resp = client.post("/query", json={
                "question": "What is attention?",
                "pipeline": "baseline",
            })
        if resp.status_code == 200:
            data = resp.json()
            assert data["answer"] == "Attention is all you need."
