"""
tests/test_retrieval.py — Unit tests for hybrid retrieval and RRF.

Uses toy documents so no ChromaDB or model is needed.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


class TestRRFMerge:
    """Test the Reciprocal Rank Fusion merge function with toy data."""

    def _make_doc(self, content: str, source: str = "test.pdf") -> Document:
        return Document(page_content=content, metadata={"source": source, "page": 1})

    def test_rrf_prefers_doc_in_both_lists(self):
        """A document appearing in both ranked lists should rank higher than one in only one."""
        from rag.retrieval import _rrf_merge  # noqa: PLC0415

        shared_doc = self._make_doc("shared content about attention mechanisms")
        vec_only_doc = self._make_doc("vector only content about BERT")
        bm25_only_doc = self._make_doc("bm25 only content about GPT3")

        vec_docs = [shared_doc, vec_only_doc]
        bm25_docs = [shared_doc, bm25_only_doc]

        merged = _rrf_merge(vec_docs, bm25_docs, k=3)

        # The shared document should be first
        assert merged[0].page_content == shared_doc.page_content

    def test_rrf_deduplicates_by_content(self):
        """Same document in both lists should appear only once in output."""
        from rag.retrieval import _rrf_merge  # noqa: PLC0415

        doc = self._make_doc("duplicate content in both lists")
        merged = _rrf_merge([doc, doc], [doc, doc], k=5)

        # Should appear only once
        contents = [d.page_content for d in merged]
        assert contents.count(doc.page_content) == 1

    def test_rrf_respects_k_limit(self):
        """Output should contain at most k documents."""
        from rag.retrieval import _rrf_merge  # noqa: PLC0415

        docs = [self._make_doc(f"document {i}") for i in range(10)]
        merged = _rrf_merge(docs[:5], docs[5:], k=3)
        assert len(merged) <= 3

    def test_rrf_handles_empty_bm25(self):
        """If BM25 returns nothing, vector results should pass through."""
        from rag.retrieval import _rrf_merge  # noqa: PLC0415

        docs = [self._make_doc(f"vec doc {i}") for i in range(3)]
        merged = _rrf_merge(docs, [], k=3)
        assert len(merged) == 3


class TestDocsToDict:
    def test_converts_document_to_dict(self):
        from rag.retrieval import docs_to_dicts  # noqa: PLC0415

        doc = Document(
            page_content="test content",
            metadata={"source": "/path/to/file.pdf", "page": 2},
        )
        result = docs_to_dicts([doc])
        assert len(result) == 1
        assert result[0]["text"] == "test content"
        assert result[0]["source"] == "file.pdf"  # basename only
        assert result[0]["page"] == 2

    def test_deduplicates_by_content(self):
        from rag.retrieval import docs_to_dicts  # noqa: PLC0415

        doc = Document(page_content="same content", metadata={"source": "a.pdf", "page": 1})
        result = docs_to_dicts([doc, doc, doc])
        assert len(result) == 1
