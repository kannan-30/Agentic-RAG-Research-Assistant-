"""
rag/retrieval.py — Retrieval layer: vector search, BM25, and hybrid RRF fusion.

Implements:
  - Vector similarity search (ChromaDB cosine)
  - BM25 keyword search (rank_bm25)
  - Reciprocal Rank Fusion (RRF) to merge both ranked lists
  - Source filtering (restrict to one file)
  - Context formatting with source attribution
  - Full legacy ``rag_query`` pipeline (used by baseline and old endpoints)

Design decisions:
- BM25 index is built lazily from the Chroma collection and cached in a
  module-level variable.  It is invalidated (set to None) whenever
  ``reset_vectorstore`` is called after ingestion.
- ``RETRIEVAL_MODE`` env var controls whether hybrid or vector-only is used
  (default: hybrid).  This lets the eval sweep test both modes.
- Heavy objects (vectorstore, embedder) are singletons (see ingestion.py for
  the embedder singleton rationale).

RRF formula (from Cormack et al., 2009):
    score(d) = Σ  1 / (k + rank_i(d))   for each ranked list i
    k = 60 (constant that dampens the influence of very high ranks)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.ingestion import load_vector_store, CHROMA_PERSIST_DIR

load_dotenv()

logger = logging.getLogger(__name__)

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # "vector" | "hybrid"
RRF_K = 60  # Standard RRF constant

# ── Singletons ────────────────────────────────────────────────────────────────
_vectorstore: Chroma | None = None
_bm25_index: Any = None         # rank_bm25.BM25Okapi instance
_bm25_corpus: list[Document] = []  # parallel list to _bm25_index


# ── Vectorstore management ────────────────────────────────────────────────────

def get_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Return the cached vectorstore, loading it on first call."""
    global _vectorstore  # noqa: PLW0603
    if _vectorstore is None:
        logger.info("Loading vectorstore from %s", persist_dir)
        _vectorstore = load_vector_store(persist_dir)
    return _vectorstore


def reset_vectorstore() -> None:
    """
    Invalidate both the vectorstore and BM25 index singletons.

    Call this after ingestion so that the next query uses fresh data.
    """
    global _vectorstore, _bm25_index, _bm25_corpus  # noqa: PLW0603
    _vectorstore = None
    _bm25_index = None
    _bm25_corpus = []
    logger.info("Vectorstore and BM25 index singletons reset.")


# ── BM25 index ────────────────────────────────────────────────────────────────

def _get_bm25_index() -> tuple[Any, list[Document]]:
    """
    Build (or return cached) BM25 index over all ingested chunks.

    BM25 is built from the Chroma collection's stored texts.  We fetch all
    documents once at index-build time and cache them alongside the index.
    Building takes ~0.5 s for 1 000 chunks; subsequent calls return instantly.
    """
    global _bm25_index, _bm25_corpus  # noqa: PLW0603

    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus

    try:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415
    except ImportError:
        logger.warning("rank_bm25 not installed; BM25 unavailable. pip install rank-bm25")
        return None, []

    vs = get_vectorstore()
    try:
        raw = vs.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("Could not fetch documents for BM25 index: %s", exc)
        return None, []

    texts: list[str] = raw.get("documents") or []
    metadatas: list[dict] = raw.get("metadatas") or []

    if not texts:
        logger.warning("Vector store is empty; BM25 index not built.")
        return None, []

    # Simple whitespace tokenisation (fast, good enough for retrieval)
    tokenised = [t.lower().split() for t in texts]
    _bm25_index = BM25Okapi(tokenised)

    # Reconstruct LangChain Documents so we can return them in the same format.
    _bm25_corpus = [
        Document(page_content=texts[i], metadata=metadatas[i] if i < len(metadatas) else {})
        for i in range(len(texts))
    ]

    logger.info("BM25 index built over %d chunks.", len(texts))
    return _bm25_index, _bm25_corpus


# ── Core retrieval functions ───────────────────────────────────────────────────

def vector_retrieve(
    query: str,
    k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
) -> list[Document]:
    """
    Top-k cosine similarity search against ChromaDB.

    Args:
        query:         Query string (will be embedded).
        k:             Number of results.
        source_filter: If given, only return chunks whose ``source`` metadata
                       contains this string (case-insensitive filename match).
    """
    vs = get_vectorstore()
    where: dict | None = None
    if source_filter:
        # Chroma metadata filter: source field contains the filter string.
        where = {"source": {"$contains": source_filter}}

    try:
        if where:
            docs = vs.similarity_search(query, k=k, filter=where)
        else:
            docs = vs.similarity_search(query, k=k)
    except Exception as exc:
        logger.warning("vector_retrieve failed: %s", exc)
        docs = []

    return docs


def bm25_retrieve(
    query: str,
    k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
) -> list[Document]:
    """
    BM25 keyword search over the stored corpus.

    Args:
        query:         Query string (tokenised on whitespace).
        k:             Number of results.
        source_filter: Optional filename substring filter.
    """
    index, corpus = _get_bm25_index()
    if index is None or not corpus:
        return []

    tokens = query.lower().split()
    scores = index.get_scores(tokens)

    # Pair each doc with its BM25 score and sort descending.
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[Document] = []
    for idx, score in ranked:
        if len(results) >= k:
            break
        if score <= 0:
            break
        doc = corpus[idx]
        if source_filter:
            src = doc.metadata.get("source", "")
            if source_filter.lower() not in src.lower():
                continue
        results.append(doc)

    return results


def _rrf_merge(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    k: int = TOP_K_RESULTS,
) -> list[Document]:
    """
    Reciprocal Rank Fusion of two ranked document lists.

    RRF score for document d:
        score(d) = 1/(RRF_K + rank_vector(d)) + 1/(RRF_K + rank_bm25(d))

    Documents not present in a list receive a very high rank (len+1) so they
    are penalised but not excluded.  After merging, we de-duplicate by content
    hash and return the top-k by RRF score.

    Design note: we identify documents by their content text rather than any
    ID because the two retrieval paths do not share ID spaces.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content[:200]  # content prefix as dedup key
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


def hybrid_retrieve(
    query: str,
    k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
) -> list[Document]:
    """
    Hybrid retrieval: BM25 + vector search merged with RRF.

    Falls back to vector-only if BM25 is unavailable or if
    ``RETRIEVAL_MODE=vector`` is set.

    Args:
        query:         Natural-language query.
        k:             Final number of results.
        source_filter: Optional filename filter.
    """
    if RETRIEVAL_MODE == "vector":
        return vector_retrieve(query, k=k, source_filter=source_filter)

    vec_docs = vector_retrieve(query, k=k * 2, source_filter=source_filter)
    bm25_docs = bm25_retrieve(query, k=k * 2, source_filter=source_filter)

    if not bm25_docs:
        logger.debug("BM25 returned no results; using vector-only.")
        return vec_docs[:k]

    return _rrf_merge(vec_docs, bm25_docs, k=k)


def retrieve_chunks(
    query: str,
    k: int = TOP_K_RESULTS,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> list[Document]:
    """
    Legacy wrapper: returns top-k chunks for the baseline pipeline.

    Delegates to ``hybrid_retrieve`` so the baseline also benefits from
    improved retrieval while remaining backward-compatible.
    """
    return hybrid_retrieve(query, k=k)


# ── Context formatting ────────────────────────────────────────────────────────

def format_context(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a context string with source attribution.

    Each chunk is tagged with its source filename and page number so the LLM
    can produce inline citations.
    """
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        source_name = Path(source).name if source != "Unknown" else "Unknown"
        header = f"[Source {i}: {source_name}, Page {page}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def docs_to_dicts(docs: list[Document]) -> list[dict]:
    """
    Convert LangChain Documents to the state dict format used in AgentState.

    Returns a list of:
        {text, score, source, page, chunk_id}
    Score defaults to 1.0 for BM25-only results (no cosine score available).
    """
    result: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        key = doc.page_content[:200]
        if key in seen:
            continue
        seen.add(key)
        source = doc.metadata.get("source", "Unknown")
        result.append({
            "text": doc.page_content,
            "score": float(doc.metadata.get("score", 1.0)),
            "source": Path(source).name,
            "page": doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id", ""),
        })
    return result


# ── Legacy full RAG pipeline (kept for baseline comparisons) ──────────────────

def rag_query(
    question: str,
    k: int = TOP_K_RESULTS,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> dict[str, Any]:
    """
    Full legacy RAG pipeline:
      1. Retrieve top-k chunks (hybrid)
      2. Format context
      3. Generate answer via Gemini

    Returns:
        {"answer", "sources", "context", "retrieved_chunks"}

    This function is kept for backward compatibility with the ``/query``
    endpoint and the baseline evaluator.
    """
    from rag.llm import generate_answer  # noqa: PLC0415

    docs = hybrid_retrieve(question, k=k)

    if not docs:
        return {
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "context": "",
            "retrieved_chunks": 0,
        }

    context = format_context(docs)
    answer = generate_answer(question, context)

    sources: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        source_name = Path(source).name
        page = doc.metadata.get("page", "?")
        key = f"{source_name}::{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source_name,
                "page": page,
                "content": doc.page_content[:300] + "..."
                if len(doc.page_content) > 300
                else doc.page_content,
            })

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "retrieved_chunks": len(docs),
    }
