"""
api.py — FastAPI backend for the Agentic RAG Q&A Chatbot.

Endpoints (all original endpoints preserved; new ones added):
  GET  /health              → Health check
  GET  /stats               → ChromaDB collection stats
  POST /ingest              → Ingest PDFs from data/pdfs/ (legacy)
  POST /ingest/upload       → Upload + ingest PDF files (idempotent)
  POST /query               → Legacy RAG query (routes to agent or baseline via ?pipeline=)
  POST /evaluate            → Run RAGAs evaluation on sample questions
  POST /chat                → Agentic RAG (with thread_id, citations, steps)
  POST /chat/stream         → SSE streaming version of /chat
  POST /feedback            → Log thumbs up/down to LangSmith
  GET  /threads/{thread_id} → Conversation history for a thread

Design decisions:
- Heavy singletons (embedder, vectorstore, LLM, graph) are initialized via the
  FastAPI lifespan context manager, not per-request.
- /ingest/upload is idempotent: it hashes file content and skips already-ingested files.
- /chat/stream uses StreamingResponse with SSE (text/event-stream) so the
  Gradio and any frontend can show incremental node-level progress.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
COLLECTION_NAME = "rag_documents"


# ── Startup: initialise all singletons once ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared resources at startup so they are ready before the first
    request.  Using the lifespan pattern (FastAPI 0.95+) rather than
    on_event('startup') which is deprecated.
    """
    logger.info("Initializing shared resources...")
    try:
        from rag.ingestion import get_embeddings  # noqa: PLC0415
        get_embeddings()          # warm up the 90 MB MiniLM model
        logger.info("Embeddings model loaded.")
    except Exception as exc:
        logger.warning("Embeddings warm-up failed: %s", exc)

    try:
        from rag.llm import get_llm  # noqa: PLC0415
        get_llm()
        logger.info("LLM client initialized.")
    except Exception as exc:
        logger.warning("LLM init failed (GOOGLE_API_KEY may be missing): %s", exc)

    try:
        from rag.graph.builder import get_graph  # noqa: PLC0415
        get_graph()
        logger.info("LangGraph agent compiled and ready.")
    except Exception as exc:
        logger.warning("Agent graph init failed: %s", exc)

    yield  # Application is running

    logger.info("Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Document Q&A API",
    description=(
        "LangGraph-powered Retrieval-Augmented Generation chatbot with "
        "routing, corrective retrieval, grounding checks, and tool use. "
        "Backed by ChromaDB, hybrid BM25+vector search, and Gemini."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    pipeline: Optional[str] = "agent"  # "agent" | "baseline"


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    retrieved_chunks: int


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]
    confidence: float
    route: str
    steps: List[str]
    thread_id: str
    pipeline_version: str = "agent"


class FeedbackRequest(BaseModel):
    run_id: str
    score: float           # 0.0 (thumbs down) or 1.0 (thumbs up)
    comment: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    pdf_files: Optional[int] = None
    pages_loaded: Optional[int] = None
    chunks_created: Optional[int] = None
    added: Optional[int] = None
    skipped_duplicates: Optional[int] = None
    message: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic RAG Q&A API", "version": "2.0.0"}


@app.get("/stats")
def collection_stats():
    """Return ChromaDB collection statistics."""
    from rag.ingestion import get_collection_stats  # noqa: PLC0415
    return get_collection_stats()


@app.post("/ingest", response_model=IngestResponse)
def ingest(pdf_dir: Optional[str] = None):
    """Trigger the PDF ingestion pipeline from the configured directory."""
    from rag.ingestion import ingest_documents  # noqa: PLC0415
    from rag.retrieval import reset_vectorstore  # noqa: PLC0415
    target_dir = pdf_dir or PDF_DIR
    result = ingest_documents(pdf_dir=target_dir)
    if result["status"] == "success":
        reset_vectorstore()
    return result


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_uploaded_files(files: List[UploadFile] = File(...)):
    """
    Upload PDF files and ingest them into the vector store.

    Idempotent: files whose content hash already exists in the collection
    are skipped.  Returns counts of added vs. skipped files.
    """
    import hashlib  # noqa: PLC0415
    from rag.ingestion import ingest_documents, CHROMA_PERSIST_DIR, COLLECTION_NAME as CNAME  # noqa: PLC0415
    from rag.retrieval import reset_vectorstore  # noqa: PLC0415
    import chromadb  # noqa: PLC0415

    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Load existing content hashes from ChromaDB manifest metadata
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(CNAME)
        existing_meta = collection.get(include=["metadatas"])
        existing_hashes = {
            m.get("content_hash", "")
            for m in (existing_meta.get("metadatas") or [])
            if m.get("content_hash")
        }
    except Exception:
        existing_hashes = set()

    saved_files: list[str] = []
    skipped: list[str] = []

    for file in files:
        if not (file.filename or "").endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        content = await file.read()
        content_hash = hashlib.sha256(content).hexdigest()

        if content_hash in existing_hashes:
            skipped.append(file.filename)
            continue

        dest = pdf_dir / file.filename
        with open(dest, "wb") as f:
            f.write(content)
        saved_files.append(file.filename)

    if not saved_files:
        return {
            "status": "success",
            "added": 0,
            "skipped_duplicates": len(skipped),
            "message": f"All {len(skipped)} file(s) already ingested.",
        }

    result = ingest_documents(pdf_dir=str(pdf_dir))
    if result["status"] == "success":
        reset_vectorstore()

    result["added"] = len(saved_files)
    result["skipped_duplicates"] = len(skipped)
    return result


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    RAG query endpoint (legacy + pipeline toggle).

    Use ?pipeline=baseline to compare against the original linear pipeline.
    Defaults to the new LangGraph agent.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.pipeline == "baseline":
        from rag.retrieval import rag_query  # noqa: PLC0415
        result = rag_query(request.question, k=request.top_k)
    else:
        # Route through agent, present in legacy format
        thread_id = f"query-{uuid.uuid4().hex[:8]}"
        from rag.graph.builder import run_agent  # noqa: PLC0415
        agent_result = run_agent(question=request.question, thread_id=thread_id)
        citations = agent_result.get("citations", [])
        result = {
            "answer": agent_result.get("answer", ""),
            "sources": [{"source": c["source"], "page": c["page"], "content": ""} for c in citations],
            "retrieved_chunks": len(citations),
        }

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "retrieved_chunks": result.get("retrieved_chunks", 0),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Agentic RAG chat endpoint with full LangGraph pipeline.

    Returns answer, inline citations, confidence score, route taken,
    step trace, and thread_id for follow-up questions.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex}"

    from rag.graph.builder import run_agent  # noqa: PLC0415
    try:
        result = run_agent(question=request.question, thread_id=thread_id)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        confidence=result.get("confidence", 0.0),
        route=result.get("route", "rag"),
        steps=result.get("steps", []),
        thread_id=thread_id,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat using Server-Sent Events (SSE).

    Each node completion sends an event so the frontend can show live progress:
      data: {"type": "step", "content": "route: rag — single-paper lookup"}
      data: {"type": "step", "content": "retrieved: 5 chunks via hybrid search"}
      data: {"type": "answer", "content": "The attention mechanism..."}
      data: {"type": "done", "thread_id": "..."}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex}"

    async def event_generator() -> AsyncGenerator[str, None]:
        from rag.graph.builder import get_graph  # noqa: PLC0415
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        initial_state = {
            "question": request.question,
            "messages": [HumanMessage(content=request.question)],
            "retries": 0,
            "regenerations": 0,
            "trace_notes": [],
            "retrieved": [],
            "sub_questions": [],
            "citations": [],
        }

        try:
            for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    notes = node_output.get("trace_notes", [])
                    for note in notes:
                        event = json.dumps({"type": "step", "node": node_name, "content": note})
                        yield f"data: {event}\n\n"

                    # Emit the answer as soon as it is generated
                    if "answer" in node_output and node_output["answer"]:
                        event = json.dumps({"type": "answer", "content": node_output["answer"]})
                        yield f"data: {event}\n\n"

        except Exception as exc:
            event = json.dumps({"type": "error", "content": str(exc)})
            yield f"data: {event}\n\n"

        # Final done event
        event = json.dumps({"type": "done", "thread_id": thread_id})
        yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """
    Log thumbs-up / thumbs-down feedback to LangSmith.

    Args:
        run_id: The LangSmith run ID returned by /chat.
        score:  1.0 = positive, 0.0 = negative.
        comment: Optional free-text comment.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key == "your_langsmith_api_key_here":
        return {"status": "skipped", "reason": "LANGSMITH_API_KEY not configured"}

    try:
        from langsmith import Client  # noqa: PLC0415
        client = Client()
        client.create_feedback(
            run_id=request.run_id,
            key="user_feedback",
            score=request.score,
            comment=request.comment,
        )
        return {"status": "ok", "run_id": request.run_id, "score": request.score}
    except Exception as exc:
        logger.warning("Feedback logging failed: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/threads/{thread_id}")
def get_thread_history(thread_id: str):
    """Return conversation history for a given thread_id."""
    try:
        from rag.graph.builder import get_graph  # noqa: PLC0415
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id!r} not found.")
        messages = state.values.get("messages", [])
        return {
            "thread_id": thread_id,
            "messages": [
                {"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
                 "content": m.content}
                for m in messages
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("get_thread_history error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate")
def evaluate():
    """
    Run RAGAs evaluation on sample questions.

    Computes faithfulness and answer_relevancy scores using the existing pipeline.
    This may take 2–5 minutes depending on the number of questions.
    """
    from rag.evaluation import evaluate_with_sample_questions  # noqa: PLC0415
    from rag.retrieval import rag_query  # noqa: PLC0415
    return evaluate_with_sample_questions(rag_query)
