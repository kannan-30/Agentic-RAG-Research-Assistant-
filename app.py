"""
app.py — Gradio UI for the RAG Document Q&A Chatbot.

This is the main entry point for:
  - Local development: python app.py
  - HuggingFace Spaces deployment (sdk: gradio, app_file: app.py)

Features:
  - Premium dark-themed chat interface
  - Source document attribution panel
  - PDF upload & ingestion tab
  - RAGAs evaluation dashboard
  - Live ChromaDB stats
"""

import os
import time
import json
import uuid
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)

# ── Lazy imports so the UI loads even if ingestion hasn't happened ─────────────
def _get_rag_query():
    from rag.retrieval import rag_query
    return rag_query

def _get_agent_runner():
    from rag.graph.builder import run_agent
    return run_agent

def _get_ingest():
    from rag.ingestion import ingest_documents, get_collection_stats
    return ingest_documents, get_collection_stats

def _get_evaluation():
    from rag.evaluation import evaluate_with_sample_questions
    return evaluate_with_sample_questions


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep: #060610;
    --bg-primary: #0c0c1a;
    --bg-card: #12122a;
    --bg-elevated: #1a1a35;
    --bg-input: #16163a;
    --accent-purple: #7c3aed;
    --accent-violet: #8b5cf6;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;
    --gradient-primary: linear-gradient(135deg, #7c3aed, #3b82f6);
    --gradient-glow: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(59,130,246,0.3));
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: rgba(124, 58, 237, 0.25);
    --border-subtle: rgba(255,255,255,0.06);
    --shadow-glow: 0 0 40px rgba(124,58,237,0.15);
    --radius: 12px;
    --radius-lg: 18px;
}

/* ── Global reset ── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* ── Header ── */
#header-banner {
    background: linear-gradient(135deg, #0d0d2b 0%, #1a0a3d 50%, #0d1a3d 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow);
}

#header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(124,58,237,0.12) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(59,130,246,0.08) 0%, transparent 50%);
    animation: shimmer 8s ease-in-out infinite alternate;
}

@keyframes shimmer {
    0%   { transform: translate(-5%, -5%); }
    100% { transform: translate(5%, 5%);  }
}

#app-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: var(--gradient-primary) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}

#app-subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    margin-top: 6px !important;
    position: relative;
    z-index: 1;
}

/* ── Badges ── */
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; position: relative; z-index: 1; }
.badge {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border-color: rgba(124,58,237,0.4); }
.badge-blue   { background: rgba(59,130,246,0.15);  color: #93c5fd; border-color: rgba(59,130,246,0.4); }
.badge-cyan   { background: rgba(6,182,212,0.15);   color: #67e8f9; border-color: rgba(6,182,212,0.4); }
.badge-pink   { background: rgba(236,72,153,0.15);  color: #f9a8d4; border-color: rgba(236,72,153,0.4); }
.badge-green  { background: rgba(16,185,129,0.15);  color: #6ee7b7; border-color: rgba(16,185,129,0.4); }

/* ── Tabs ── */
.tab-nav {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    margin-bottom: 16px !important;
    gap: 4px !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.tab-nav button.selected, .tab-nav button:hover {
    background: var(--gradient-primary) !important;
    color: white !important;
}

/* ── Cards / Panels ── */
.panel-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
}

/* ── Chatbot ── */
#chatbot {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

#chatbot .message {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
}

#chatbot .user { 
    background: var(--gradient-primary) !important; 
    color: white !important;
    margin-left: 60px !important;
}

#chatbot .bot { 
    background: var(--bg-elevated) !important; 
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    margin-right: 60px !important;
}

/* ── Inputs ── */
textarea, input[type="text"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    transition: border-color 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
    outline: none !important;
}

/* ── Buttons ── */
button.primary {
    background: var(--gradient-primary) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-size: 0.93rem !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
}

button.secondary {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    padding: 9px 20px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover {
    border-color: var(--accent-purple) !important;
    color: var(--text-primary) !important;
}

/* ── Accordions ── */
.gr-accordion {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
}

/* ── Stats boxes ── */
.stat-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Score circles ── */
.score-circle {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: conic-gradient(var(--accent-purple) 0%, transparent 0%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 8px;
    position: relative;
}
.score-circle::before {
    content: '';
    position: absolute;
    inset: 8px;
    background: var(--bg-card);
    border-radius: 50%;
}
.score-inner {
    position: relative;
    z-index: 1;
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text-primary);
}

/* ── Source snippets ── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #a78bfa;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Upload area ── */
.upload-area {
    border: 2px dashed rgba(124,58,237,0.4) !important;
    background: rgba(124,58,237,0.04) !important;
    border-radius: var(--radius-lg) !important;
    transition: all 0.2s ease !important;
}
.upload-area:hover {
    border-color: var(--accent-purple) !important;
    background: rgba(124,58,237,0.08) !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 99px; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease; }
"""


# ── Helper functions ──────────────────────────────────────────────────────────
def chat_fn(
    message: str,
    history: List,
    pipeline: str = "LangGraph Agent (Flagship)",
    thread_id: str | None = None,
) -> Tuple[List, str, str, str]:
    """
    Process a chat message through the LangGraph Agent or Baseline pipeline.
    Returns updated history, sources HTML with execution trace, status, and thread_id.
    """
    if not message.strip():
        return history, "", "⚠️ Please enter a question.", thread_id or ""

    if not thread_id:
        thread_id = f"ui-{uuid.uuid4().hex[:8]}"

    try:
        t0 = time.perf_counter()
        if pipeline == "LangGraph Agent (Flagship)":
            run_agent = _get_agent_runner()
            result = run_agent(question=message, thread_id=thread_id)
            latency = time.perf_counter() - t0

            answer = result["answer"]
            route = result.get("route", "rag")
            confidence = result.get("confidence", 0.0)
            steps = result.get("steps", [])
            citations = result.get("citations", [])
            retrieved = result.get("retrieved", [])

            # Route badge color & label
            route_colors = {
                "rag": ("#8b5cf6", "🔍 Standard RAG"),
                "compare": ("#3b82f6", "🔀 Query Decomposition"),
                "tool": ("#10b981", "🛠️ Tool Execution"),
                "off_topic": ("#ef4444", "🛡️ Out-of-Scope Refusal"),
            }
            rc, r_label = route_colors.get(route, ("#8b5cf6", route.upper()))

            # Parse steps into readable chips with icons
            step_icons = {
                "rewrite": "⚡",
                "route": "🔀",
                "decompose": "🧩",
                "retrieve": "🔍",
                "grade": "⚖️",
                "rewrite_query": "🔄",
                "generated": "✍️",
                "grounding": "🛡️",
                "regenerated": "🩹",
                "tool": "🛠️",
                "fallback": "🚫",
                "confidence": "🎯",
            }
            step_items = []
            for s in steps:
                icon = "•"
                for k, ic in step_icons.items():
                    if k in s.lower():
                        icon = ic
                        break
                step_items.append(
                    f"<div style='display:flex;align-items:flex-start;gap:8px;font-size:0.77rem;color:#cbd5e1;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.03);'>"
                    f"<span style='font-size:0.85rem'>{icon}</span><span>{s}</span></div>"
                )
            steps_html = "".join(step_items) if step_items else "<p style='color:#64748b;font-size:0.75rem'>No steps logged</p>"

            # Citations HTML
            cites_html = ""
            if citations:
                cites_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;'>"
                for c in citations:
                    cites_html += (
                        f"<span class='source-chip' style='background:rgba(139,92,246,0.15);border-color:rgba(139,92,246,0.4);'>"
                        f"📄 {c.get('source')} <strong style='color:#c084fc'>p.{c.get('page')}</strong></span>"
                    )
                cites_html += "</div>"
            else:
                cites_html = "<p style='color:#64748b;font-size:0.75rem;margin:4px 0'>No inline citations (out-of-scope or general response).</p>"

            # Retrieved chunks preview
            retrieved_html = ""
            if retrieved:
                for idx, r in enumerate(retrieved[:4], 1):
                    preview = (r.get("text", "")[:180] + "...") if len(r.get("text", "")) > 180 else r.get("text", "")
                    retrieved_html += f"""
                    <div style='background:#161633;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px;margin-bottom:6px;'>
                        <div style='display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:3px;'>
                            <span style='color:#a78bfa;font-family:monospace;font-weight:600'>[{idx}] {r.get('source')}</span>
                            <span style='color:#94a3b8'>p.{r.get('page')}</span>
                        </div>
                        <p style='color:#94a3b8;font-size:0.75rem;margin:0;line-height:1.4'>{preview}</p>
                    </div>"""

            sources_html = f"""
            <div class='fade-in'>
                <!-- Status card -->
                <div style='background:#12122a;border:1px solid rgba(124,58,237,0.3);border-radius:10px;padding:12px;margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span style='background:{rc}22;border:1px solid {rc}66;color:{rc};padding:3px 10px;border-radius:99px;font-size:0.75rem;font-weight:600;text-transform:uppercase;'>
                            {r_label}
                        </span>
                        <span style='color:#10b981;font-weight:600;font-size:0.8rem;'>
                            🎯 Confidence: {confidence*100:.0f}%
                        </span>
                    </div>
                    <div style='color:#94a3b8;font-size:0.75rem;margin-top:6px;'>
                        ⏱️ Latency: <strong style='color:#e2e8f0'>{latency:.2f}s</strong> · Session: <code style='color:#a78bfa'>{thread_id}</code>
                    </div>
                </div>

                <!-- Execution trace -->
                <details open style='background:#0f0f26;border:1px solid rgba(124,58,237,0.25);border-radius:10px;padding:10px;margin-bottom:12px;'>
                    <summary style='color:#a78bfa;font-size:0.8rem;font-weight:600;cursor:pointer;user-select:none;'>
                        ⚡ Agent Reasoning Trace ({len(steps)} steps)
                    </summary>
                    <div style='margin-top:8px;'>
                        {steps_html}
                    </div>
                </details>

                <!-- Citations -->
                <div style='background:#12122a;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:10px;margin-bottom:12px;'>
                    <div style='font-size:0.78rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;'>📚 Citations</div>
                    {cites_html}
                </div>

                <!-- Retrieved Chunks Preview -->
                <details style='background:#0f0f26;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:10px;'>
                    <summary style='color:#94a3b8;font-size:0.78rem;font-weight:600;cursor:pointer;user-select:none;'>
                        📄 Retrieved Context Chunks ({len(retrieved)})
                    </summary>
                    <div style='margin-top:8px;'>
                        {retrieved_html}
                    </div>
                </details>
            </div>
            """
            status = f"✅ Agent completed in {latency:.2f}s · Route: {route.upper()} · Confidence: {confidence*100:.0f}%"

        else:
            # Baseline Naive RAG
            rag_query = _get_rag_query()
            result = rag_query(message)
            latency = time.perf_counter() - t0
            answer = result["answer"]
            sources = result["sources"]
            n_chunks = result["retrieved_chunks"]

            sources_html = "<div class='fade-in'>"
            sources_html += f"""
            <div style='background:#12122a;border:1px solid rgba(59,130,246,0.3);border-radius:10px;padding:10px;margin-bottom:12px;'>
                <div style='color:#60a5fa;font-size:0.78rem;font-weight:600;'>ℹ️ Baseline Linear Pipeline</div>
                <div style='color:#94a3b8;font-size:0.75rem;margin-top:3px;'>
                    Single vector search (k=5) → LLM generate. No routing, no CRAG grading, no memory.
                </div>
                <div style='color:#94a3b8;font-size:0.75rem;margin-top:4px;'>
                    ⏱️ Latency: <strong style='color:#e2e8f0'>{latency:.2f}s</strong> · Chunks: {n_chunks}
                </div>
            </div>
            """
            for s in sources:
                sources_html += f"""
                <div style='background:#1a1a35;border:1px solid rgba(124,58,237,0.3);border-radius:8px;padding:10px;margin-bottom:6px;'>
                    <div style='display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px;'>
                        <span style='color:#a78bfa;font-family:monospace;font-weight:600'>📄 {s['source']}</span>
                        <span style='color:#94a3b8'>Page {s['page']}</span>
                    </div>
                    <p style='color:#94a3b8;font-size:0.75rem;line-height:1.4;margin:0'>{s['content'][:200]}...</p>
                </div>"""
            sources_html += "</div>"
            status = f"✅ Baseline completed in {latency:.2f}s · Retrieved {n_chunks} chunks"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, sources_html, status, thread_id

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "", f"❌ {str(e)}", thread_id or ""


def ingest_fn(files) -> str:
    """Handle PDF ingestion from uploaded files."""
    if not files:
        return "⚠️ No files uploaded."

    ingest_documents, _ = _get_ingest()
    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = pdf_dir / Path(f.name).name
        import shutil
        shutil.copy(f.name, dest)
        saved.append(Path(f.name).name)

    result = ingest_documents(pdf_dir=str(pdf_dir))

    if result["status"] == "success":
        from rag.retrieval import reset_vectorstore
        reset_vectorstore()
        return (
            f"✅ **Ingestion Complete!**\n\n"
            f"- 📄 PDF files: **{result['pdf_files']}**\n"
            f"- 📃 Pages loaded: **{result['pages_loaded']}**\n"
            f"- ✂️ Chunks created: **{result['chunks_created']}**\n"
            f"- 🔢 Embedding model: `{result['embedding_model']}`\n"
            f"- 💾 Stored in: `{result['persist_dir']}`"
        )
    else:
        return f"❌ Error: {result.get('message', 'Unknown error')}"


def get_stats_html() -> str:
    """Return HTML for the stats panel."""
    try:
        _, get_collection_stats = _get_ingest()
        stats = get_collection_stats()
        count = stats.get("vector_count", 0)
        status = stats.get("status", "unknown")
        model = stats.get("embedding_model", "all-MiniLM-L6-v2")

        color = "#10b981" if status == "ready" else "#f59e0b"
        status_icon = "🟢" if status == "ready" else "🟡"

        return f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>
            <div class='stat-box'>
                <div class='stat-number'>{count:,}</div>
                <div class='stat-label'>Vectors Stored</div>
            </div>
            <div class='stat-box'>
                <div style='font-size:1.6rem;margin-bottom:4px'>{status_icon}</div>
                <div class='stat-label'>DB Status: {status.upper()}</div>
            </div>
            <div class='stat-box'>
                <div style='font-size:0.85rem;color:#a78bfa;font-family:monospace;margin-bottom:4px'>MiniLM-L6-v2</div>
                <div class='stat-label'>Embedding Model</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<p style='color:#f59e0b'>⚠️ Could not load stats: {e}</p>"


def run_evaluation_fn() -> Tuple[str, str]:
    """Run RAGAs evaluation and return results."""
    try:
        evaluate_with_sample_questions = _get_evaluation()
        rag_query = _get_rag_query()

        result = evaluate_with_sample_questions(rag_query)

        if result["status"] == "success":
            faithfulness = result.get("faithfulness", 0)
            relevancy = result.get("answer_relevancy", 0)
            n = result.get("questions_evaluated", 0)

            # Color based on score
            def score_color(s):
                if s >= 0.8:   return "#10b981"
                elif s >= 0.6: return "#f59e0b"
                else:          return "#ef4444"

            html = f"""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px'>
                <div class='stat-box' style='padding:24px'>
                    <div class='stat-number' style='color:{score_color(faithfulness)}'>{faithfulness:.1%}</div>
                    <div class='stat-label'>Faithfulness Score</div>
                    <div style='color:#475569;font-size:0.77rem;margin-top:6px'>Answer stays faithful to retrieved context</div>
                </div>
                <div class='stat-box' style='padding:24px'>
                    <div class='stat-number' style='color:{score_color(relevancy)}'>{relevancy:.1%}</div>
                    <div class='stat-label'>Answer Relevancy</div>
                    <div style='color:#475569;font-size:0.77rem;margin-top:6px'>How relevant is the answer to the question</div>
                </div>
            </div>
            <div style='background:#12122a;border:1px solid rgba(124,58,237,0.2);border-radius:10px;padding:14px'>
                <p style='color:#94a3b8;font-size:0.85rem;margin:0'>
                    📊 Evaluated on <strong style='color:#e2e8f0'>{n} questions</strong> · 
                    Judge LLM: <strong style='color:#a78bfa'>Gemini 1.5 Flash</strong> via RAGAs
                </p>
            </div>
            """
            summary = (
                f"✅ Evaluation complete!\n\n"
                f"**Faithfulness:** {faithfulness:.1%}\n"
                f"**Answer Relevancy:** {relevancy:.1%}\n"
                f"**Questions tested:** {n}"
            )
        else:
            html = f"<p style='color:#ef4444'>❌ {result.get('message', 'Evaluation failed')}</p>"
            summary = f"❌ Error: {result.get('message')}"

        return html, summary

    except Exception as e:
        return f"<p style='color:#ef4444'>❌ Error: {e}</p>", f"❌ {str(e)}"


APP_THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


# ── Build Gradio App ──────────────────────────────────────────────────────────
def build_app():
    with gr.Blocks(title="Agentic RAG Document Q&A") as demo:

        # ── Header ──
        gr.HTML("""
        <div id="header-banner">
            <h1 id="app-title">🔍 Agentic RAG Document Q&A</h1>
            <p id="app-subtitle">
                Adaptive LangGraph State Machine · Hybrid BM25 + Dense ChromaDB (RRF) · Self-Correcting CRAG · Multi-turn Memory · AST-Safe Tools
            </p>
            <div class="badge-row">
                <span class="badge badge-purple">LangGraph State Machine</span>
                <span class="badge badge-blue">Hybrid BM25 + Dense RRF</span>
                <span class="badge badge-cyan">CRAG Self-Correction</span>
                <span class="badge badge-pink">SQLite Memory</span>
                <span class="badge badge-green">AST-Safe Tools</span>
                <span class="badge badge-purple">FastAPI / SSE</span>
                <span class="badge badge-blue">ChromaDB</span>
                <span class="badge badge-cyan">LangSmith Tracing</span>
            </div>
        </div>
        """)

        # ── Tabs ──
        with gr.Tabs(elem_id="main-tabs"):

            # ════════════════════════════════════════
            # TAB 1: Chat
            # ════════════════════════════════════════
            with gr.Tab("💬 Agent Chat", elem_id="tab-chat"):
                session_thread = gr.State(lambda: f"ui-{uuid.uuid4().hex[:8]}")

                with gr.Row():
                    pipeline_mode = gr.Radio(
                        choices=["LangGraph Agent (Flagship)", "Baseline Linear RAG"],
                        value="LangGraph Agent (Flagship)",
                        label="⚡ Execution Architecture",
                        interactive=True,
                        scale=3,
                    )

                with gr.Row():
                    # Left: chat
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="",
                            elem_id="chatbot",
                            height=520,
                            show_label=False,
                            avatar_images=(None, "🤖"),
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask about the 10 research papers, comparative topics, or calculations...",
                                show_label=False,
                                scale=5,
                                lines=1,
                                elem_id="msg-input",
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Session", variant="secondary", scale=1)
                            status_box = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                placeholder="Agent routing and status will appear here...",
                                scale=3,
                            )

                        # Example questions demonstrating diverse routing
                        gr.HTML("<p style='color:#475569;font-size:0.8rem;margin-top:4px'>💡 Try these pipeline capability demonstrations:</p>")
                        with gr.Row():
                            ex1 = gr.Button("📌 Factual: Multi-Head Attention", variant="secondary", size="sm")
                            ex2 = gr.Button("🔀 Compare: BERT vs GPT-3", variant="secondary", size="sm")
                        with gr.Row():
                            ex3 = gr.Button("🛡️ Refusal: Chocolate Cake Recipe", variant="secondary", size="sm")
                            ex4 = gr.Button("🧮 Tool: LoRA Param Scaling", variant="secondary", size="sm")

                    # Right: execution trace & sources
                    with gr.Column(scale=2):
                        gr.HTML("<p style='color:#94a3b8;font-size:0.85rem;font-weight:600;margin-bottom:8px'>⚡ Agent Execution & Sources</p>")
                        sources_panel = gr.HTML(
                            value="<p style='color:#475569;font-style:italic;font-size:0.88rem;padding:16px'>Agent reasoning trace, confidence scores, and source citations will appear here after you ask a question.</p>"
                        )

                # Wire up events
                def send(message, history, pipeline, thread_id):
                    history, sources, status, thread_id = chat_fn(message, history, pipeline, thread_id)
                    return history, sources, status, "", thread_id

                send_btn.click(
                    send,
                    inputs=[msg_input, chatbot, pipeline_mode, session_thread],
                    outputs=[chatbot, sources_panel, status_box, msg_input, session_thread],
                )
                msg_input.submit(
                    send,
                    inputs=[msg_input, chatbot, pipeline_mode, session_thread],
                    outputs=[chatbot, sources_panel, status_box, msg_input, session_thread],
                )
                clear_btn.click(
                    lambda: ([], "<p style='color:#475569;font-style:italic;font-size:0.88rem;padding:16px'>Session cleared.</p>", "Chat cleared.", f"ui-{uuid.uuid4().hex[:8]}"),
                    outputs=[chatbot, sources_panel, status_box, session_thread],
                )

                ex1.click(lambda: "What is multi-head attention and why is it useful in transformers?", outputs=msg_input)
                ex2.click(lambda: "Compare the pretraining objectives and architectures of BERT vs GPT-3.", outputs=msg_input)
                ex3.click(lambda: "Can you give me a recipe to bake a chocolate cake?", outputs=msg_input)
                ex4.click(lambda: "Calculate the trainable parameter count for LoRA with rank r=8, d_model=4096, and 4 adapted matrices.", outputs=msg_input)

            # ════════════════════════════════════════
            # TAB 2: Upload & Ingest
            # ════════════════════════════════════════
            with gr.Tab("📄 Upload Documents", elem_id="tab-upload"):
                gr.HTML("""
                <div style='background:#12122a;border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:20px;margin-bottom:16px'>
                    <h3 style='color:#e2e8f0;margin:0 0 8px'>Upload PDF Documents</h3>
                    <p style='color:#94a3b8;font-size:0.88rem;margin:0'>
                        Upload 1–30 PDF files. They will be chunked (1,000 chars, 200 overlap),
                        embedded with <strong style='color:#a78bfa'>all-MiniLM-L6-v2</strong>,
                        indexed in <strong style='color:#67e8f9'>ChromaDB</strong>, and indexed with <strong style='color:#f472b6'>BM25</strong> for hybrid search.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Drop PDF files here",
                            file_types=[".pdf"],
                            file_count="multiple",
                            elem_classes=["upload-area"],
                        )
                        ingest_btn = gr.Button("⚡ Ingest Documents", variant="primary")

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style='background:#12122a;border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:16px;'>
                            <p style='color:#a78bfa;font-size:0.85rem;font-weight:600;margin-bottom:12px'>⚙️ Ingestion Pipeline</p>
                            <div style='display:flex;flex-direction:column;gap:8px'>
                                <div style='display:flex;align-items:center;gap:8px;color:#94a3b8;font-size:0.83rem'>
                                    <span style='color:#7c3aed'>①</span> PyPDFDirectoryLoader
                                </div>
                                <div style='display:flex;align-items:center;gap:8px;color:#94a3b8;font-size:0.83rem'>
                                    <span style='color:#7c3aed'>②</span> RecursiveTextSplitter (1000/200)
                                </div>
                                <div style='display:flex;align-items:center;gap:8px;color:#94a3b8;font-size:0.83rem'>
                                    <span style='color:#7c3aed'>③</span> all-MiniLM-L6-v2 Embeddings
                                </div>
                                <div style='display:flex;align-items:center;gap:8px;color:#94a3b8;font-size:0.83rem'>
                                    <span style='color:#7c3aed'>④</span> ChromaDB Vector Store
                                </div>
                                <div style='display:flex;align-items:center;gap:8px;color:#94a3b8;font-size:0.83rem'>
                                    <span style='color:#7c3aed'>⑤</span> BM25 Okapi Sparse Index
                                </div>
                            </div>
                        </div>
                        """)

                ingest_output = gr.Markdown(
                    value="*Upload PDFs and click 'Ingest Documents' to begin.*",
                    elem_id="ingest-output",
                )
                ingest_btn.click(ingest_fn, inputs=[file_upload], outputs=[ingest_output])

                # ── Quick ingest from local dir ──
                gr.HTML("<hr style='border-color:rgba(124,58,237,0.2);margin:24px 0'>")
                gr.HTML("""
                <p style='color:#94a3b8;font-size:0.88rem;margin-bottom:8px'>
                    <strong style='color:#e2e8f0'>Need sample documents?</strong>
                    Click below to auto-download and ingest the 10 seminal AI research papers (Transformers, BERT, RAG, GPT-3, LoRA, Mistral, etc.):
                </p>
                """)
                ingest_dir_btn = gr.Button("📥 Auto-Ingest 10 Seminal AI Papers", variant="secondary")
                ingest_dir_output = gr.Markdown()

                def ingest_from_dir():
                    ingest_documents, _ = _get_ingest()
                    result = ingest_documents()
                    if result["status"] == "success":
                        from rag.retrieval import reset_vectorstore
                        reset_vectorstore()
                        return (
                            f"✅ **Ingestion Complete!** Loaded {result['pages_loaded']} pages · "
                            f"{result['chunks_created']} chunks from **{result['pdf_files']} PDFs**!"
                        )
                    return f"❌ {result.get('message')}"

                ingest_dir_btn.click(ingest_from_dir, outputs=[ingest_dir_output])

            # ════════════════════════════════════════
            # TAB 3: Evaluate & Benchmarks
            # ════════════════════════════════════════
            with gr.Tab("📊 Benchmarks & Eval", elem_id="tab-eval"):
                gr.HTML("""
                <div style='background:#12122a;border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:20px;margin-bottom:16px'>
                    <h3 style='color:#e2e8f0;margin:0 0 8px'>System Evaluation & Empirical Sweep Results</h3>
                    <p style='color:#94a3b8;font-size:0.88rem;margin:0'>
                        Quantitative baseline-vs-agent evaluation across the <strong>38-question benchmark suite</strong>
                        (factual, comparison, out-of-scope refusal, multi-hop, and adversarial questions).
                    </p>
                </div>

                <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;'>
                    <div class='stat-box' style='padding:16px'>
                        <div class='stat-number' style='color:#10b981'>0.713</div>
                        <div class='stat-label'>Agent Correctness</div>
                        <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>vs 0.675 baseline (+5.6%)</div>
                    </div>
                    <div class='stat-box' style='padding:16px'>
                        <div class='stat-number' style='color:#10b981'>100%</div>
                        <div class='stat-label'>Refusal Accuracy</div>
                        <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>vs 87.5% baseline (7/8 → 8/8)</div>
                    </div>
                    <div class='stat-box' style='padding:16px'>
                        <div class='stat-number' style='color:#38bdf8'>11.9 ms</div>
                        <div class='stat-label'>Hybrid RRF Latency</div>
                        <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>+2.5 ms CPU over dense vector</div>
                    </div>
                    <div class='stat-box' style='padding:16px'>
                        <div class='stat-number' style='color:#c084fc'>100%</div>
                        <div class='stat-label'>Retrieval Hit Rate</div>
                        <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>at 1,000-char chunks (28/28)</div>
                    </div>
                </div>

                <div style='background:#12122a;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:18px;margin-bottom:16px;'>
                    <h4 style='color:#e2e8f0;margin:0 0 10px;font-size:0.9rem;'>📐 Measured Chunk-Size Sweep (10 Papers, 309 Pages)</h4>
                    <table style='width:100%;font-size:0.8rem;color:#94a3b8;border-collapse:collapse;'>
                        <tr style='border-bottom:1px solid rgba(255,255,255,0.1);color:#e2e8f0;text-align:left;'>
                            <th style='padding:6px;'>Chunk Size</th>
                            <th style='padding:6px;'>Overlap</th>
                            <th style='padding:6px;'>Total Chunks</th>
                            <th style='padding:6px;'>In-Scope Hit Rate</th>
                            <th style='padding:6px;'>Engineering Trade-off</th>
                        </tr>
                        <tr style='border-bottom:1px solid rgba(255,255,255,0.04);'>
                            <td style='padding:6px;'>500 chars</td>
                            <td style='padding:6px;'>100 chars</td>
                            <td style='padding:6px;'>2,492</td>
                            <td style='padding:6px;color:#f59e0b;'>96.4% (27/28)</td>
                            <td style='padding:6px;'>Over-fragmented text; split definitions across chunk boundaries</td>
                        </tr>
                        <tr style='border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(124,58,237,0.1);'>
                            <td style='padding:6px;color:#c084fc;font-weight:600;'>1,000 chars (Selected)</td>
                            <td style='padding:6px;'>200 chars</td>
                            <td style='padding:6px;'>1,290</td>
                            <td style='padding:6px;color:#10b981;font-weight:600;'>100.0% (28/28)</td>
                            <td style='padding:6px;color:#e2e8f0;'>Optimal balance of context coherence and prompt token budget</td>
                        </tr>
                        <tr>
                            <td style='padding:6px;'>1,500 chars</td>
                            <td style='padding:6px;'>300 chars</td>
                            <td style='padding:6px;'>888</td>
                            <td style='padding:6px;color:#10b981;'>100.0% (28/28)</td>
                            <td style='padding:6px;'>Diluted embedding specificity; bloated top-5 context window</td>
                        </tr>
                    </table>
                </div>

                <div style='background:#12122a;border:1px solid rgba(6,182,212,0.2);border-radius:12px;padding:16px;margin-bottom:16px'>
                    <p style='color:#67e8f9;font-size:0.83rem;font-weight:600;margin-bottom:8px'>📋 Defensible Resume Bullet Points</p>
                    <p style='color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.7'>
                        • Rebuilt a linear RAG chain into a <strong>LangGraph agent</strong> over <strong>10 research papers</strong> (1,290 chunks) with query routing, dynamic cross-paper decomposition, hybrid retrieval (BM25 + vector, RRF, <em>k</em>=60), a self-correcting CRAG retry loop, and groundedness verification.<br>
                        • Built a <strong>38-question</strong> LangSmith evaluation benchmark; in pilot experiments across representative categories, achieved <strong>correctness of 0.675 → 0.713</strong> and <strong>100% out-of-scope refusal accuracy</strong>; selected 1,000-char chunking via an empirical sweep (100% retrieval hit rate vs. 96.4% at 500 chars).<br>
                        • Added tool use (document search, arXiv lookup, safe AST-based calculator), checkpointed multi-turn memory via <code>SqliteSaver</code>, and a human-approval interrupt before ingesting new papers.<br>
                        • Shipped a streaming FastAPI backend (SSE <code>/chat/stream</code>) and Gradio UI with feedback logged to LangSmith, plus 53 pytest unit/contract tests, Docker, and GitHub Actions CI; deployed on HuggingFace Spaces.
                    </p>
                </div>
                """)

                eval_btn = gr.Button("🧪 Run Live RAGAs Sample Evaluation", variant="secondary")
                eval_scores_html = gr.HTML(
                    value="<p style='color:#475569;font-style:italic;padding:12px'>Click above to run live sample questions through RAGAs.</p>"
                )
                eval_summary = gr.Markdown()
                eval_btn.click(run_evaluation_fn, outputs=[eval_scores_html, eval_summary])

            # ════════════════════════════════════════
            # TAB 4: Stats & Info
            # ════════════════════════════════════════
            with gr.Tab("ℹ️ Stats & Info", elem_id="tab-stats"):
                refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                stats_html = gr.HTML(get_stats_html())
                refresh_btn.click(get_stats_html, outputs=[stats_html])

                gr.HTML("""
                <div style='margin-top:20px;background:#12122a;border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:20px'>
                    <p style='color:#a78bfa;font-size:0.9rem;font-weight:600;margin-bottom:16px'>🏗️ System Architecture & Stack</p>
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
                        <div>
                            <p style='color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Agent Core & Retrieval</p>
                            <ul style='color:#94a3b8;font-size:0.83rem;line-height:2;list-style:none;padding:0;margin:0'>
                                <li>⚡ <strong>LangGraph</strong> (State machine, routing, CRAG retry loop)</li>
                                <li>🔍 <strong>Hybrid RRF</strong> (BM25 Okapi + ChromaDB cosine, k=60)</li>
                                <li>💾 <strong>SqliteSaver</strong> (Stateful conversational memory by thread_id)</li>
                                <li>🛡️ <strong>AST Calculator</strong> (Whitelisted math parser without eval())</li>
                                <li>🧠 <strong>Google Gemini</strong> (2.0 Flash / 1.5 Flash)</li>
                            </ul>
                        </div>
                        <div>
                            <p style='color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>Serving & Observability</p>
                            <ul style='color:#94a3b8;font-size:0.83rem;line-height:2;list-style:none;padding:0;margin:0'>
                                <li>🚀 <strong>FastAPI</strong> (SSE streaming endpoint /chat/stream)</li>
                                <li>🎨 <strong>Gradio</strong> (Dual-pipeline comparison UI)</li>
                                <li>🔭 <strong>LangSmith</strong> (Full trace observability & human feedback)</li>
                                <li>🧪 <strong>pytest</strong> (53 offline unit & contract tests)</li>
                                <li>🐳 <strong>Docker & HF Spaces</strong> (Containerized deployment)</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """)

        # ── Footer ──
        gr.HTML("""
        <div style='text-align:center;padding:24px 0 8px;color:#2d2d5a;font-size:0.78rem'>
            Agentic RAG Document Q&A · LangGraph · ChromaDB · BM25 RRF · Gemini · LangSmith
        </div>
        """)

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
        theme=APP_THEME,
        css=CUSTOM_CSS,
    )
