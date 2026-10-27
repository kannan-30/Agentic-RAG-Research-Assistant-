"""
rag/graph/tools.py — Tool definitions for the LangGraph tool_agent node.

Tools are plain Python callables decorated with @tool (LangChain) so that
the ReAct tool-agent loop can call them with structured arguments.

Tools implemented here:
1. search_documents  — hybrid retriever (delegates to rag/retrieval.py)
2. list_documents    — names and page counts of ingested papers
3. calculator        — AST-based safe arithmetic (never uses eval())
4. search_arxiv      — arXiv API lookup (public, no key needed)

Safety notes:
- ``calculator`` uses ``ast.parse`` + a whitelist visitor; raw ``eval()`` is
  never called, preventing code execution from user input.
- ``search_arxiv`` returns metadata only; it does NOT auto-ingest — that
  requires explicit human confirmation via the interrupt mechanism.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import os
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 1. search_documents ───────────────────────────────────────────────────────


@tool
def search_documents(query: str, source_filter: Optional[str] = None, k: int = 5) -> str:
    """
    Search the ingested documents using hybrid vector + BM25 retrieval.

    Args:
        query:         Natural-language search query.
        source_filter: Optional filename to restrict search to one document
                       (e.g., "08_lora.pdf").
        k:             Number of chunks to return (default 5).

    Returns:
        Formatted string of retrieved chunks with source and page attribution.
    """
    from rag.retrieval import hybrid_retrieve, format_context  # noqa: PLC0415

    docs = hybrid_retrieve(query, k=k, source_filter=source_filter)
    if not docs:
        return "No relevant documents found."
    return format_context(docs)


# ── 2. list_documents ─────────────────────────────────────────────────────────


@tool
def list_documents() -> str:
    """
    List all documents currently ingested into the vector store.

    Returns a formatted list of filenames with their page counts so the user
    can see what the assistant has access to.
    """
    try:
        import chromadb  # noqa: PLC0415
        from rag.ingestion import CHROMA_PERSIST_DIR, COLLECTION_NAME  # noqa: PLC0415

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(COLLECTION_NAME)

        # Retrieve metadata for all items (chroma returns up to 10 000 by default)
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        # Aggregate page counts per source file
        source_pages: dict[str, set[int]] = {}
        for m in metadatas:
            src = os.path.basename(m.get("source", "unknown"))
            page = m.get("page", 0)
            source_pages.setdefault(src, set()).add(page)

        if not source_pages:
            return "No documents are currently ingested. Use the Upload tab or POST /ingest."

        lines = ["Ingested documents:"]
        for src, pages in sorted(source_pages.items()):
            lines.append(f"  • {src}  ({len(pages)} pages)")
        lines.append(f"\nTotal: {len(source_pages)} documents, {sum(len(p) for p in source_pages.values())} pages.")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("list_documents failed: %s", exc)
        return f"Could not retrieve document list: {exc}"


# ── 3. calculator ─────────────────────────────────────────────────────────────

# Whitelist of allowed AST node types (no function calls, no attribute access).
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
}

# Safe mapping from AST operator types to Python callables.
_BINOP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOP_MAP = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST node."""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(f"Unsafe operation: {type(node).__name__}")
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        fn = _BINOP_MAP.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unknown binary operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large (> 100); refusing to compute.")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        fn = _UNOP_MAP.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unknown unary operator: {type(node.op).__name__}")
        return fn(operand)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical arithmetic expression.

    Only supports: +  -  *  /  //  %  ** and numeric literals.
    Attribute access, function calls, imports, and string operations are all
    rejected.  This prevents code-injection attacks.

    Args:
        expression: An arithmetic expression string, e.g. "(3 + 4) * 2 ** 8".

    Returns:
        The numeric result as a string, or an error message.
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression."
    # Quick guard: reject anything with dots (attribute access) or underscores.
    if "__" in expression:
        return "Error: expression contains disallowed characters."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        # Format nicely: int if no fractional part.
        if math.isfinite(result) and result == int(result):
            return str(int(result))
        return f"{result:.6g}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, TypeError) as exc:
        return f"Error: {exc}"
    except SyntaxError as exc:
        return f"Error: invalid expression syntax — {exc}"


# ── 4. search_arxiv ───────────────────────────────────────────────────────────

ARXIV_API = "https://export.arxiv.org/api/query"


@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for papers matching the query using the public Atom API.

    Returns titles, authors, abstract snippets and PDF links.
    Does NOT automatically ingest; the user must confirm via the chat interface.

    Args:
        query:       Title or topic to search for.
        max_results: Number of results to return (default 3, max 10).

    Returns:
        Formatted string with paper metadata, or an error message.
    """
    max_results = min(int(max_results), 10)
    try:
        resp = requests.get(
            ARXIV_API,
            params={"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("search_arxiv HTTP error: %s", exc)
        return f"arXiv API error: {exc}"

    ns = "http://www.w3.org/2005/Atom"
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        return f"arXiv response parse error: {exc}"

    entries = root.findall(f"{{{ns}}}entry")
    if not entries:
        return f"No arXiv results found for: {query!r}"

    lines: list[str] = [f"arXiv search results for: {query!r}\n"]
    for i, entry in enumerate(entries, 1):
        title_el = entry.find(f"{{{ns}}}title")
        title = title_el.text.strip() if title_el is not None else "Unknown title"

        authors = [
            a.find(f"{{{ns}}}name").text.strip()
            for a in entry.findall(f"{{{ns}}}author")
            if a.find(f"{{{ns}}}name") is not None
        ]
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

        summary_el = entry.find(f"{{{ns}}}summary")
        abstract = summary_el.text.strip()[:300] + "..." if summary_el is not None else ""

        # Prefer the PDF link (rel=related, type=application/pdf)
        pdf_link = ""
        for link in entry.findall(f"{{{ns}}}link"):
            if link.get("type") == "application/pdf":
                pdf_link = link.get("href", "")
                break
        if not pdf_link:
            id_el = entry.find(f"{{{ns}}}id")
            if id_el is not None:
                pdf_link = id_el.text.replace("abs", "pdf").strip()

        lines.append(
            f"[{i}] {title}\n"
            f"    Authors: {author_str}\n"
            f"    Abstract: {abstract}\n"
            f"    PDF: {pdf_link}\n"
        )

    return "\n".join(lines)


# ── Tool registry (used by the tool_agent node) ────────────────────────────────
ALL_TOOLS = [search_documents, list_documents, calculator, search_arxiv]
