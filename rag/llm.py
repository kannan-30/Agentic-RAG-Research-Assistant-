"""
llm.py — Gemini LLM client via LangChain Google GenAI.

Uses Google Gemini (default: gemini-2.0-flash) for answer generation.
Model name is configurable via the LLM_MODEL / GEMINI_MODEL env var.

Design decision: a module-level singleton ``_llm`` is created once at first
call and reused for subsequent requests.  This avoids re-authenticating and
re-instantiating the client on every query (which added ~50 ms per request).
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# LLM_MODEL is the canonical env var; GEMINI_MODEL is kept for back-compat.
# gemini-3.5-flash-lite is confirmed available and active on the free-tier AI Studio key.
GEMINI_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Module-level singleton — built on first call to get_llm().
_llm: "ChatGoogleGenerativeAI | None" = None

# ── RAG Prompt Template ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI research assistant. Your job is to answer questions \
accurately using ONLY the provided document context.

Rules:
- Base your answer strictly on the given context.
- If the context doesn't contain enough information, say: "I don't have enough information in the \
provided documents to answer this confidently."
- Always be concise, clear, and factual.
- Cite which part of the context supports your answer when possible.
- Never hallucinate or make up information."""

RAG_PROMPT_TEMPLATE = """Use the following retrieved context to answer the question below.

──────────── CONTEXT ────────────
{context}
─────────────────────────────────

Question: {question}

Answer:"""


def get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """
    Return a configured Gemini LLM instance (singleton unless model_name overrides).

    When ``model_name`` is None we return the module-level singleton, building
    it on first call.  Passing an explicit name bypasses the singleton so that
    the fallback candidate loop in ``generate_answer`` can try alternate models.
    """
    global _llm  # noqa: PLW0603

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key.strip() == "" or api_key.strip() == "your_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY not found or invalid. "
            "Set it in .env locally or in Space Settings → Secrets on HuggingFace."
        )

    target_model = model_name or GEMINI_MODEL

    # Return singleton when no override is requested.
    if model_name is None and _llm is not None:
        return _llm

    instance = ChatGoogleGenerativeAI(
        model=target_model,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        google_api_key=api_key.strip(),
    )

    if model_name is None:
        _llm = instance

    return instance


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer given a question and retrieved context.
    Attempts candidate models if the configured model returns a 404 NOT_FOUND.
    
    Args:
        question: The user's question.
        context: Retrieved document chunks concatenated.
    
    Returns:
        The LLM-generated answer string.
    """
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Start with the configured model; fall through to known-good models on 404.
    candidate_models = list(dict.fromkeys([
        GEMINI_MODEL,
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
    ]))

    last_exception = None
    for model_name in candidate_models:
        try:
            llm = get_llm(model_name=model_name)
            response = llm.invoke(messages)
            if isinstance(response.content, list):
                text_content = "".join(
                    [chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in response.content]
                )
            else:
                text_content = str(response.content)
            return text_content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "404" in err_str or "not_found" in err_str or "not found" in err_str:
                last_exception = e
                continue
            raise e

    if last_exception:
        raise ValueError(
            "Gemini API 404 Error: Google Generative Language API could not access Gemini models with your GOOGLE_API_KEY. "
            "Please ensure: 1) Your key was generated at https://aistudio.google.com/app/apikey (not Google Cloud Console). "
            "2) Generative Language API is enabled. 3) No extra quotes or trailing spaces are in your GOOGLE_API_KEY secret."
        )

