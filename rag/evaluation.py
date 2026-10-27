"""
evaluation.py — RAGAs evaluation metrics for the RAG pipeline.

Computes:
  - Faithfulness: Does the answer stay faithful to the retrieved context?
  - Answer Relevancy: How relevant is the answer to the question?

Uses RAGAs library with Gemini as the judge LLM.
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()


# ── Sample evaluation questions (for demo evaluation) ─────────────────────────
SAMPLE_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "How does BERT differ from GPT in its training approach?",
    "What are the key advantages of retrieval-augmented generation?",
    "Explain the concept of fine-tuning in large language models.",
    "What is the role of embeddings in natural language processing?",
]


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None,
) -> Dict[str, Any]:
    """
    Run RAGAs evaluation on a set of question-answer-context triples.
    
    Args:
        questions: List of questions asked.
        answers: List of generated answers.
        contexts: List of lists — retrieved chunks per question.
        ground_truths: Optional list of reference answers.
    
    Returns:
        Dict with faithfulness, answer_relevancy scores and per-question results.
    """
    try:
        import sys
        import types
        # Compatibility patch for ragas importing deprecated langchain_community.chat_models.vertexai
        if "langchain_community.chat_models.vertexai" not in sys.modules:
            try:
                import langchain_community.chat_models.vertexai
            except (ImportError, ModuleNotFoundError):
                mod = types.ModuleType("langchain_community.chat_models.vertexai")
                mod.ChatVertexAI = None
                sys.modules["langchain_community.chat_models.vertexai"] = mod

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Configure Gemini as the judge LLM for RAGAs
        gemini_llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        judge_llm = LangchainLLMWrapper(gemini_llm)

        # Build the evaluation dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        from ragas.embeddings import LangchainEmbeddingsWrapper
        from rag.ingestion import get_embeddings

        embeddings_wrapper = LangchainEmbeddingsWrapper(get_embeddings())

        # Configure metrics to use Gemini judge LLM and HuggingFace embeddings
        metrics = [faithfulness, answer_relevancy]
        for metric in metrics:
            metric.llm = judge_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings_wrapper

        # Run evaluation
        result = evaluate(dataset, metrics=metrics)
        result_df = result.to_pandas()

        return {
            "status": "success",
            "faithfulness": round(float(result_df["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(result_df["answer_relevancy"].mean()), 4),
            "per_question": result_df.to_dict(orient="records"),
            "questions_evaluated": len(questions),
        }

    except ImportError as e:
        return {
            "status": "error",
            "message": f"RAGAs import failed: {e}. Run: pip install ragas datasets",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def evaluate_with_sample_questions(rag_query_fn) -> Dict[str, Any]:
    """
    Run evaluation using the built-in sample questions.
    
    Args:
        rag_query_fn: The rag_query function from retrieval.py
    
    Returns:
        Evaluation results dict.
    """
    print(f"🧪 Running RAGAs evaluation on {len(SAMPLE_QUESTIONS)} questions...")
    
    questions, answers, contexts = [], [], []

    for q in SAMPLE_QUESTIONS:
        try:
            result = rag_query_fn(q)
            if result.get("answer") and result.get("sources"):
                questions.append(q)
                answers.append(result["answer"])
                # contexts is a list of lists (chunks per question)
                chunk_texts = [s["content"] for s in result["sources"]]
                contexts.append(chunk_texts)
        except Exception as e:
            print(f"⚠️  Skipping question due to error: {e}")

    if not questions:
        return {
            "status": "error",
            "message": "No questions could be answered. Ensure documents are ingested.",
        }

    return run_ragas_evaluation(questions, answers, contexts)
