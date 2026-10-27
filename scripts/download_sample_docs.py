"""
download_sample_docs.py — Downloads 10 open-access AI research papers from arXiv.

Run this once to populate data/pdfs/ so the RAG chatbot works out-of-the-box.
No API key needed — these are all publicly available papers.

Usage:
    python scripts/download_sample_docs.py
"""

import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

# Force UTF-8 stdout/stderr encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── Papers to download (arXiv open-access PDFs) ───────────────────────────────
PAPERS = [
    {
        "title": "Attention Is All You Need (Transformers)",
        "url": "https://arxiv.org/pdf/1706.03762",
        "filename": "01_attention_is_all_you_need.pdf",
    },
    {
        "title": "BERT: Pre-training Deep Bidirectional Transformers",
        "url": "https://arxiv.org/pdf/1810.04805",
        "filename": "02_bert.pdf",
    },
    {
        "title": "RAG: Retrieval-Augmented Generation for NLP",
        "url": "https://arxiv.org/pdf/2005.11401",
        "filename": "03_rag_paper.pdf",
    },
    {
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "url": "https://arxiv.org/pdf/2005.14165",
        "filename": "04_gpt3.pdf",
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "url": "https://arxiv.org/pdf/2302.13971",
        "filename": "05_llama.pdf",
    },
    {
        "title": "Mistral 7B",
        "url": "https://arxiv.org/pdf/2310.06825",
        "filename": "06_mistral_7b.pdf",
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning",
        "url": "https://arxiv.org/pdf/2201.11903",
        "filename": "07_chain_of_thought.pdf",
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "url": "https://arxiv.org/pdf/2106.09685",
        "filename": "08_lora.pdf",
    },
    {
        "title": "InstructGPT: Training Language Models to Follow Instructions",
        "url": "https://arxiv.org/pdf/2203.02155",
        "filename": "09_instructgpt.pdf",
    },
    {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "url": "https://arxiv.org/pdf/1908.10084",
        "filename": "10_sentence_bert.pdf",
    },
]

PDF_DIR = Path("./data/pdfs")


def download_paper(paper: dict, dest_dir: Path) -> bool:
    """Download a single PDF paper with retry logic."""
    dest = dest_dir / paper["filename"]

    if dest.exists():
        print(f"  ✅ Already exists: {paper['filename']}")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAG-Chatbot-Demo/1.0; research purposes)"
    }

    for attempt in range(3):
        try:
            print(f"  ⬇️  Downloading: {paper['title']}")
            response = requests.get(paper["url"], headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"     {paper['filename']}",
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"  ✅ Saved: {paper['filename']} ({dest.stat().st_size / 1024:.0f} KB)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff

    print(f"  ❌ Failed to download: {paper['filename']}")
    return False


def main():
    print("\n" + "=" * 60)
    print("  RAG Chatbot — Sample Document Downloader")
    print("  Downloading 10 AI research papers from arXiv")
    print("=" * 60 + "\n")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving to: {PDF_DIR.resolve()}\n")

    success, failed = 0, 0

    for i, paper in enumerate(PAPERS, 1):
        print(f"[{i}/{len(PAPERS)}] {paper['title']}")
        if download_paper(paper, PDF_DIR):
            success += 1
        else:
            failed += 1
        # Be polite to arXiv servers
        if i < len(PAPERS):
            time.sleep(1.5)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Downloaded: {success} papers")
    print(f"  ❌ Failed:     {failed} papers")
    print(f"{'=' * 60}\n")

    if success > 0:
        print("🚀 Next step — ingest documents into ChromaDB:")
        print("   python -c \"from rag.ingestion import ingest_documents; ingest_documents()\"")
        print("\n   Or just launch the app and use the 'Upload Documents' tab:")
        print("   python app.py\n")


if __name__ == "__main__":
    main()
