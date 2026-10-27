"""
ingestion.py — PDF loading, chunking, embedding, and ChromaDB storage.

Pipeline:
  PDF files → LangChain DirectoryLoader → RecursiveCharacterTextSplitter
            → HuggingFace all-MiniLM-L6-v2 embeddings → ChromaDB
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
COLLECTION_NAME = "rag_documents"

# Module-level singleton: the 90 MB MiniLM model is downloaded once and reused.
_embeddings: "HuggingFaceEmbeddings | None" = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the Sentence Transformers embedding model (singleton).

    The model is ~90 MB and takes ~2 s to initialise; loading it once at
    process start and reusing it eliminates that cost from every query.
    """
    global _embeddings  # noqa: PLW0603
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def load_pdfs(pdf_dir: str = PDF_DIR) -> List:
    """Load all PDF files from a directory using LangChain's PyPDFDirectoryLoader."""
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("📁 No PDF files found in directory. Automatically downloading 10 sample arXiv papers...")
        try:
            from scripts.download_sample_docs import main as download_main
            download_main()
            pdf_files = list(pdf_dir.glob("*.pdf"))
        except Exception as e:
            print(f"⚠️ Could not auto-download sample docs: {e}")

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}. Please upload PDFs or place them in data/pdfs/.")

    print(f"📄 Found {len(pdf_files)} PDF(s). Loading...")
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from {len(pdf_files)} document(s).")
    return documents


def chunk_documents(documents: List) -> List:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    
    Strategy: splits on paragraph breaks → newlines → sentences → words
    This preserves context better than fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def create_vector_store(chunks: List, persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Embed chunks and persist them to ChromaDB."""
    print(f"🔢 Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = get_embeddings()

    # Clear existing collection if it exists
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Cleared existing ChromaDB collection.")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Stored {len(chunks)} vectors in ChromaDB at '{persist_dir}'.")
    return vectorstore


def load_vector_store(persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def ingest_documents(pdf_dir: str = PDF_DIR, persist_dir: str = CHROMA_PERSIST_DIR) -> dict:
    """
    Full ingestion pipeline:
      1. Load PDFs  →  2. Chunk  →  3. Embed  →  4. Store in ChromaDB
    
    Returns a summary dict with stats.
    """
    try:
        documents = load_pdfs(pdf_dir)
        chunks = chunk_documents(documents)
        create_vector_store(chunks, persist_dir)

        # Count unique source files
        sources = set(c.metadata.get("source", "") for c in chunks)

        return {
            "status": "success",
            "pdf_files": len(sources),
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "persist_dir": persist_dir,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_collection_stats(persist_dir: str = CHROMA_PERSIST_DIR) -> dict:
    """Return stats about the current ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(COLLECTION_NAME)
        return {
            "status": "ready",
            "collection_name": COLLECTION_NAME,
            "vector_count": collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "persist_dir": persist_dir,
        }
    except Exception as e:
        return {
            "status": "empty",
            "vector_count": 0,
            "message": "No documents ingested yet. Run ingestion first.",
        }
