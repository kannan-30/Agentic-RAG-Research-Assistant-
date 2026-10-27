FROM python:3.10-slim

LABEL org.opencontainers.image.title="Agentic RAG Document Q&A Chatbot"
LABEL org.opencontainers.image.description="LangGraph agentic RAG with FastAPI + Gradio"

# Install system deps for sentence-transformers and ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories that are expected at runtime
RUN mkdir -p data/pdfs chroma_db eval/results

# Default port for Gradio (HuggingFace Spaces uses 7860)
EXPOSE 7860

# Health check against the FastAPI backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the Gradio app (which also starts FastAPI in a thread)
CMD ["python", "app.py"]
