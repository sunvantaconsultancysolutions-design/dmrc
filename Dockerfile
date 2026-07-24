FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + the pre-built ChromaDB vector store (already in the repo).
# NOTE: retrieval_caps.py is now imported directly by src/app.py, so unlike
# the notebook's server-launch cell, no sitecustomize.py / PYTHONPATH dance
# is needed here -- app.py is this container's own entrypoint.
COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# Retrieval caps (from 02_Gemma_Inference_and_Serving.ipynb, cell "Runtime
# caps on retrieval breadth") -- kept as env vars, not notebook magic.
ENV RAG_MAX_CANDIDATES=20 \
    RAG_MAX_CONTEXT=4 \
    GEMMA_USE_4BIT=0 \
    GEMMA_MAX_NEW_TOKENS=320 \
    ALLOWED_ORIGINS="*"

EXPOSE 8000

# HF weights (BGE-M3, BGE-reranker, Gemma-2-9B-it) download on first
# container start into HF_HOME. Mount a persistent volume at /models on
# your GPU host so redeploys don't re-download ~20GB every time.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
