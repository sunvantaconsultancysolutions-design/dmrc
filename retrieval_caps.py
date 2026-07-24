"""
retrieval_caps.py

Runtime caps on retrieval breadth.

Originally a `%%writefile` cell in 02_Gemma_Inference_and_Serving.ipynb --
promoted here to a permanent, version-controlled module so production
deployments (Docker/RunPod) don't depend on a notebook having been run
first to generate this file.

Imported for its side effects: wraps hybrid_search() and rerank() in place
so every caller -- including app.py's /ask endpoint -- gets bounded output
without any change to app.py itself. app.py imports this module once at
startup (see the import added near the top of app.py).
"""
import os

import src.hybrid_retriever as hybrid_retriever
import src.reranker as reranker

MAX_CANDIDATES = int(os.environ.get("RAG_MAX_CANDIDATES", "20"))  # into reranker
MAX_CONTEXT = int(os.environ.get("RAG_MAX_CONTEXT", "4"))          # into the LLM

_orig_hybrid = hybrid_retriever.hybrid_search
_orig_rerank = reranker.rerank


def _capped_hybrid(*args, **kwargs):
    hits = _orig_hybrid(*args, **kwargs)
    if hits and len(hits) > MAX_CANDIDATES:
        print(f"[cap] hybrid_search {len(hits)} -> {MAX_CANDIDATES}", flush=True)
        hits = hits[:MAX_CANDIDATES]
    return hits


def _capped_rerank(*args, **kwargs):
    out = _orig_rerank(*args, **kwargs)
    if out and len(out) > MAX_CONTEXT:
        print(f"[cap] rerank {len(out)} -> {MAX_CONTEXT}", flush=True)
        out = out[:MAX_CONTEXT]
    return out


hybrid_retriever.hybrid_search = _capped_hybrid
reranker.rerank = _capped_rerank
