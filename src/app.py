"""
app.py

Chapter 12 -- FastAPI Implementation.

Pure orchestration layer: this module does not retrieve, rerank, or
build prompts itself -- it wires together the three pipeline stages
that already exist, unchanged:

    User Query
        |
        v
    hybrid_retriever.hybrid_search()   (Chapter 9  -- dense + BM25 + merge)
        |
        v
    reranker.rerank()                  (Chapter 10 -- BGE cross-encoder)
        |
        v
    prompt_engineering.build_prompt()  (Chapter 11 -- prompt assembly)
        |
        v
    JSON response

Gemma 3 inference (Chapter 12.12's final "Gemma 3 generates the
answer" step) is intentionally NOT wired in yet -- per this chapter's
scope, the /ask endpoint returns the fully-assembled prompt string as
"answer" so the retrieval -> rerank -> prompt pipeline can be verified
end-to-end over the API before the LLM call exists.

------------------------------------------------------------------------
12.6 API Endpoints implemented in this module
------------------------------------------------------------------------
    GET  /        Health Check
    POST /ask     Question Answering (returns the built prompt, not yet an LLM answer)
    GET  /status  System Status

/upload is intentionally NOT implemented here -- per this task's scope,
document ingestion is handled by separate ingestion scripts, not the
API layer.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .hybrid_retriever import hybrid_search
from .prompt_engineering import NO_CONTEXT_ANSWER, build_prompt, has_usable_context
from .query import get_model as get_dense_model
from .query import extract_clause_no, get_chunk_by_clause_no
from .reranker import get_reranker_model, rerank
from . import query as query_module
from . import reranker as reranker_module
from .gemma_inference import generate_answer, get_gemma_model

logger = logging.getLogger("dmrc_rag.api")


# ---------------------------------------------------------------------------
# 12.12 / 9.12 / 14.7 Retrieval + Reranking Parameters
#
# Call-site overrides of hybrid_retriever.py's and reranker.py's own
# defaults -- NOT edits to those files. These match Table 9.2 / Table
# 14.1's recommended production values (wider candidate pool feeding a
# smaller reranked set) rather than hybrid_retriever.py's smaller
# CLI-friendly defaults.
# ---------------------------------------------------------------------------

DENSE_TOP_K = 20        # Table 9.2: Dense Top-k
BM25_TOP_K = 20         # Table 9.2: BM25 Top-k
MERGED_CANDIDATE_POOL = 40   # Table 9.2 / 14.7: Merged Candidates / Retrieved Documents
RERANK_TOP_N = 10       # Table 9.2 / 14.7: Final Re-ranked / Re-ranked Documents


# ---------------------------------------------------------------------------
# 12.3 / 12.10 Pydantic Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """12.7 Request Model. `query` is the only required field, matching
    the design doc's example exactly: a bare {"query": "..."} request.
    """

    query: str


class SourceItem(BaseModel):
    """One entry in the response's "sources" list, populated from a
    reranked candidate's metadata (Chapter 6/9's metadata schema).
    Fields are optional because not every chunk carries every field
    (e.g. a contract clause has no item_number; a BOQ row has no
    clause_no).
    """

    clause: Optional[str] = None
    page: Optional[int] = None
    document: Optional[str] = None
    item_number: Optional[str] = None
    retrieval_source: Optional[str] = None
    reranker_score: Optional[float] = None
    chunk_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """12.8 Response Model. `confidence` is left as None for now since
    no LLM inference (and therefore no confidence signal) exists yet
    in this chapter's scope.
    """

    answer: str
    sources: List[SourceItem]
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    """12.6 GET /status -- System Status. Reports whether the heavy
    models are actually loaded in memory yet, rather than just
    asserting the process is up, since model loading is the slow /
    failure-prone part of startup (Section 13.15's memory/GPU caveat).
    Also reports whether the ChromaDB connection backing retrieval is
    reachable, since a "running" process with a dead vector store
    would otherwise silently fail on the first /ask request.
    """

    status: str
    embedding_model: str
    reranker_model: str
    dense_model_loaded: bool
    reranker_model_loaded: bool
    chromadb_connected: bool


# ---------------------------------------------------------------------------
# 14.9 Model warm-up during startup (lifespan) -- load the dense
# embedding model and the reranker model once, here, so the first real
# request isn't the one paying the (multi-second, sometimes
# multi-minute on first download) model-load cost. Failures are
# logged, not raised: a slow/failed warm-up shouldn't prevent the API
# from starting, it should just mean the first /ask request pays the
# load cost lazily instead (both get_model() and get_reranker_model()
# are safe to call again on demand).
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Warming up embedding model and reranker model...")
    try:
        get_dense_model()
    except Exception:
        logger.exception("Dense embedding model warm-up failed; will retry lazily on first request.")
    try:
        get_reranker_model()
    except Exception:
        logger.exception("Reranker model warm-up failed; will retry lazily on first request.")
    try:
        get_gemma_model()
    except Exception:
        logger.exception(
            "Gemma 3 warm-up failed; will retry lazily on first request."
        )
    yield


# ---------------------------------------------------------------------------
# 12.9 FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DMRC Contract Intelligence",
    version="1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET / -- 12.6 Health Check
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


# ---------------------------------------------------------------------------
# Helpers -- ChromaDB connectivity check for GET /status
# ---------------------------------------------------------------------------

def _check_chromadb_connected() -> bool:
    """Best-effort check of whether the ChromaDB connection used by
    query.py is up. Deliberately defensive/read-only: any failure to
    reach or introspect the client is treated as "not connected"
    rather than raised, since /status must never itself 500 just
    because the database happens to be down.
    """
    try:
        get_collection = getattr(query_module, "get_collection", None)
        if callable(get_collection):
            collection = get_collection()
            collection.count()
            return True

        client = getattr(query_module, "_client", None) or getattr(query_module, "client", None)
        if client is not None:
            client.heartbeat()
            return True

        collection = getattr(query_module, "_collection", None) or getattr(query_module, "collection", None)
        if collection is not None:
            collection.count()
            return True

        logger.error("No known ChromaDB client/collection accessor found on query module.")
        return False
    except Exception:
        logger.exception("ChromaDB connectivity check failed.")
        return False


# ---------------------------------------------------------------------------
# GET /status -- 12.6 System Status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
def system_status() -> StatusResponse:
    return StatusResponse(
        status="running",
        embedding_model=query_module.MODEL_NAME,
        reranker_model=reranker_module.MODEL_NAME,
        dense_model_loaded=query_module._model is not None,
        reranker_model_loaded=reranker_module._model is not None,
        chromadb_connected=_check_chromadb_connected(),
    )


# ---------------------------------------------------------------------------
# Helpers -- building the "sources" list from reranked metadata (12.8)
# ---------------------------------------------------------------------------

def _build_sources(reranked_candidates: List[Dict[str, Any]]) -> List[SourceItem]:
    sources: List[SourceItem] = []
    for candidate in reranked_candidates:
        metadata = candidate.get("metadata") or {}
        sources.append(
            SourceItem(
                clause=metadata.get("clause_no"),
                page=metadata.get("pdf_page"),
                document=metadata.get("document_name"),
                item_number=metadata.get("item_number"),
                retrieval_source=candidate.get("retrieval_source"),
                reranker_score=candidate.get("reranker_score"),
                chunk_id=candidate.get("chunk_id") or metadata.get("chunk_id"),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# POST /ask -- 12.6 / 12.11 / 12.12 Question Answering Endpoint
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QueryRequest) -> AnswerResponse:
    """12.12 RAG Pipeline Integration (retrieval + rerank + prompt +
    Gemma 3 generation):

        1. Validate the request.                       (FastAPI + Pydantic)
        2. NEW: exact clause-number fast path.          -> get_chunk_by_clause_no()
           If the query names a clause number found verbatim in
           ChromaDB's metadata, use that match directly and skip
           hybrid retrieval entirely. Otherwise fall through to step 3
           exactly as before.
        3. Hybrid retrieval over ChromaDB + BM25.       -> hybrid_search()
        4. Rerank the candidates.                       -> rerank()
        5. Build the final LLM prompt.                  -> build_prompt()
        6. Generate the answer.                         -> generate_answer()
        7. Return the answer + sources as JSON.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="`query` must not be empty.")

    # ----------------------------------------------------------------
    # NEW: Exact clause-number fast path (see query.py's
    # extract_clause_no()/get_chunk_by_clause_no() docstrings for the
    # full rationale). Dense + BM25 rank on chunk TEXT, not on the
    # `clause_no` metadata field, so an exact clause reference like
    # "Explain Clause 1.2.1" can still fail to surface that clause
    # first. Checking metadata directly for an exact match up front
    # fixes that without touching embeddings, BM25, or the merge logic.
    #
    # `candidates` is deliberately left empty (rather than raising) on
    # a clause-lookup failure or "clause named but not found in the
    # corpus" -- either case just means we fall back to the existing
    # hybrid_search() pipeline below, unchanged.
    # ----------------------------------------------------------------
    candidates: List[Dict[str, Any]] = []
    clause_no = extract_clause_no(query_text)
    if clause_no:
        try:
            candidates = get_chunk_by_clause_no(clause_no)
        except Exception:
            logger.exception("Exact clause lookup failed for clause_no: %r", clause_no)
            candidates = []

    if not candidates:
        try:
            candidates = hybrid_search(
                query_text,
                top_k_dense=DENSE_TOP_K,
                top_k_bm25=BM25_TOP_K,
                final_top_k=MERGED_CANDIDATE_POOL,
            )
        except Exception as exc:
            logger.error("Hybrid retrieval failed for query: %r", query_text, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    try:
        reranked = rerank(query_text, candidates, top_n=RERANK_TOP_N)
    except Exception as exc:
        logger.error("Reranking failed for query: %r", query_text, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reranking failed: {exc}") from exc

    # 11.14 Handling Missing Information -- no reason to build a prompt
    # (or call an LLM) if retrieval found nothing at all.
    if not has_usable_context(reranked):
        return AnswerResponse(answer=NO_CONTEXT_ANSWER, sources=[], confidence=None)

    try:
        prompt = build_prompt(query_text, reranked)
    except Exception as exc:
        logger.error(
            "Prompt construction failed for query: %r",
            query_text,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prompt construction failed: {exc}",
        ) from exc

    try:
        answer = generate_answer(prompt)
    except Exception as exc:
        logger.error(
            "Gemma inference failed for query: %r",
            query_text,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Gemma inference failed: {exc}",
        ) from exc

    return AnswerResponse(
        answer=answer,
        sources=_build_sources(reranked),
        confidence=None,
    )
