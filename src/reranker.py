"""
reranker.py

Chapter 10 -- BGE Reranker.

------------------------------------------------------------------------
10.1  Role of the Reranker in the Pipeline
------------------------------------------------------------------------
This module sits directly after Chapter 9's Hybrid Retrieval stage:

    User Query
        |
        v
    Hybrid Retrieval (dense + BM25, merged -- Chapter 9)
        |
        v
    BGE Reranker  (this module)
        |
        v
    Top-N re-ranked chunks --> Prompt Builder (Chapter 11)

Hybrid Retrieval's merge step (Chapter 9.9) compares dense similarity
and BM25 scores only after independently min-max normalizing each list
-- a necessary heuristic for combining two incompatible scales, but it
never lets the query and a candidate chunk interact directly.

The BGE reranker (BAAI/bge-reranker-v2-m3) is a cross-encoder: it takes
the full (query, document) pair as a single joint input and produces
one calibrated relevance score per pair. This is far more accurate than
comparing independently-computed dense/BM25 scores, but also far more
expensive (full attention over query+document for every pair), which is
why it is only ever run over the small merged candidate pool produced
by Hybrid Retrieval (e.g. ~40 candidates), never over the full corpus.

This module intentionally does NOT call ChromaDB, BM25, or the dense
embedding model directly -- it is a pure post-processing step over
whatever candidate list Chapter 9's `hybrid_search()` (or
`merge_candidates()`) already produced. Each input candidate is
expected to be a dict shaped like Chapter 9's merged output, i.e. it
must contain at least:

    {
        "chunk_id": ...,
        "document": ...,   # the chunk text
        "metadata": ...,   # unchanged, passed through as-is
        "score": ...,               # Chapter 9's fused score
        "retrieval_source": ...,    # "dense" / "sparse" / "dense+sparse"
        "dense_score": ...,
        "bm25_score": ...,
    }

Every field already present on a candidate is preserved unchanged; this
module only adds one new field, `reranker_score`, so the output remains
a drop-in replacement for Chapter 9's output when handed to Chapter
11's Prompt Builder.
"""

import argparse
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_N = 10       # how many re-ranked results are returned to the caller
DEFAULT_BATCH_SIZE = 16  # matches Section 14.7's recommended reranker batch size
DEFAULT_MAX_LENGTH = 512  # matches Section 5.11's max token budget for a chunk

_tokenizer = None
_model = None
_device = None


# ---------------------------------------------------------------------------
# 10.2  Model Loading
# ---------------------------------------------------------------------------

def get_reranker_model():
    """Lazily load and cache the BGE reranker model + tokenizer (loaded
    once per process, mirroring query.py's get_model() pattern for the
    dense embedding model).
    """
    global _tokenizer, _model, _device
    if _model is None:
        print(f"Loading reranker model: {MODEL_NAME} ...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


# ---------------------------------------------------------------------------
# 10.3  Pairwise Scoring
# ---------------------------------------------------------------------------

def _score_pairs(
    query: str,
    documents: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list:
    """Builds (query, document) pairs and computes one relevance score
    per pair using the cross-encoder, processed in batches for GPU
    utilization (Section 14.7).

    Returns a list of floats in [0, 1] (sigmoid of the model's raw
    relevance logit -- BGE reranker's standard scoring convention),
    same length and order as `documents`.
    """
    tokenizer, model, device = get_reranker_model()

    scores = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits.view(-1).float()
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            scores.extend(batch_scores)

    return scores


# ---------------------------------------------------------------------------
# 10.4  Reranking Entry Point
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list,
    top_n: int = DEFAULT_TOP_N,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list:
    """Re-ranks Chapter 9's merged hybrid-retrieval candidates using the
    BGE cross-encoder.

    Parameters
    ----------
    query : str
        The original user query (same string passed into
        hybrid_search()).
    candidates : list[dict]
        Chapter 9's merged candidate list (output of
        hybrid_retriever.hybrid_search() or merge_candidates()). Each
        dict must contain "document"; every other field (chunk_id,
        metadata, score, retrieval_source, dense_score, bm25_score,
        ...) is preserved unchanged on the output.
    top_n : int
        Number of top re-ranked candidates to return (default 10, per
        Chapter 10's spec / Section 14.7's "Re-ranked Documents" = 10).
    batch_size : int
        Cross-encoder batch size (default 16, per Section 14.7).
    max_length : int
        Max token length per (query, document) pair (default 512, per
        Section 5.11's chunk token budget).

    Returns
    -------
    list[dict]
        The input candidates, each with a new "reranker_score" field
        (float, [0, 1]), sorted descending by reranker_score, truncated
        to `top_n`. All existing fields/metadata are preserved as-is.
    """
    if not candidates:
        return []

    documents = [c["document"] for c in candidates]
    scores = _score_pairs(query, documents, batch_size=batch_size, max_length=max_length)

    reranked = []
    for candidate, score in zip(candidates, scores):
        entry = dict(candidate)  # shallow copy -- preserve every existing field/metadata
        entry["reranker_score"] = round(float(score), 4)
        reranked.append(entry)

    reranked.sort(key=lambda c: c["reranker_score"], reverse=True)
    return reranked[:top_n]


# ---------------------------------------------------------------------------
# CLI -- mirrors query.py / hybrid_retriever.py's CLI shape for a
# consistent developer experience. Only imports hybrid_search lazily,
# inside main(), so this module has no hard dependency on the retriever
# package layout and can be unit-tested with plain candidate dicts.
# ---------------------------------------------------------------------------

def print_reranked_results(query: str, results: list) -> None:
    """Pretty-prints reranked results, including the original
    retrieval_source and both underlying scores alongside the new
    reranker_score -- useful for manually validating that reranking is
    actually reordering the merged candidates sensibly.
    """
    print("=" * 70)
    print(f"Reranked Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] reranker_score={result['reranker_score']:.4f}  "
            f"fused_score={result.get('score')}  "
            f"source={result.get('retrieval_source')}  "
            f"chunk_id={result['chunk_id']}"
        )
        print(
            f"    clause_no={metadata.get('clause_no', 'N/A')}  "
            f"heading={metadata.get('heading', 'N/A')}  "
            f"pdf_page={metadata.get('pdf_page', 'N/A')}  "
            f"document_name={metadata.get('document_name', 'N/A')}"
        )
        text_preview = result["document"][:300]
        print(f"    text: {text_preview}{'...' if len(result['document']) > 300 else ''}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="BGE cross-encoder reranking over Chapter 9's hybrid retrieval candidates."
    )
    parser.add_argument("query", type=str, help="Natural language or clause-number query.")
    parser.add_argument("--top_k_dense", type=int, default=10)
    parser.add_argument("--top_k_bm25", type=int, default=10)
    parser.add_argument(
        "--merged_top_k",
        type=int,
        default=40,
        help="Size of the merged candidate pool handed to the reranker (Section 14.7).",
    )
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N, help="Final reranked result count.")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional metadata filter as key=value, e.g. --filter clause_no=6.8.2",
    )
    return parser.parse_args()


def main():
    # Imported here (not at module scope) so reranker.py has no hard
    # dependency on the retriever package's import layout, per the
    # instruction not to modify hybrid_retriever.py.
    from .hybrid_retriever import hybrid_search
    from .query import build_filter

    args = parse_args()
    metadata_filter = build_filter(args.filter)

    candidates = hybrid_search(
        args.query,
        top_k_dense=args.top_k_dense,
        top_k_bm25=args.top_k_bm25,
        final_top_k=args.merged_top_k,
        metadata_filter=metadata_filter,
    )

    results = rerank(args.query, candidates, top_n=args.top_n)
    print_reranked_results(args.query, results)


if __name__ == "__main__":
    main()
