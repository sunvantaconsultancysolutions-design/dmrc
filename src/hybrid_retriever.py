"""
hybrid_retriever.py

Chapter 9.4 / 9.8 / 9.9 / 9.10 / 9.12 -- Hybrid Retrieval Architecture,
Retrieval Workflow, Candidate Merging, Metadata Filtering, and
Retrieval Parameters.

------------------------------------------------------------------------
9.4  Hybrid Retrieval Architecture (documentation)
------------------------------------------------------------------------
This module is the single orchestration point that combines the two
independent retrieval paths already available in this project:

    - Dense retrieval  (Chapter 8, reused as-is): query.py's `search()`,
      which embeds the query with BGE-M3 and does ANN similarity search
      against the existing ChromaDB collection.
    - Sparse retrieval (Chapter 9.6, new): bm25_index.py's `BM25Index`,
      which does lexical term-matching over the same corpus of chunk
      text, read directly out of the same ChromaDB collection.

Neither retrieval path is replaced or weakened by the other -- both run
independently against the same underlying chunk corpus and the same
metadata schema, and their results are only combined at the very end
(see 9.9 Candidate Merging below). This keeps the two retrieval
mechanisms decoupled and independently testable/tunable, which is why
they live in separate modules (bm25_index.py, hybrid_retriever.py)
instead of being interleaved into one function.

------------------------------------------------------------------------
9.7  Dense vs Sparse Retrieval (documentation only, no code)
------------------------------------------------------------------------
Dense retrieval (BGE-M3 + ChromaDB):
  + Captures semantic/paraphrase similarity ("contractor obligations"
    matches a clause about "responsibilities of the Contractor" even
    with no shared words).
  + Robust to synonyms, rephrasing, and cross-lingual-style variation.
  - Can under-rank exact identifiers: a query containing a literal
    clause number ("6.8.2"), a rare part name, or a specific numeric
    value competes for meaning in embedding space against much more
    "semantically typical" chunks, and can lose to a chunk that's
    topically similar but doesn't contain the exact token the user
    actually wanted.
  - Opaque: harder to explain *why* a given chunk ranked where it did.

Sparse retrieval (BM25):
  + Excellent at exact / near-exact term matches -- clause numbers,
    BOQ identifiers, proper nouns, acronyms (e.g. "ECS", "DLP", "SAT").
  + Fast to build, fully explainable (term frequency + IDF), and needs
    no GPU/model inference.
  - Blind to meaning: "contractor duties" will not match a clause about
    "responsibilities of the Contractor" unless the words overlap.
  - Sensitive to exact phrasing/vocabulary choice.

Together: dense retrieval provides recall on meaning, BM25 provides
precision on exact terms -- Chapter 9.9's merge step is what lets a
single query benefit from both simultaneously instead of picking one.

------------------------------------------------------------------------
9.13 Advantages of Hybrid Retrieval (documentation only, no code)
------------------------------------------------------------------------
  - Higher recall than either method alone: a query that dense retrieval
    misses because of vocabulary mismatch can still be caught by BM25
    (and vice versa).
  - Robustness to query style: users who paste an exact clause number
    ("show me 6.10.2") and users who ask a natural-language question
    ("what does the contractor need to submit?") are both served well
    by the same retriever, without the caller needing to know which
    style of query they're issuing.
  - No additional embedding cost: BM25 is built from text already
    embedded and stored -- there is no new encoding step, no GPU work,
    and no duplicate vector storage (see bm25_index.py docstring).
  - Graceful degradation: if the embedding model or ChromaDB ANN index
    were ever unavailable, BM25 alone can still serve lexical queries
    (and symmetrically for dense retrieval), since the two paths do not
    depend on each other.
"""

import argparse
from typing import Optional

from .bm25_index import get_bm25_index
from .query import search as dense_search, build_filter, get_model  # noqa: F401  (get_model re-exported for CLI warm-up parity with query.py)


# ---------------------------------------------------------------------------
# 9.12 Retrieval Parameters -- configurable, overridable per call.
# ---------------------------------------------------------------------------

TOP_K_DENSE = 10   # how many candidates dense retrieval contributes before merging
TOP_K_BM25 = 10    # how many candidates BM25 retrieval contributes before merging
FINAL_TOP_K = 5    # size of the final merged result list returned to the caller


# ---------------------------------------------------------------------------
# 9.9 Candidate Merging
# ---------------------------------------------------------------------------

def _normalize_scores(results: list, score_key: str) -> list:
    """Min-max normalizes `score_key` across `results` to the [0, 1]
    range, writing the result into a new "normalized_score" field on
    each dict (the original score is left untouched).

    Why this is necessary: dense similarity_score is already in [0, 1]
    (cosine similarity of L2-normalized vectors), but BM25's bm25_score
    is an unbounded, corpus-dependent value that can be far outside that
    range. Comparing them directly ("keep highest score for duplicates")
    would always favor whichever scale happens to produce bigger raw
    numbers rather than whichever result is actually the better match.
    Normalizing each list independently onto a common [0, 1] scale
    before merging is the standard way to make "highest score wins"
    meaningful across two retrieval methods with different scoring
    functions.
    """
    if not results:
        return results

    scores = [r[score_key] for r in results]
    lo, hi = min(scores), max(scores)
    spread = hi - lo

    for r in results:
        if spread == 0:
            # All candidates tied (e.g. a single result, or a query with
            # zero BM25 term overlap for every candidate) -- treat them
            # as equally strong rather than dividing by zero.
            r["normalized_score"] = 1.0
        else:
            r["normalized_score"] = (r[score_key] - lo) / spread

    return results


def merge_candidates(dense_results: list, sparse_results: list, final_top_k: int = FINAL_TOP_K) -> list:
    """Implements Chapter 9.9's merge rules exactly:

      1. Merge dense and sparse result lists.
      2. Remove duplicate chunk_ids.
      3. For a chunk_id present in both lists, keep the higher score.
      4. Return the Top-K merged results.

    "Score" is compared on the normalized_score computed by
    `_normalize_scores` (see that function's docstring for why raw
    dense/BM25 scores cannot be compared directly). Each output entry
    also carries a `retrieval_source` field ("dense", "sparse", or
    "dense+sparse") so callers/UI can show why a result was surfaced.
    """
    dense_results = _normalize_scores(list(dense_results), "similarity_score")
    sparse_results = _normalize_scores(list(sparse_results), "bm25_score")

    merged: dict = {}  # chunk_id -> merged candidate dict

    for r in dense_results:
        merged[r["chunk_id"]] = {
            "chunk_id": r["chunk_id"],
            "document": r["document"],
            "metadata": r["metadata"],
            "score": r["normalized_score"],
            "retrieval_source": "dense",
            "dense_score": r["similarity_score"],
            "bm25_score": None,
        }

    for r in sparse_results:
        existing = merged.get(r["chunk_id"])
        if existing is None:
            merged[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": r["normalized_score"],
                "retrieval_source": "sparse",
                "dense_score": None,
                "bm25_score": r["bm25_score"],
            }
        else:
            # Duplicate chunk_id found in both lists: keep the higher
            # score, and record that it was surfaced by both methods.
            existing["retrieval_source"] = "dense+sparse"
            existing["bm25_score"] = r["bm25_score"]
            if r["normalized_score"] > existing["score"]:
                existing["score"] = r["normalized_score"]

    merged_list = list(merged.values())
    merged_list.sort(key=lambda c: c["score"], reverse=True)
    return merged_list[:final_top_k]


# ---------------------------------------------------------------------------
# 9.8 Retrieval Workflow
#
#     User Query
#         |
#         v
#     Dense Retrieval (ChromaDB)  --\
#                                     >--  Candidate Merge  --> Top-K merged candidates
#     BM25 Retrieval              --/
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k_dense: int = TOP_K_DENSE,
    top_k_bm25: int = TOP_K_BM25,
    final_top_k: int = FINAL_TOP_K,
    metadata_filter: Optional[dict] = None,
) -> list:
    """Runs the full Chapter 9.8 pipeline for a single query:

        1. Dense retrieval over ChromaDB (query.py's existing search()).
        2. BM25 retrieval over the same corpus (bm25_index.py).
        3. Candidate merge (see merge_candidates above).

    metadata_filter (9.10): the SAME filter dict is passed to both
    retrieval calls, so a filter like {"clause_no": "6.8.2"} or
    {"chapter": "Chapter 3"} restricts the candidate pool identically
    for dense and sparse retrieval -- neither path can return a chunk
    the other path would have excluded.
    """
    dense_results = dense_search(query, top_k=top_k_dense, metadata_filter=metadata_filter)

    bm25_index = get_bm25_index()
    sparse_results = bm25_index.search(query, top_k=top_k_bm25, metadata_filter=metadata_filter)

    return merge_candidates(dense_results, sparse_results, final_top_k=final_top_k)


# ---------------------------------------------------------------------------
# CLI -- mirrors query.py's CLI shape for a consistent developer experience.
# ---------------------------------------------------------------------------

def print_hybrid_results(query: str, results: list) -> None:
    """Pretty-prints hybrid results, including which retrieval path(s)
    surfaced each result -- useful for validating 9.9's merge behavior
    by eye during manual testing.
    """
    print("=" * 70)
    print(f"Hybrid Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] score={result['score']:.4f}  "
            f"source={result['retrieval_source']}  "
            f"chunk_id={result['chunk_id']}"
        )
        print(
            f"    clause_no={metadata.get('clause_no', 'N/A')}  "
            f"heading={metadata.get('heading', 'N/A')}  "
            f"pdf_page={metadata.get('pdf_page', 'N/A')}  "
            f"document_name={metadata.get('document_name', 'N/A')}"
        )
        if result["dense_score"] is not None:
            print(f"    dense_similarity={result['dense_score']:.4f}", end="  ")
        if result["bm25_score"] is not None:
            print(f"bm25_score={result['bm25_score']:.4f}", end="")
        print()
        text_preview = result["document"][:300]
        print(f"    text: {text_preview}{'...' if len(result['document']) > 300 else ''}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid (dense + BM25) retrieval over the DMRC ChromaDB collection."
    )
    parser.add_argument("query", type=str, help="Natural language or clause-number query.")
    parser.add_argument("--top_k_dense", type=int, default=TOP_K_DENSE)
    parser.add_argument("--top_k_bm25", type=int, default=TOP_K_BM25)
    parser.add_argument("--final_top_k", type=int, default=FINAL_TOP_K)
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional metadata filter as key=value, e.g. --filter clause_no=6.8.2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_filter = build_filter(args.filter)
    results = hybrid_search(
        args.query,
        top_k_dense=args.top_k_dense,
        top_k_bm25=args.top_k_bm25,
        final_top_k=args.final_top_k,
        metadata_filter=metadata_filter,
    )
    print_hybrid_results(args.query, results)


if __name__ == "__main__":
    main()
