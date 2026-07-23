"""
query.py

Semantic similarity search over the existing DMRC ChromaDB collection
using the already-trained BAAI/bge-m3 embedding model. Read-only:
this script only queries the vector store populated by the Chapter 7
embedding pipeline, it never writes to it. Covers Chapter 8.10
(Similarity Search) and 8.11 (Metadata Filtering).
"""

import argparse
import re

from sentence_transformers import SentenceTransformer

from .storage import get_collection

MODEL_NAME = "BAAI/bge-m3"

EXAMPLE_QUERIES = [
    "scope of work",
    "testing",
    "contractor obligations",
    "maintenance",
    "commissioning",
    "payment",
]

_model = None


def get_model():
    """Lazily load and cache the BGE-M3 embedding model (loaded once per run)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(query: str):
    """Convert a user query into a normalized dense embedding vector."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def search(query: str, top_k: int = 5, metadata_filter: dict = None):
    """
    Perform semantic similarity search over the ChromaDB collection.

    Parameters
    ----------
    query : str
        The natural language question to search for.
    top_k : int
        Number of top results to return (default 5).
    metadata_filter : dict, optional
        ChromaDB ``where`` filter, e.g. {"clause_no": "6.8.2"} or
        {"chapter": "Chapter 3"}. If None, search runs over the whole
        collection with no metadata restriction.

    Returns
    -------
    list[dict]
        One entry per result: chunk_id, document text, metadata, and
        a similarity_score plus the raw distance.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    results = collection.query(**query_kwargs)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    formatted = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        # storage.py stores normalized embeddings; ChromaDB's default
        # "l2" space returns squared L2 distance, so for unit vectors
        # cosine_similarity = 1 - (squared_l2_distance / 2).
        similarity_score = 1 - (distance / 2)
        formatted.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "distance": round(distance, 4),
                "similarity_score": round(similarity_score, 4),
            }
        )

    return formatted


# ---------------------------------------------------------------------------
# NEW -- Exact Clause-Number Fast Path
#
# Problem: dense retrieval (semantic similarity on chunk TEXT) and BM25
# (lexical term matching on chunk TEXT) both rank candidates by how well
# the query's *wording* matches the chunk's wording -- neither one does an
# exact lookup against the `clause_no` metadata field. So a query like
# "Explain Clause 1.2.1" can fail to rank the 1.2.1 chunk first, even
# though some chunk's metadata says clause_no == "1.2.1" verbatim.
#
# Fix: before hybrid retrieval runs at all, check whether the query names
# an explicit clause number. If it does, do a metadata-only ChromaDB
# lookup (collection.get(where=...), NOT collection.query()) -- this
# needs no embedding, no BM25, and always finds the clause if it exists.
# This does not touch embeddings, bm25_index.py, hybrid_retriever.py, or
# metadata_loader.py. If no exact match is found, the caller (app.py)
# falls back to the existing hybrid_search() pipeline unchanged.
# ---------------------------------------------------------------------------

# Matches clause numbers like "1.2.1", "6.8.2", "6.7.2-1", "6.7.2.1":
# one or more digits followed by 1-4 more separator+digits groups, where
# the separator can be "." or "-" since users type both forms naturally
# (the metadata loader's normalized form uses "." but the as-authored
# clause label may use "-"). Requires at least one separator so a bare
# number ("Section 5", "page 3") is never mistaken for a clause number.
CLAUSE_NO_PATTERN = re.compile(r"\b(\d+(?:[.-]\d+){1,4})\b")


def extract_clause_no(query_text: str):
    """Return the first clause-number-shaped token found in `query_text`,
    exactly as typed (e.g. "6.7.2-1" out of "Explain Clause 6.7.2-1", or
    "6.7.2.1" out of "Explain Clause 6.7.2.1"), or None if the query
    doesn't contain one. Accepts both "." and "-" separators. Pure string
    matching -- no model call, no DB call, no normalization here (that
    happens in get_chunk_by_clause_no, only if the as-typed form doesn't
    match anything).
    """
    match = CLAUSE_NO_PATTERN.search(query_text)
    return match.group(1) if match else None


def get_chunk_by_clause_no(clause_no: str):
    """Exact metadata lookup for a clause number, bypassing dense/BM25
    retrieval entirely.

    Users may type a clause number with either separator (e.g.
    "6.7.2-1" or "6.7.2.1"), and the metadata loader may store the
    as-authored form under `clause_no` (e.g. "6.7.2-1") while keeping a
    dot-normalized form under `clause_no_normalized` (e.g. "6.7.2.1").
    To support both input styles without touching metadata_loader.py or
    re-ingesting any data, this tries two lookups in order:

        1. where={"clause_no": clause_no}                   (as typed)
        2. where={"clause_no_normalized": normalized_clause} ("-" -> ".")

    Only falls through to the second lookup if the first finds nothing,
    so a corpus that only ever populates `clause_no` (never
    `clause_no_normalized`) still works unchanged.

    Uses collection.get(where=...) rather than collection.query(...):
    .get() is ChromaDB's metadata-only accessor (no query embedding, no
    ANN search), which is exactly what an exact clause match needs.

    Returns a list of candidate dicts shaped to match
    hybrid_retriever.merge_candidates()'s output (chunk_id, document,
    metadata, score, retrieval_source, dense_score, bm25_score) so it can
    be fed straight into the existing rerank() / build_prompt() /
    _build_sources() code with no changes to any of them. Returns []
    if no chunk carries this clause number under either field, so the
    caller falls back to hybrid_search() exactly as before.
    """
    collection = get_collection()

    result = collection.get(
        where={"clause_no": clause_no},
        include=["documents", "metadatas"],
    )

    if not result.get("ids"):
        normalized_clause_no = clause_no.replace("-", ".")
        result = collection.get(
            where={"clause_no_normalized": normalized_clause_no},
            include=["documents", "metadatas"],
        )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    candidates = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "score": 1.0,  # exact metadata match -- highest possible confidence
                "retrieval_source": "exact_clause_match",
                "dense_score": None,
                "bm25_score": None,
            }
        )
    return candidates


def print_results(query: str, results: list):
    """Pretty-print search results to the console."""
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"\n[{rank}] similarity={result['similarity_score']} "
            f"(distance={result['distance']})  chunk_id={result['chunk_id']}"
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


def build_filter(filter_arg: str):
    """Parse a single ``key=value`` CLI filter into a ChromaDB ``where`` dict."""
    if not filter_arg:
        return None
    key, sep, value = filter_arg.partition("=")
    if not sep or not key:
        raise ValueError(
            "Filter must be in the form key=value, e.g. --filter clause_no=6.8.2"
        )
    if value.isdigit():
        value = int(value)
    return {key: value}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic similarity search over the DMRC ChromaDB collection."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural language query. If omitted, runs the built-in example queries.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help=(
            "Optional metadata filter as key=value, e.g. "
            "--filter clause_no=6.8.2, --filter chapter='Chapter 3', "
            "--filter approval_status=approved, --filter pdf_page=5"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_filter = build_filter(args.filter)

    if args.query:
        results = search(args.query, top_k=args.top_k, metadata_filter=metadata_filter)
        print_results(args.query, results)
    else:
        print("No query supplied - running example queries:\n")
        for example_query in EXAMPLE_QUERIES:
            results = search(example_query, top_k=args.top_k, metadata_filter=metadata_filter)
            print_results(example_query, results)


if __name__ == "__main__":
    main()
