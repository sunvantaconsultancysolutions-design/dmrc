"""
bm25_index.py

Chapter 9.6 — Sparse Retrieval (BM25).

Implements lexical (keyword-based) retrieval over the SAME clause-level
chunk corpus already stored in ChromaDB by the Chapter 7/8 pipeline.

Design constraints followed from the Chapter 9 spec:
  - Does NOT duplicate embeddings. BM25 does not use the BGE-M3 vectors
    at all -- it builds its own lightweight lexical index (term
    frequencies) directly from the same `document` text and `metadata`
    already sitting in the existing ChromaDB collection.
  - Does NOT modify storage.py or metadata_loader.py. This module only
    READS from the existing collection via storage.get_collection();
    it never writes to Chroma.
  - Reuses the same metadata schema (clause_no, chapter, document_type,
    document_name, etc.) for filtering, so a filter behaves identically
    whether it's applied to the dense or the sparse retrieval path.

Why BM25 in addition to dense retrieval (see also Chapter 9.7):
  Dense (BGE-M3) retrieval is strong at semantic/paraphrase matching but
  can under-rank exact identifiers -- e.g. a query containing the literal
  clause number "6.8.2" or a rare technical term like "Cardex" is often
  better served by lexical term matching than by embedding similarity.
  BM25 complements dense retrieval for exactly these lookups.
"""

import re
import threading
from typing import Optional

from rank_bm25 import BM25Okapi

from .storage import get_collection


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Keeps alphanumerics together, INCLUDING the punctuation that appears
# inside clause numbers ("6.7.2-1") and BOQ-style identifiers, so a query
# for "6.7.2-1" tokenizes to a single term instead of being shattered into
# "6", "7", "2", "1" and losing all discriminative power. Everything else
# (whitespace, stray punctuation) is a token boundary.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-/]*")


def tokenize(text: str) -> list:
    """Lowercases and splits text into BM25 terms.

    Kept deliberately simple (no stemming/stopword removal) to mirror
    the "no stop-word removal" philosophy already established for the
    dense pipeline in text_normalization.py -- BM25's own IDF term
    already down-weights common words, so external stopword lists are
    unnecessary and only risk stripping a term a query later needs.
    """
    if not text:
        return []
    raw_tokens = _TOKEN_PATTERN.findall(text.lower())
    # Strip a trailing sentence-ending "." (e.g. "equipment." -> "equipment")
    # but leave it alone when it's part of a clause number, which always
    # ends in a digit (e.g. "6.7.2-1"), never a bare trailing period.
    return [t[:-1] if t.endswith(".") else t for t in raw_tokens]


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over the existing ChromaDB chunk corpus.

    Built once (lazily, on first use) and cached for the lifetime of the
    process. Rebuilding is cheap relative to embedding generation (no
    model inference involved) but there is no reason to redo it on every
    query, so `get_bm25_index()` below memoizes a single instance.
    """

    def __init__(self, chunk_ids: list, documents: list, metadatas: list):
        self.chunk_ids = chunk_ids
        self.documents = documents
        self.metadatas = metadatas

        # One tokenized document per corpus entry, in the same order as
        # chunk_ids/documents/metadatas -- this alignment is what lets us
        # map a BM25 score at position i back to the right chunk.
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _matches_filter(self, metadata: dict, metadata_filter: Optional[dict]) -> bool:
        """Applies the same equality-based filter semantics as ChromaDB's
        `where` clause in query.py's dense search -- every key in the
        filter dict must match exactly for the chunk to be a candidate.
        Filtering happens BEFORE ranking (see `search`), the same way a
        Chroma `where` clause restricts the ANN search space before
        similarity ranking, so dense and sparse filtering behave
        identically from the caller's point of view.
        """
        if not metadata_filter:
            return True
        for key, value in metadata_filter.items():
            if str(metadata.get(key, "")) != str(value):
                return False
        return True

    def search(self, query: str, top_k: int = 10, metadata_filter: Optional[dict] = None) -> list:
        """Ranks the corpus against `query` using BM25 and returns the
        top_k highest-scoring chunks, restricted to those matching
        metadata_filter (if given).

        Returns a list of dicts shaped to match query.py's dense
        `search()` output (chunk_id, document, metadata, bm25_score) so
        the two result lists can be merged uniformly in hybrid_retriever.py.
        """
        tokenized_query = tokenize(query)
        # get_scores returns one float per corpus document, aligned by
        # index with self.chunk_ids/self.documents/self.metadatas.
        scores = self._bm25.get_scores(tokenized_query)

        candidates = []
        for idx, score in enumerate(scores):
            metadata = self.metadatas[idx]
            if not self._matches_filter(metadata, metadata_filter):
                continue
            candidates.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "document": self.documents[idx],
                    "metadata": metadata,
                    "bm25_score": float(score),
                }
            )

        candidates.sort(key=lambda c: c["bm25_score"], reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Lazy singleton so the corpus is only pulled from ChromaDB and the BM25
# index only built once per process, not once per query.
# ---------------------------------------------------------------------------

_bm25_index_instance: Optional[BM25Index] = None
_bm25_index_lock = threading.Lock()


def build_bm25_index() -> BM25Index:
    """Pulls the full chunk corpus (documents + metadata) out of the
    EXISTING ChromaDB collection and builds a fresh BM25Index from it.

    This is the only place this module talks to ChromaDB, and it only
    ever reads (`collection.get`) -- it never upserts, so it cannot
    corrupt or duplicate anything storage.py already wrote.
    """
    collection = get_collection()
    # include=["documents", "metadatas"] is enough; embeddings are not
    # needed for BM25 and skipping them avoids pulling 1024-dim vectors
    # out of the DB for no reason.
    result = collection.get(include=["documents", "metadatas"])

    chunk_ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]

    return BM25Index(chunk_ids, documents, metadatas)


def get_bm25_index() -> BM25Index:
    """Returns the process-wide BM25Index, building it on first call and
    reusing it afterwards (thread-safe double-checked locking).
    """
    global _bm25_index_instance
    if _bm25_index_instance is None:
        with _bm25_index_lock:
            if _bm25_index_instance is None:
                _bm25_index_instance = build_bm25_index()
    return _bm25_index_instance


def rebuild_bm25_index() -> BM25Index:
    """BUGFIX: force a fresh BM25Index from the current ChromaDB
    collection, replacing the cached singleton.

    get_bm25_index() only ever builds once per process and never
    notices new chunks written by an ingestion script afterwards --
    dense search hits ChromaDB live on every call so it sees new data
    immediately, but BM25 silently keeps searching a stale snapshot
    until the process restarts. Call this once after any ingestion run
    (e.g. after BOQ rows are added) so BM25 and dense retrieval stay in
    sync. Also exposed as POST /admin/reload-bm25 in app.py.
    """
    global _bm25_index_instance
    with _bm25_index_lock:
        _bm25_index_instance = build_bm25_index()
    return _bm25_index_instance


if __name__ == "__main__":
    # Minimal manual smoke test: build the index against the real
    # collection and run one query.
    index = get_bm25_index()
    print(f"BM25 index built over {len(index.chunk_ids)} chunks.")
    for result in index.search("spares list", top_k=5):
        print(f"  {result['chunk_id']}  score={result['bm25_score']:.3f}  "
              f"clause_no={result['metadata'].get('clause_no')}")
