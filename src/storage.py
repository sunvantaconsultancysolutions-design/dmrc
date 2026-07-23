"""
storage.py

Persists embedding vectors + finalized metadata + chunk text into a
local ChromaDB collection. Covers Chapter 7.12 (Embedding Storage).
No retrieval logic lives here -- this module only writes.
"""

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "dmrc_be12be14_ecs"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": "BAAI/bge-m3",
            "chunking_strategy": "clause-level",
        },
    )
    return collection


def sanitize_metadata(metadata: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- lists and
    nested objects (e.g. stamps[]) are JSON-encoded to strings.
    """
    import json
    clean = {}
    for k, v in metadata.items():
        if isinstance(v, (list, dict)):
            clean[k] = json.dumps(v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = v
    return clean


def store_chunks(chunk_ids, texts, embeddings, metadatas):
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        metadatas=[sanitize_metadata(m) for m in metadatas],
    )
    return len(chunk_ids)
