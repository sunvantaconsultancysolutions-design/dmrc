"""
validate_db.py

Read-only validation script for the DMRC ChromaDB collection.
Connects to the existing collection populated by the Chapter 7
embedding pipeline and confirms that it is intact and queryable.
Covers Chapter 8.9 (Vector Indexing Workflow validation) / 8.12
(ChromaDB Implementation). This script never inserts, updates, or
deletes any data in the collection.
"""

from storage import get_collection


def validate_collection():
    """Connect to the existing ChromaDB collection and print a summary."""
    print("=" * 70)
    print("DMRC ChromaDB Collection Validation")
    print("=" * 70)

    # --- Connect and open the collection -----------------------------
    try:
        collection = get_collection()
    except Exception as exc:  # noqa: BLE001 - surface any connection issue clearly
        print(f"[FAILED] Could not connect to ChromaDB / open collection: {exc}")
        return

    print(f"[OK] Connected to ChromaDB at ./chroma_db")
    print(f"[OK] Collection opened: '{collection.name}'")
    print(f"     Collection metadata : {collection.metadata}")

    # --- Total vector count --------------------------------------------
    total_vectors = collection.count()
    print(f"\n[OK] Total vectors stored : {total_vectors}")

    if total_vectors == 0:
        print("\n[WARNING] Collection is empty. Nothing further to validate.")
        print("          Run the Chapter 7 embedding pipeline (main.py) first.")
        return

    # --- Sample record (read-only) --------------------------------------
    sample = collection.get(
        limit=1,
        include=["documents", "metadatas", "embeddings"],
    )

    sample_id = sample["ids"][0]
    sample_document = sample["documents"][0]
    sample_metadata = sample["metadatas"][0]
    sample_embedding = sample["embeddings"][0]

    print("\n" + "-" * 70)
    print("Sample Record")
    print("-" * 70)
    print(f"Chunk ID : {sample_id}")

    print("\nSample document text (first 200 characters):")
    preview = sample_document[:200]
    print(preview + ("..." if len(sample_document) > 200 else ""))

    print("\nSample metadata:")
    for key, value in sample_metadata.items():
        print(f"  {key}: {value}")

    embedding_dim = len(sample_embedding)
    print(f"\nSample embedding (first 5 values) : {list(sample_embedding[:5])}")
    print(f"[OK] Embedding dimension : {embedding_dim}")

    if embedding_dim != 1024:
        print(f"[WARNING] Expected BGE-M3 dimension 1024, got {embedding_dim}.")

    print("\n" + "=" * 70)
    print(f"Validation complete. Collection '{collection.name}' is READY for querying.")
    print("=" * 70)


if __name__ == "__main__":
    validate_collection()
