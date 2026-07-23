"""
embed_single.py

Loads BAAI/bge-m3 locally via sentence-transformers and encodes a single
clause chunk into a dense embedding vector. This module covers Chapter
7.9 (Local Python Implementation) of the Software Design Report.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

# Loaded once per process. Loading downloads (or reads from local HF
# cache) the ~2.27GB BGE-M3 checkpoint and initializes it on GPU if
# available, else CPU.
model = SentenceTransformer(MODEL_NAME)


def embed_chunk(text: str):
    """Encodes a single normalized clause string into a dense vector.

    normalize_embeddings=True L2-normalizes the output vector so that
    a plain dot product between two embeddings is equivalent to cosine
    similarity -- this is a storage/encoding-time property, independent
    of how retrieval later uses the vectors.
    """
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


if __name__ == "__main__":
    sample_text = (
        "Clause 2.1 | Scope of work | Contract BE-12 LOT3 & BE-14 LOT3 "
        "Scope of work - ECS\n"
        "The detailed scope of work for ENVIRONMENT CONTROL SYSTEM (ECS) "
        "is described in the specification."
    )
    vector = embed_chunk(sample_text)
    print(f"Embedding dimension: {vector.shape[0]}")
    print(f"L2 norm (should be ~1.0): {(vector ** 2).sum() ** 0.5:.6f}")
