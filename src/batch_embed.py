"""
batch_embed.py

Batch encoding of clause chunks using BGE-M3. Covers Chapter 7.10
(Batch Embedding Generation) of the Software Design Report.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME)


def embed_batch(texts: list, batch_size: int = 32):
    """Encodes a list of normalized clause strings in batches.

    Batching amortizes fixed per-call overhead (Python <-> tensor
    dispatch, padding-mask construction) across many chunks in a single
    forward pass, and lets the GPU/CPU process a full batch matrix
    instead of one 1xD vector at a time -- this is the dominant factor
    in throughput for transformer encoders.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


if __name__ == "__main__":
    sample_texts = [
        "Clause 1.4 | Relevant Documents\nThis Specification should be "
        "read in conjunction with the General Conditions of Contract "
        "(GCC), the Special Conditions of Contract (SCC).",
        "Clause 1.5 | Verification and validation of design\nAlthough "
        "responsibility for the design service of the Works lies with "
        "the Detailed Design Consultants (DDC), the Contractor shall "
        "satisfy himself of the tentative capacities.",
        "Clause 2.1 | Scope of work\nThe detailed scope of work for "
        "ENVIRONMENT CONTROL SYSTEM (ECS) is described in the "
        "specification.",
    ]
    vectors = embed_batch(sample_texts, batch_size=32)
    print(f"Encoded {len(vectors)} chunks, dimension {vectors.shape[1]}")
