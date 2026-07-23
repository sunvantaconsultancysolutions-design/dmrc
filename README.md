# DMRC Contract Intelligence — Embedding Generation Module (Chapter 7)

Local, offline embedding generation pipeline using **BAAI/bge-m3** over the
finalized Chapter 6 metadata schema and clause-level chunks. No retrieval or
LLM prompting is implemented here — this module only produces and stores
embeddings.

## Folder structure

```
dmrc-embedding-pipeline/
├── README.md
├── requirements.txt
├── main.py                     # pipeline entry point
├── data/                       # parsed contract JSON (chunked, Chapter 6 output)
│   ├── DMRC_Chapter1_transcription.json
│   ├── DMRC_Chapter2_transcription.json
│   └── chapter3.json
├── src/
│   ├── text_normalization.py   # 7.6 — normalization, embedding-input builder
│   ├── metadata_loader.py      # 7.11 — finalized schema mapping (no redesign)
│   ├── embed_single.py         # 7.9  — single-chunk BGE-M3 encoding
│   ├── batch_embed.py          # 7.10 — batched BGE-M3 encoding
│   └── storage.py              # 7.12 — ChromaDB persistence
└── chroma_db/                  # created on first run (persistent local vector store)
```

## Prerequisites

- Python 3.10–3.11
- ~6 GB free disk (BGE-M3 checkpoint + ChromaDB)
- Optional: NVIDIA GPU + CUDA for faster encoding (falls back to CPU automatically)

## Setup

```bash
# 1. Unzip / cd into the project
cd dmrc-embedding-pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first run of `main.py` will download the `BAAI/bge-m3` checkpoint
(~2.27 GB) from Hugging Face into your local HF cache
(`~/.cache/huggingface`). No API key is required — the model runs fully
locally after this one-time download.

## Run

Embed a single file:

```bash
python main.py --input data/DMRC_Chapter1_transcription.json
```

Embed all three files in one pass:

```bash
python main.py --input-dir data/
```

Expected output:

```
[DMRC_Chapter1_transcription.json] 14 chunks parsed, 9 eligible for embedding
(5 filtered: cover/index/marginalia).
Batches: 100%|████████████| 1/1 [00:02<00:00, 2.31s/it]
[DMRC_Chapter1_transcription.json] Stored 9 embeddings to ChromaDB.
...
Done. 41 clause embeddings written to ./chroma_db
```

## Verify

```bash
python -c "
from src.storage import get_collection
c = get_collection()
print('Total vectors stored:', c.count())
print(c.peek(1))
"
```

## Sanity-check a single embedding

```bash
python src/embed_single.py
```

```
Embedding dimension: 1024
L2 norm (should be ~1.0): 1.000000
```

## Notes

- `chroma_db/` is a local persistent directory — delete it to reset the
  vector store and re-embed from scratch.
- Switching to Qdrant later only requires replacing `src/storage.py`;
  `main.py`, `metadata_loader.py`, and the embedding modules are unchanged.
- The metadata schema is consumed as-is from Chapter 6
  (`DMRC_Unified_Metadata_Schema.md`) — this module does not modify it.
