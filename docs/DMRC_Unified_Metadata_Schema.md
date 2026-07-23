# Unified Metadata Schema — DMRC BE-12 LOT3 & BE-14 LOT3 Contract
### For LangChain + BGE-M3 + ChromaDB/Qdrant RAG Pipeline

---

## 1. Source Structure Analysis

All three files share one JSON shape:

```
{
  "pages": [
    {
      "pdf_page": int,
      "printed_page": str,        // often "" in Chapter 1, populated "0000NN" elsewhere
      "section_heading": str,     // often "" on cover/index pages
      "clauses": [
        {
          "clause_no": str,       // "", "6.7.1", "6.7.2-1", "6.10.2" etc. — hierarchical, inconsistent delimiters
          "heading": str,         // "", "General", "STAMP", "COVER PAGE", "MARGINALIA", "INDEX"
          "text": str,
          "continues_previous": bool,
          "low_confidence": bool  // OPTIONAL — only present on some clauses
        }
      ]
    }
  ]
}
```

Key observations that drive the schema design:

- **No explicit project/document/volume fields exist in the JSON.** Project name, contract number, employer, contractor, and volume number only appear as free text inside `clauses[].text` on cover pages (e.g. "VOLUME 2", "VOLUME 3(I)"). These must be extracted once per document during preprocessing and then **propagated as constants** to every chunk — they are not per-page/per-clause fields.
- **`heading` is overloaded.** It's used both for real clause titles ("General", "Penalty Clause") and for non-substantive content markers ("STAMP", "COVER PAGE", "MARGINALIA", "INDEX"). This needs to be split into a semantic `chunk_type` field so retrieval can filter out stamps/marginalia noise.
- **`clause_no` is inconsistent** ("6.7.2-1" vs "6.7.2.1" vs "6.10.2") and frequently empty (continuation text, stamps, unheaded paragraphs).
- **`continues_previous` is a genuine cross-chunk linking signal** — critical for retrieval so a chunk's context isn't orphaned from its preceding clause.
- **`low_confidence` is sparse/optional** — an OCR/transcription confidence flag, useful to down-rank or flag content for human review, but must default to `false` when absent.
- **`printed_page` is unreliable** — blank in Chapter 1 cover/front matter, populated as a zero-padded control number elsewhere. Treat as optional/nullable, not a primary key.

---

## 2. Unified Metadata Schema

### A. Project Metadata *(constant across the entire contract — same value on every chunk from all 3 files)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| project_name | string | Extracted (from cover-page text, once) | `clauses[].text` (cover page) | Full MRTS project name | "Phase-II Delhi MRTS Project – CS to Qutab Minar & CS to Badarpur Corridors" | Mandatory |
| contract_number | string | Extracted | `clauses[].text` (cover page) | Contract package identifier | "BE-12 LOT3 & BE-14 LOT3" | Mandatory |
| employer_name | string | Extracted | `clauses[].text` (cover page) | Contracting authority | "Delhi Metro Rail Corporation Ltd. (DMRC)" | Mandatory |
| contractor_name | string | Extracted | `clauses[].text` (cover page) | Executing contractor | "M/s Blue Star Limited" | Mandatory |
| system_scope | string | Extracted | `clauses[].text` (cover page) | System(s) covered by contract | "Environment Control System (ECS) & Building Management System (BMS)" | Optional |
| stations_covered | list[string] | Extracted | `clauses[].text` (cover page) | Named stations in scope | ["Hauz Khas", "Malviya Nagar", "Saket", "Central Secretariat", "Khan Market", "JLN Stadium", "Jungpura"] | Optional |

### B. Document Metadata *(constant per source file, differs across the 3 files)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| document_id | string | Generated | — | Stable unique ID for the source document, used as a Foreign key | "DMRC-BE12BE14-VOL2-CH1" | Mandatory |
| document_name | string | Extracted/Default | filename / cover text | Human-readable document name | "Scope of Work – ECS (Chapter 1)" | Mandatory |
| document_type | string (enum) | Extracted | `section_heading` / heading text | Category of content | "Scope of Work" \| "Specification" \| "Data Sheet" | Mandatory |
| volume | string | Extracted | `clauses[].text` (cover page, e.g. "VOLUME 2") | Contract volume number | "Volume 2" | Optional |
| chapter | string | Generated (from filename) | filename | Chapter label mapped from filename | "Chapter 1" | Mandatory |
| source_file | string | Generated | filename | Original filename ingested | "DMRC_Chapter1_transcription.json" | Mandatory |
| total_pages_in_document | integer | Generated | `len(pages)` | Page count for the file | 9 | Optional |

### C. Structural Metadata *(varies per page / clause — the positional fingerprint of a chunk)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| section_heading | string | Extracted | `pages[].section_heading` | Running header/section title of the page | "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS" | Optional (often "") |
| clause_no | string | Extracted | `clauses[].clause_no` | Raw clause/sub-clause number as transcribed | "6.7.2-1" | Optional (often "") |
| clause_no_normalized | string | Generated | derived from `clause_no` | Delimiter-normalized clause number for consistent filtering | "6.7.2.1" | Optional |
| parent_clause | string | Generated | derived from `clause_no_normalized` | The immediate parent clause one level up the hierarchy — enables "show me the whole 6.7.2 family" style navigation | "6.7.2" | Optional |
| hierarchy_level | integer | Generated | derived from `clause_no_normalized` (count of segments) | Depth of the clause in the numbering hierarchy: `6`→1, `6.7`→2, `6.7.2`→3, `6.7.2.1`→4 | 4 | Optional |
| heading | string | Extracted | `clauses[].heading` | Clause title (only populated for real clauses — see `chunk_type`) | "Competency of Personnel" | Optional |
| chunk_type | string (enum) | Generated | derived from `heading`/`clause_no`/`text` | Semantic classification of the chunk's *content role* | "clause" \| "marginalia" \| "cover_page" \| "index" \| "continuation" | Mandatory |
| stamps | list[object] | Generated | extracted out of `clauses[]` where `heading == "STAMP"` | Structured stamp/seal metadata pulled off the page and attached to the nearest preceding real clause, instead of appearing as its own noisy chunk. Each object: `{organization, type, page_control, date}` | `[{"organization":"DMRC","type":"Confidential seal","page_control":"000017","date":"Feb 2008"},{"organization":"Blue Star Ltd.","type":"Company stamp"}]` | Optional (`[]` if none on page) |
| pdf_page | integer | Extracted | `pages[].pdf_page` | Physical page number in the source PDF | 1 | Mandatory |
| printed_page | string | Extracted | `pages[].printed_page` | Printed/control page number stamped on the page | "000017" | Optional (often "") |
| continues_previous | boolean | Extracted | `clauses[].continues_previous` | Whether this clause's text continues from the prior page/clause | false | Mandatory |
| low_confidence | boolean | Extracted/Default | `clauses[].low_confidence` | OCR/transcription confidence flag; defaults to `false` if key absent | true | Mandatory (defaulted) |

### D. Retrieval Metadata *(generated during preprocessing — one unique value per chunk)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| chunk_id | string | Generated | — | Globally unique chunk identifier (UUID or deterministic hash) | "DMRC-BE12BE14-VOL2-CH3-p1-c6.7.1-000" | Mandatory |
| chunk_number | integer | Generated | — | Sequential position of chunk *within its document type/section grouping* (e.g. Nth clause chunk) | 42 | Mandatory |
| document_sequence | integer | Generated | — | Absolute sequential position of the chunk across the *entire source document*, 1..N, regardless of chunk_type | 57 | Mandatory |
| chunk_hash | string | Generated | — | Content hash for deduplication/change-detection | "a91f3e2b..." | Optional |
| language | string | Default | — | Language of the chunk text (ISO 639-1) | "en" | Mandatory |

### E. Version Metadata *(governance/lifecycle — constant or slowly changing)*

| Field Name | Data Type | Source | Source JSON Field | Description | Example Value | Mandatory |
|---|---|---|---|---|---|---|
| revision | string | Default/Extracted | date stamped in "STAMP" clauses (e.g. "Feb 2008") | Contract revision/date reference | "Feb 2008" | Optional |
| approval_status | string (enum) | Default | — | Governance status of the ingested content | "draft" \| "approved" \| "superseded" | Mandatory (default "approved") |

> **Removed from per-chunk metadata:** `embedding_model`, `vector_store`, `token_count`, `ingestion_timestamp`, `ingestion_pipeline_version`, `schema_version`. These are pipeline/infrastructure properties, not properties of the content — stapling them to every one of thousands of chunks is redundant and couples the metadata to a specific backend or run. See §7A for where this information should live instead.

---

## 3. Why These Fields Matter for Retrieval

- **`chunk_type`** is the single highest-value derived field. Without it, "COVER PAGE," "INDEX," and "MARGINALIA" entries — which are transcription noise, not contract content — pollute similarity search results. Filtering `chunk_type == "clause"` before/after retrieval dramatically improves precision. Stamps no longer get their own `chunk_type` — see below.
- **`stamps`** keeps clause text clean for embedding while preserving provenance information (who sealed the page, confidentiality markings, page-control numbers). A stamp was never semantically independent content — it's an annotation *about* the page it sits on — so folding it into the metadata of the real clause it accompanies (rather than emitting it as its own retrievable chunk) both improves embedding quality and keeps the information available for provenance/audit queries.
- **`clause_no` / `clause_no_normalized` / `parent_clause` / `hierarchy_level`** together enable exact-match and hierarchical lookups ("what does clause 6.8.1 say?", "show me everything under 6.7.2") that pure vector similarity handles poorly — this is where metadata filtering beats embeddings. `parent_clause` also lets you retrieve sibling/parent context around a matched chunk even when the match itself was narrow.
- **`continues_previous`** lets the retriever stitch together a clause that was split across a page boundary, avoiding truncated/incomplete answers.
- **`pdf_page` / `printed_page`** support citation — telling the user exactly where in the physical contract an answer came from, which matters heavily in a legal/engineering-contract context.
- **`document_type` / `volume` / `chapter`** allow scoping a query to "only Specifications" or "only Volume 2" without re-embedding — essential once the corpus grows beyond these 3 chapters to the full multi-volume contract.
- **`low_confidence`** allows the system to warn users ("this clause was transcribed with lower OCR confidence — verify against the original") rather than presenting uncertain OCR output as fact.
- **`revision` / `approval_status`** matter because engineering contracts get amended; once addenda are ingested, these fields let the system prefer the latest approved version over a superseded one.
- **`document_sequence`** preserves reading order independent of how chunks were typed or grouped — useful for reconstructing "show me this section of the document as originally laid out" even after chunk_type-based filtering has been applied.

## 4. Fields to Index for Metadata Filtering (ChromaDB/Qdrant payload indexes)

Prioritize indexing (i.e., structured/filterable, not just stored) on:
- `document_id`, `chapter`, `document_type`, `volume` (scoping queries to a document subset)
- `chunk_type` (excluding marginalia/cover/index noise from retrieval)
- `clause_no_normalized`, `parent_clause`, `hierarchy_level` (direct clause lookup and hierarchical navigation)
- `pdf_page` (page-range filtering, e.g. "show me pages 10–20")
- `approval_status`, `revision` (retrieving only current/approved content)
- `low_confidence` (optionally excluding uncertain transcriptions from high-stakes answers)

`stamps` is stored but typically not indexed for filtering unless you expect queries like "show me every page stamped confidential" — in that case index `stamps[].organization` / `stamps[].type`.

## 5. Fields That Should Never Be Embedded (Stored Alongside the Vector Only)

None of the metadata fields above should be part of the text sent to the embedding model — only the clause `text` (optionally prefixed with `heading`/`section_heading` for context) should be embedded. Fields that are purely payload/metadata and must **never** enter the embedding input:
`chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`, `document_id`, `source_file`, `pdf_page`, `printed_page`, `stamps`, `approval_status`, `revision`, `low_confidence`, `continues_previous`, `parent_clause`, `hierarchy_level`.
(Optionally, `clause_no` + `heading` + `section_heading` CAN be lightly prepended to the embedded text as context, since they carry semantic value — but the rest are pure structured metadata.)

## 6. Constant vs. Per-Chunk Fields

**Constant across every chunk in the whole corpus (Project Metadata):**
`project_name`, `contract_number`, `employer_name`, `contractor_name`, `system_scope`, `stations_covered`

**Constant per source file (Document Metadata):**
`document_id`, `document_name`, `document_type`, `volume`, `chapter`, `source_file`, `total_pages_in_document`

**Changes for every chunk (Structural + Retrieval Metadata):**
`section_heading`, `clause_no`, `clause_no_normalized`, `parent_clause`, `hierarchy_level`, `heading`, `chunk_type`, `stamps`, `pdf_page`, `printed_page`, `continues_previous`, `low_confidence`, `chunk_id`, `chunk_number`, `document_sequence`, `chunk_hash`

## 7A. Where the Removed Fields Actually Belong

`embedding_model` and `vector_store` shouldn't ride on every chunk, but you still need to know, at the collection level, which model produced the vectors currently in a given Qdrant/Chroma collection — otherwise a future re-embed with a different model silently mixes incompatible vector spaces. Keep this as **one record per collection/run** (e.g. a `collections` config table or a README next to the vector store), not per chunk:
```json
{ "collection_name": "dmrc_be12be14", "embedding_model": "BAAI/bge-m3", "vector_store": "qdrant", "built_at": "2026-07-21", "chunking_strategy": "clause-level" }
```
`token_count` — if you want it, compute it at chunking time and log it in your ETL run log for chunk-size tuning; it's a build-time diagnostic, not a durable property of the content. `ingestion_timestamp` / `ingestion_pipeline_version` — same treatment: useful in a pipeline run log, not dissertation-relevant as per-chunk payload.

---

## 7. Complete Unified Metadata Schema (JSON, Placeholder Values)

```json
{
  "project_name": "Phase-II Delhi MRTS Project – CS to Qutab Minar & CS to Badarpur Corridors",
  "contract_number": "BE-12 LOT3 & BE-14 LOT3",
  "employer_name": "Delhi Metro Rail Corporation Ltd. (DMRC)",
  "contractor_name": "M/s Blue Star Limited",

  "document_id": "DMRC-BE12BE14-VOL2-CH1",
  "document_name": "Scope of Work – ECS (Chapter 1)",
  "document_type": "Scope of Work",
  "volume": "Volume 2",
  "chapter": "Chapter 1",
  "source_file": "DMRC_Chapter1_transcription.json",

  "section_heading": "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS",
  "clause_no": "6.7.2-1",
  "clause_no_normalized": "6.7.2.1",
  "parent_clause": "6.7.2",
  "hierarchy_level": 4,
  "heading": "Competency of Personnel",

  "pdf_page": 1,
  "printed_page": "000017",

  "chunk_id": "DMRC-BE12BE14-VOL2-CH1-p1-c6.7.2.1-000",
  "chunk_number": 42,

  "chunk_type": "clause",
  "continues_previous": false,
  "low_confidence": false,

  "stamps": [
    {
      "organization": "DMRC",
      "type": "Confidential seal",
      "page_control": "000017",
      "date": "Feb 2008"
    },
    {
      "organization": "Blue Star Ltd.",
      "type": "Company stamp"
    }
  ],

  "revision": "Feb 2008",
  "approval_status": "approved"
}
```

Note the `stations_covered`, `system_scope`, `total_pages_in_document`, `document_sequence`, and `language` fields from §2 are omitted here purely to mirror the exact field set you specified — reinstate any of them if you want them; they were left out of your list, not flagged as wrong.
