# Chapter 7: Chunking Strategy

## 7.1 Introduction

Chunking is the process of dividing a large source document into smaller, retrievable units of text — *chunks* — each of which is independently embedded, indexed, and returned as a candidate context passage during retrieval. In a Retrieval-Augmented Generation (RAG) system, chunking sits between document parsing and embedding generation: it takes the structured output of the parsing stage (in this case, the page/clause JSON produced for the DMRC BE-12 LOT3 & BE-14 LOT3 contract) and converts it into the atomic units that the rest of the pipeline operates on.

Chunking is necessary because a Large Language Model cannot be handed an entire multi-hundred-page engineering contract as retrieval context; retrieval must instead locate the *specific* passage relevant to a user's query and supply only that passage (plus limited surrounding context) to the generation model. The quality of this selection is bounded by the quality of the chunks it is selecting from — no retrieval algorithm can compensate for chunks that are badly formed.

**Problems caused by extremely small chunks:**
- Fragments lose the surrounding context needed to be understood in isolation (e.g. a bullet item such as "• TVS works under whose scope..." without the parent clause that introduces it).
- The corpus becomes dominated by low-information fragments (a single stamp description, a section-header stub with no body text), diluting the proportion of semantically meaningful content in the index.
- More chunks must be retrieved to reconstruct a complete answer, increasing the chance of missing a needed fragment.

**Problems caused by extremely large chunks:**
- A single embedding vector must represent multiple unrelated ideas at once, which blurs the semantic signal and reduces retrieval precision — a query about "Penalty Clause" may retrieve an oversized chunk that also contains "Spare Parts" content, diluting relevance scoring.
- Large chunks waste context-window budget on the generation model with irrelevant text surrounding the actually relevant sentence.
- Citation becomes coarse: if a 3,000-word chunk is returned, the system cannot tell the user precisely which clause within it answered the question.

Chunking improves retrieval quality specifically when chunk boundaries are aligned with the document's own semantic boundaries — in this case, the clause. A well-chunked contract allows the retriever to return exactly the clause that answers a query, with clean provenance and no unrelated content attached.

## 7.2 Design Objectives

The chunking strategy for the DMRC contract corpus is designed to satisfy the following objectives:

| Objective | Description |
|---|---|
| Preserve semantic meaning | Each chunk should represent one coherent unit of contractual meaning, not an arbitrary slice of text. |
| Preserve clause hierarchy | The numbering structure (e.g. `6.7.2` under `6.7` under `6`) must remain recoverable from each chunk's metadata. |
| Maintain document traceability | Every chunk must be traceable back to its exact source file, PDF page, and printed page number. |
| Support metadata attachment | Chunk boundaries must align with the unit at which the Chapter 6 metadata schema attaches fields (clause-level), not a coarser or finer unit. |
| Improve retrieval precision | Chunks should contain one topic each, so that a query matches a chunk because of genuine topical relevance rather than incidental co-occurrence. |
| Enable citation | The system must be able to tell a user "this answer comes from Clause 6.7.2-4, Penalty Clause, page 000018" — a claim that is only possible if chunk boundaries respect clause boundaries. |

## 7.3 Chunking Strategy Selection

| Approach | Advantages | Disadvantages | Suitability for Engineering Contracts |
|---|---|---|---|
| Fixed-size chunking (e.g. 512 tokens) | Simple, uniform chunk size, easy to implement | Cuts across clause boundaries arbitrarily; splits a penalty clause from its trigger condition; ignores document structure entirely | Poor — the source JSON has no notion of a "512-token window," and fixed windows would routinely bisect clauses such as 6.7.2-1 through 6.7.2-4 |
| Recursive chunking (split on paragraph, then sentence, as needed to fit a size budget) | Adapts to content length; better than fixed-size at respecting paragraph breaks | Still size-driven rather than meaning-driven; a long clause may still be split mid-argument; a short clause may be merged with an unrelated neighboring paragraph | Weak — the JSON already delivers clean paragraph-equivalent units (clauses); recursive splitting re-introduces the boundary problem the parser already solved |
| Sentence-based chunking | Maximum granularity, precise citation to a single sentence | Destroys clause-level coherence; a clause like 6.2 "Test Programmes and Procedures" loses its internal logical flow across sentences; multiplies chunk count enormously (some clauses in this corpus exceed 2,600 characters) | Poor — over-fragments legally binding clauses that must be read as a whole |
| Paragraph-based chunking | Reasonably coherent units | The source `text` field for a clause is frequently itself a single paragraph or a short list, so paragraph-based chunking would collapse to clause-based chunking in most cases anyway, while mishandling clauses that legitimately span multiple paragraphs (e.g. Clause 3 "Drawings, Documents, Records and Manuals," which is followed by lettered sub-clauses within the same clause number) | Marginal — offers no benefit over clause-based chunking on this corpus and risks re-splitting clauses that the parser already kept intact |
| Section-based chunking (one chunk per `section_heading`) | Very few chunks, strong topical grouping at chapter level | `section_heading` in this corpus is a running page header (e.g. "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS") that stays identical across dozens of pages and tens of clauses; a section-level chunk would span the entire chapter, reproducing the "extremely large chunk" problem described in §7.1 | Poor — `section_heading` is far too coarse-grained in the observed data to serve as a chunk boundary |
| **Clause-based chunking** | Aligns exactly with the atomic unit already present in the parsed JSON (`clauses[]`); aligns with the unit at which the Chapter 6 metadata schema is defined (`clause_no`, `heading`, `hierarchy_level`, `parent_clause`); supports precise citation; naturally excludes non-substantive content types (`STAMP`, `COVER PAGE`, `INDEX`, `MARGINALIA`) via `chunk_type` filtering rather than requiring a separate splitting pass | Requires explicit rules for edge cases: empty-text header clauses, multi-page continuations, and clauses that are unusually short or unusually long (addressed in §7.4–7.7) | **Strong** — the source JSON was already parsed at clause granularity; clause-based chunking requires no re-segmentation of the text, only rules for merging, excluding, and packaging what is already there |

**Justification:** The three source files already deliver the document pre-segmented into `clauses[]`, each with its own `clause_no`, `heading`, and `text`. This unit is precisely the level at which a construction/engineering contract carries independent legal and technical meaning — a single clause (e.g. `6.7.2-4`, "Penalty Clause") states one obligation, one requirement, or one procedure. Re-chunking by fixed size, sentence, or paragraph would discard structure the parser has already established and risk severing clauses like `6.7.2-1`–`6.7.2-4` from one another or from their parent `6.7.2`. Clause-based chunking is therefore selected as the chunking unit for this corpus.

## 7.4 Chunk Formation Rules

The following rules govern how a chunk is formed from each entry in `clauses[]`, based on patterns directly observed across the three source files (151 total clause entries).

| Case | Observed in source data | Handling rule |
|---|---|---|
| One clause = one chunk | e.g. `6.8.1` "Tools and Test Equipment" | Default rule. Each clause with non-empty `text` becomes exactly one chunk. |
| Empty-heading, empty-`text` header clauses | 9 instances observed, e.g. `clause_no="1"`, `heading="Introduction"`, `text=""`; `clause_no="6.7.2"`, `heading="Maintenance During Defects Liability Period"`, `text=""` | These are section-header stubs, not content. They are **not embedded as standalone chunks** (there is no text to embed) but their `clause_no`/`heading` are retained and propagated as `parent_clause` context to the sub-clauses that immediately follow them, preserving the hierarchy signal for retrieval. |
| Continuation across pages | 15 instances observed, always with `clause_no=""`, `heading=""`, `continues_previous=true`, appearing as the **first** clause entry on a page | The continuation fragment is **merged into the chunk of the clause it continues** (the last real clause — non-stamp, non-empty `clause_no`/`heading` — on the preceding page), rather than emitted as its own chunk. See §7.5 for the exact merge algorithm. |
| Long clauses | Observed up to 2,661 characters in a single `text` field (e.g. Clause `1.5` "Verification and validation of design") | Retained as a single chunk. No further splitting is applied — splitting would break the clause-level metadata alignment established in §7.2. Long clauses are logged for size review (see §7.7) but not force-split. |
| Short clauses | Observed as low as 26 characters | Retained as a single chunk. No merging with neighboring clauses is performed, since each retains independent `clause_no` identity and may be independently cited even if brief. |
| Lists | Several clauses embed bulleted or numbered lists within a single `text` field (e.g. Clause `4.2` "Interfacing Agencies," Clause `3.1` "Within two months after the Notice to Proceed") | The list is **not** decomposed into separate chunks. The list is contractually meaningful only as a set belonging to its parent clause, so it is kept intact inside the parent clause's chunk. |
| Tables | None of the three source files contain a native tabular structure in the JSON (no `table` field is present in the schema); tabular content, if present in the original PDF, has evidently been transcribed as running or listed text within a clause's `text` field | No table-specific handling is required for the current three files. Should a future file introduce a dedicated table representation, the rule is: one table = one chunk, linked via `parent_clause` to the clause that introduces it, so it is not force-merged into surrounding prose. |
| Cover pages | 6 instances observed, `heading` values such as `"COVER PAGE"`, `"COVER PAGE - SCOPE OF WORK / SPECIFICATIONS"`, `"COVER PAGE - VOLUME 3(I)"` | Emitted as chunks with `chunk_type="cover_page"` per the Chapter 6 schema. Their `text` also feeds the one-time extraction of Project and Document metadata (§2A/2B of the Chapter 6 schema) but the cover-page chunk itself is retained (not discarded) so provenance of the extraction is auditable. |
| Index pages | 1 instance observed, `section_heading="INDEX"`, `heading="INDEX"` | Emitted as a chunk with `chunk_type="index"`. Excluded from default retrieval scope via `chunk_type` filtering (§4 of the Chapter 6 schema), consistent with its role as a navigation aid rather than contractual content. |
| Marginalia | 2 instances observed: one illegible handwritten mark (`low_confidence=true`), one explanatory annotation about a content re-ordering | Emitted as a chunk with `chunk_type="marginalia"`. Because marginalia is annotation *about* the page rather than independent contractual content, it is excluded from default retrieval scope, mirroring the treatment of stamps. |
| Stamps | 57 instances observed — by far the most frequent clause type (≈38% of all clause entries), present on nearly every page | **Never emitted as an independent chunk.** Consistent with the Chapter 6 metadata design, all `STAMP` clauses on a page are collected into the `stamps[]` metadata array and attached to the nearest preceding real (non-stamp) clause chunk on that page. If a page contains only stamps and no real clause (none observed in this corpus, but possible in general), the stamps are attached to the nearest real clause chunk on the previous page. |

## 7.5 Chunk Boundary Identification

Chunk boundaries are identified using a deterministic pass over `pages[].clauses[]`, in document order, using four signals from the source JSON:

- **`clause_no`** — presence of a non-empty value is the primary signal that a clause entry is a new, independently numbered unit and should start a new chunk.
- **`heading`** — used to classify the clause entry's semantic role. Values `"STAMP"`, values containing `"COVER PAGE"`, `"INDEX"`, and `"MARGINALIA"` route the entry to the non-substantive handling described in §7.4 rather than the default one-clause-one-chunk path.
- **`section_heading`** (page-level) — carried into each chunk's `section_heading` metadata field as contextual/running-header information. It is **not** used as a boundary signal in this corpus, since it remains constant across many consecutive pages and clauses (§7.3).
- **`continues_previous`** — the decisive signal for **not** starting a new chunk. When `true`, the clause entry (which always has empty `clause_no` and `heading` in the observed data) is treated as a continuation fragment.

**Merge algorithm for multi-page clauses:**

1. Process pages and their clauses in order (`pdf_page` ascending, then array order within the page).
2. Maintain a reference to the "current open clause" — the most recently started real-clause chunk (non-stamp, non-cover, non-index, non-marginalia).
3. When a clause entry with `continues_previous=true` is encountered, append its `text` to the current open clause's accumulated text (with a paragraph-break separator) rather than starting a new chunk. The chunk's `pdf_page`/`printed_page` metadata remains that of the clause's *origin* page (where `clause_no` and `heading` were first observed); the continuation does not create a second value for these fields.
4. When a clause entry with a non-empty `clause_no` or a substantive `heading` is encountered, close the current open clause (finalize its chunk) and open a new one.
5. Cross-file boundaries are handled identically to cross-page boundaries: the first clause of `DMRC_Chapter2_transcription.json` (`continues_previous=true`, continuing the discussion of works included in the Services) is merged into the chunk for Clause `1.3` "Scope of the work of supply," whose text began at the end of `DMRC_Chapter1_transcription.json`. This is possible only because chunking is performed over the full three-file document sequence rather than file-by-file in isolation.

Note that in this corpus, no continuation fragment ever carries its own `clause_no` or `heading` — the signal is unambiguous. Where `heading="6.7"` ("OPERATION & MAINTENANCE") appears at the end of `DMRC_Chapter2_transcription.json` with only a heading and no body text (`low_confidence=true`), and the *next* file (`chapter3.json`) opens with a fully-formed new clause `6.7.1` rather than a `continues_previous=true` fragment, the two are **not** merged — `6.7` remains a header-only stub (per §7.4) and `6.7.1` begins its own chunk, since the continuity signal (`continues_previous`) was not set.

## 7.6 Chunk Structure

Each finalized chunk is represented as an object with two top-level components: `page_content` (the text sent to the embedding model) and `metadata` (the full field set defined in the Chapter 6 Unified Metadata Schema, unmodified). `page_content` is composed of the clause's `heading` and `text` (with any merged continuation text appended), optionally prefixed with `section_heading`, per the embedding-input guidance already established in Chapter 6 §5. No metadata field is embedded.

```json
{
  "page_content": "Competency of Personnel\nDuring the DLP the Contractor shall support the Authority with sufficient personnel possessing the relevant skills and competence to manage and execute the maintenance obligations...",
  "metadata": {
    "project_name": "Phase-II Delhi MRTS Project – CS to Qutab Minar & CS to Badarpur Corridors",
    "contract_number": "BE-12 LOT3 & BE-14 LOT3",
    "employer_name": "Delhi Metro Rail Corporation Ltd. (DMRC)",
    "contractor_name": "M/s Blue Star Limited",

    "document_id": "DMRC-BE12BE14-VOL2-CH3",
    "document_name": "Scope of Work – ECS (Chapter 3)",
    "document_type": "Scope of Work",
    "volume": "Volume 2",
    "chapter": "Chapter 3",
    "source_file": "chapter3.json",

    "section_heading": "Contract BE-12 LOT3 & BE-14 LOT3 Scope of work –ECS",
    "clause_no": "6.7.2-1",
    "clause_no_normalized": "6.7.2.1",
    "parent_clause": "6.7.2",
    "hierarchy_level": 4,
    "heading": "Competency of Personnel",

    "pdf_page": 1,
    "printed_page": "000017",

    "chunk_id": "DMRC-BE12BE14-VOL2-CH3-p1-c6.7.2.1-000",
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
}
```

This example reflects an actual clause observed in `chapter3.json` (`pdf_page=1`), with the three `STAMP` entries on that same page folded into `stamps[]` rather than emitted as separate chunks, per §7.4.

## 7.7 Chunk Size Analysis

An empirical size analysis was performed across all 151 clause entries in the three source files (85 of which carry non-empty, non-stamp body text and become substantive `chunk_type="clause"` chunks).

| Metric | Value (characters) |
|---|---|
| Minimum clause text length | 26 |
| Maximum clause text length | 2,661 |
| Median clause text length | 416 |
| Mean clause text length | ~575 |
| Non-substantive `STAMP` entries excluded from embedding | 57 of 151 (≈38%) |
| Empty-text header stubs (§7.4) excluded from embedding | 9 of 151 |

**Small chunks:** The minimum-observed clause (26 characters) is a short, self-contained statement rather than a fragment, so it is retained as-is rather than merged — merging it with an unrelated neighboring clause would violate the "one clause, one topic" objective in §7.2.

**Large chunks:** The largest observed clauses (in the 1,700–2,700 character range) correspond to clauses that legitimately require several sentences to state a single procedural or technical requirement (e.g. Clause `1.5` "Verification and validation of design"). These are within the practical context-window budget of current embedding models and are not split.

**Maximum chunk size:** No hard maximum is enforced in this design, since clause-based segmentation already keeps every observed chunk well below any embedding-model input limit. Should a future volume of the contract contain a single clause exceeding a practical size threshold (e.g. several thousand words), the recommended escalation — out of scope for embedding-model discussion here — is a sub-clause split governed by internal lettered/numbered sub-items already present in the source text, preserving `parent_clause` linkage rather than an arbitrary fixed-size cut.

**Chunk overlap:** Overlap (the common RAG technique of duplicating trailing text from one chunk at the start of the next) is **not required** for clause-based chunking on this corpus. Overlap exists to compensate for arbitrary boundaries cutting through a continuous idea; here, boundaries are the document's own clause boundaries, and the `continues_previous` merge (§7.5) already reunites any text that was split purely by a page break rather than by contractual meaning. Introducing overlap on top of clause-based chunking would duplicate clause content across neighboring chunks and degrade retrieval precision without addressing any boundary problem that still exists.

## 7.8 Chunking Workflow

```
PDF
  ↓
OCR / Parser              → produces pages[].clauses[] JSON per document
  ↓
Structured JSON            → DMRC_Chapter1_transcription.json, DMRC_Chapter2_transcription.json, chapter3.json
  ↓
Clause Detection           → classify each clause entry: real clause / STAMP / COVER PAGE / INDEX / MARGINALIA / continuation
  ↓
Chunk Generation           → apply formation rules (§7.4) and boundary/merge algorithm (§7.5) across the full ordered
                              document sequence (all three files treated as one continuous document)
  ↓
Metadata Attachment        → populate the unmodified Chapter 6 schema fields per chunk (Project, Document, Structural,
                              Retrieval, Version metadata); fold STAMP entries into stamps[]
  ↓
Chunk Objects               → { page_content, metadata } objects, ready for the embedding stage
```

**Stage explanations:**

- **OCR / Parser:** Converts the scanned or digital contract PDF into the page/clause JSON structure already used as input to this chapter. This stage is upstream of and outside the scope of the chunking design.
- **Structured JSON:** The three files analyzed in this chapter — each representing a contiguous portion of the ECS Scope of Work document, ordered `Chapter1 → Chapter2 → Chapter3`.
- **Clause Detection:** A classification pass that labels each `clauses[]` entry by its role, using the `heading` and `clause_no` signals identified in §7.5, so that downstream logic knows which formation rule in §7.4 applies.
- **Chunk Generation:** Executes the merge algorithm of §7.5 across the full three-file sequence (not per file in isolation), so that cross-file continuations — such as the boundary between Chapter 1 and Chapter 2 observed in this corpus — are correctly reunited into one chunk.
- **Metadata Attachment:** Populates every field of the Chapter 6 Unified Metadata Schema for each finalized chunk, without modification to that schema, including derived fields (`clause_no_normalized`, `parent_clause`, `hierarchy_level`, `chunk_id`, `document_sequence`, etc.) and the `stamps[]` provenance array.
- **Chunk Objects:** The final output of this chapter — a list of `{page_content, metadata}` objects per source document, ready to be passed to the embedding-generation stage (Chapter 8), which is outside the scope of this chapter.

## 7.9 Advantages of the Proposed Strategy

- **Better semantic coherence:** Each chunk corresponds to exactly one contractual clause, matching the unit at which the contract itself asserts meaning.
- **Clause-level retrieval:** A query about a specific obligation (e.g. penalty terms) can retrieve the exact clause (`6.7.2-4`, "Penalty Clause") rather than an oversized passage diluted with unrelated content.
- **Accurate citations:** Because `pdf_page`, `printed_page`, `clause_no`, and `heading` are preserved per chunk, the system can cite an answer down to the specific clause and page, which matters heavily in a legal/engineering-contract context.
- **Better filtering:** `chunk_type` allows retrieval to exclude the substantial non-content volume observed in this corpus — 57 stamp entries and several cover/index/marginalia entries — without discarding that information (it remains available via `stamps[]` and the metadata payload for provenance/audit queries).
- **Lower hallucination risk:** Precisely bounded, topically coherent chunks reduce the chance that a generation model is fed a passage that mixes two unrelated clauses, which is a common cause of the model conflating requirements from different parts of a contract.
- **Enterprise scalability:** Because the chunking unit is intrinsic to the source structure (the `clauses[]` array) rather than an externally imposed size parameter, the same rules apply unchanged as the corpus grows from these 3 chapters to the full multi-volume contract referenced in the Chapter 6 schema design.

## 7.10 Limitations

| Limitation | Observed manifestation in this corpus | Mitigation |
|---|---|---|
| Extremely long clauses | Clauses up to 2,661 characters (e.g. Clause 1.5) | Retained as single chunks within practical embedding limits (§7.7); no forced splitting that would break clause-level metadata alignment. |
| OCR errors | 3 clauses flagged `low_confidence=true`, including a heading (`6.7`) whose body text could not be transcribed | The `low_confidence` field is preserved per chunk (defaulting to `false` when absent) so the retrieval layer can down-rank or flag such chunks for human verification, per the Chapter 6 design. |
| Missing clause numbers | Several clauses carry an empty `clause_no` even when they are substantive content, not continuations (e.g. the `Routine and Corrective Maintenance Procedures` clause in `chapter3.json`, which has a `heading` but no `clause_no`) | Such clauses are still chunked individually (heading-driven boundary, per §7.5) and their `parent_clause` is inferred from the nearest preceding numbered clause, preserving hierarchical navigability even without a native number. |
| Complex tables | No native table structure is present in any of the three files; tabular content, if it existed in the source PDF, was flattened into running text during parsing | The chunking design assumes clause-level `text` already includes any such flattened content; no table-specific chunk type is required for this corpus, though §7.4 defines a rule for future files that may carry an explicit table field. |
| Scanned drawings / non-text content | Not present as distinct entries in the parsed JSON (drawings referenced only as textual mentions within clause text, e.g. "with drawing references") | Outside the scope of text chunking; if drawing images are separately extracted in a future pipeline stage, they would require their own non-text chunk type, which is not addressed by this chapter. |

## 7.11 Summary

This chapter defined a clause-based chunking strategy for the DMRC BE-12 LOT3 & BE-14 LOT3 contract corpus, selected after comparing it against fixed-size, recursive, sentence-based, paragraph-based, and section-based alternatives (§7.3). The strategy treats each entry in the source JSON's `clauses[]` array as the default chunking unit, with explicit rules (§7.4) for the non-standard cases actually observed across the three source files: 57 `STAMP` entries folded into per-chunk provenance metadata rather than emitted as chunks; 6 cover-page, 1 index, and 2 marginalia entries retained as low-priority `chunk_type` values excluded from default retrieval; 9 empty-text section-header stubs excluded from embedding but preserved as hierarchy context; and 15 multi-page continuation fragments merged into their originating clause via a deterministic boundary-detection algorithm (§7.5) that operates across the full three-file document sequence, correctly reuniting even the continuation that spans the Chapter 1 → Chapter 2 file boundary. The resulting chunk structure (§7.6) pairs each chunk's `page_content` with the complete, unmodified Chapter 6 Unified Metadata Schema, and empirical analysis (§7.7) confirms that clause-based segmentation produces chunks (median ~416 characters, maximum 2,661 characters) that require no artificial overlap to preserve context. This output — a set of `{page_content, metadata}` chunk objects — constitutes the direct input to the next stage of the pipeline, embedding generation, without further restructuring.
