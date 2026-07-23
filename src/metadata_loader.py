"""
metadata_loader.py

Loads the finalized Unified Metadata Schema (Chapter 6) for each clause
chunk. This module does NOT redesign the schema or the chunking strategy
-- it implements the schema exactly as specified in
DMRC_Unified_Metadata_Schema.md, mapping the parsed JSON structure
(pages[].clauses[]) onto the Project / Document / Structural / Retrieval
/ Version metadata groups.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


NON_CLAUSE_HEADINGS = {"MARGINALIA", "INDEX", "STAMP"}

# Document-level constants extracted once per source file (cover page).
# In production these are extracted by a preprocessing pass over each
# file's cover page text; they are hard-coded here per the finalized
# schema example values for reproducibility of this chapter's examples.
DOCUMENT_METADATA = {
    "DMRC_Chapter1_transcription.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH1",
        "document_name": "Scope of Work - ECS (Chapter 1)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 1",
    },
    "DMRC_Chapter2_transcription.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH2",
        "document_name": "Scope of Work - ECS (Chapter 2)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 2",
    },
    "chapter3.json": {
        "document_id": "DMRC-BE12BE14-VOL2-CH3",
        "document_name": "Scope of Work - ECS (Chapter 3)",
        "document_type": "Scope of Work",
        "volume": "Volume 2",
        "chapter": "Chapter 3",
    },
}

PROJECT_METADATA = {
    "project_name": "Phase-II Delhi MRTS Project - CS to Qutab Minar & CS to Badarpur Corridors",
    "contract_number": "BE-12 LOT3 & BE-14 LOT3",
    "employer_name": "Delhi Metro Rail Corporation Ltd. (DMRC)",
    "contractor_name": "M/s Blue Star Limited",
}


def normalize_clause_no(clause_no: str) -> str:
    """'6.7.2-1' -> '6.7.2.1' (delimiter normalization only)."""
    if not clause_no:
        return ""
    return re.sub(r"[-_]", ".", clause_no)


def parent_clause_of(clause_no_normalized: str) -> str:
    if not clause_no_normalized or "." not in clause_no_normalized:
        return ""
    return clause_no_normalized.rsplit(".", 1)[0]


def hierarchy_level_of(clause_no_normalized: str) -> int:
    if not clause_no_normalized:
        return 0
    return len(clause_no_normalized.split("."))


def classify_chunk_type(heading: str, clause_no: str, continues_previous: bool) -> str:
    """Classify a raw clauses[] entry into its semantic chunk_type.

    heading is matched with startswith("COVER PAGE") rather than an exact
    match, because the source JSON uses several cover-page variants
    ("COVER PAGE", "COVER PAGE - SCOPE OF WORK / SPECIFICATIONS",
    "COVER PAGE - VOLUME 3(I)", etc.) -- an exact-match set only ever
    caught the bare "COVER PAGE" heading and let the other four variants
    fall through and be classified (and therefore embedded) as "clause".
    """
    if heading == "STAMP":
        return "stamp"
    if heading == "INDEX":
        return "index"
    if heading == "MARGINALIA":
        return "marginalia"
    if heading.startswith("COVER PAGE"):
        return "cover_page"
    if continues_previous:
        return "continuation"
    return "clause"


@dataclass
class ChunkRecord:
    text: str
    metadata: dict = field(default_factory=dict)


def _recompute_hash(record: ChunkRecord) -> None:
    """Keep chunk_hash in sync with record.text whenever text changes
    (e.g. after a continuation fragment is merged in). A stale hash
    computed only at creation time would silently stop reflecting the
    chunk's actual (post-merge) content, defeating its purpose as a
    change-detection/deduplication key.
    """
    record.metadata["chunk_hash"] = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:12]


def build_chunk_records(parsed_json: dict, source_file: str, carry_over_record: Optional[ChunkRecord] = None):
    """Maps pages[].clauses[] onto ChunkRecord objects carrying the
    finalized metadata schema. Chunking granularity (one clause = one
    chunk) is unchanged from Chapter 6/7 -- what changed is that this
    function now actually implements the merge/exclusion rules the
    Chapter 7 Chunking Strategy document specifies, instead of emitting
    every clauses[] entry as its own independent chunk regardless of type.

    Parameters
    ----------
    parsed_json : dict
        The parsed pages[].clauses[] structure for one source file.
    source_file : str
        Filename, used to look up per-document metadata.
    carry_over_record : ChunkRecord, optional
        The last real-clause chunk built by the PREVIOUS file in
        processing order. If this file's first clause entry is itself a
        continuation (continues_previous=True), it is merged into this
        carried-over record instead of becoming an orphaned chunk --
        this is the cross-file continuation observed at the
        Chapter1 -> Chapter2 boundary in this corpus (Chapter 7 SS7.5).

    Returns
    -------
    (records, open_record) : tuple[list[ChunkRecord], Optional[ChunkRecord]]
        records is the list of NEW chunks built from this file (the
        carry_over_record, if merged into, is not included -- it was
        already appended to the previous file's records list by the
        caller). open_record is the last real-clause chunk left open at
        the end of this file, to be threaded into the next file's call
        as its carry_over_record.
    """
    doc_meta = DOCUMENT_METADATA.get(source_file, {})
    records = []
    document_sequence = 0
    chunk_number = 0
    pending_stamps = []

    # open_record is the "current clause" that a leading continuation
    # fragment can merge into. It starts as whatever real clause was
    # still open at the end of the previous file (may be None for the
    # first file processed, or when running a single file in isolation).
    # Only real chunk_type=="clause" chunks are ever assigned to it --
    # a continuation can only continue an actual clause.
    open_record = carry_over_record

    # last_record is the most recently built chunk of ANY type (clause,
    # cover_page, index, marginalia) and is what a trailing STAMP attaches
    # to. This must be tracked separately from open_record: a stamp on a
    # cover page belongs to that cover page, not to whatever real clause
    # happened to precede it several pages earlier. Using open_record for
    # stamps as well would let a cover page's stamps queue up in
    # pending_stamps and cascade forward onto the next real clause chunk
    # instead of staying scoped to the page they actually appear on.
    last_record = carry_over_record

    for page in parsed_json.get("pages", []):
        pdf_page = page.get("pdf_page")
        printed_page = page.get("printed_page", "")
        section_heading = page.get("section_heading", "")

        for clause in page.get("clauses", []):
            clause_no = clause.get("clause_no", "")
            heading = clause.get("heading", "")
            text = clause.get("text", "")
            continues_previous = bool(clause.get("continues_previous", False))
            low_confidence = bool(clause.get("low_confidence", False))

            chunk_type = classify_chunk_type(heading, clause_no, continues_previous)

            # --- STAMP: fold into the nearest preceding real clause -------
            # Stamps are annotations ABOUT the page they sit on, never
            # independent content, so they are never their own chunk. They
            # attach to `open_record` (the clause chunk already built,
            # possibly carried over from the previous file) rather than to
            # whatever chunk gets built next -- in the source JSON a STAMP
            # entry always trails the clause it was scanned on.
            if chunk_type == "stamp":
                # --- Change 2: detect organization from stamp text ---------
                # Simple substring match against the raw stamp text. Order
                # doesn't matter here since "DELHI" and "BLUE STAR" are
                # mutually exclusive in practice; falls back to None
                # (unchanged prior behavior) if neither is present.
                organization = None
                if "DELHI" in text:
                    organization = "DMRC"
                elif "BLUE STAR" in text:
                    organization = "Blue Star Ltd."

                # --- Change 1: preserve complete stamp information ---------
                # Added "text" field carrying the actual stamp text (was
                # previously discarded); existing fields are untouched.
                stamp_entry = {
                    "organization": organization,
                    "type": heading,
                    "text": text,
                    "page_control": printed_page,
                }
                if last_record is not None:
                    last_record.metadata["stamps"].append(stamp_entry)
                else:
                    pending_stamps.append(stamp_entry)
                continue

            # --- CONTINUATION: merge into the clause it continues ---------
            # A continues_previous=True fragment is the tail of a clause
            # that was split across a page (or file) boundary. Per Chapter
            # 7 SS7.4/SS7.5 it must be reunited with that clause's chunk,
            # not stored as its own decontextualized chunk with no
            # clause_no/heading/parent_clause of its own.
            if chunk_type == "continuation":
                if open_record is not None:
                    open_record.text = (open_record.text.rstrip() + "\n\n" + text.strip()).strip()
                    if low_confidence:
                        open_record.metadata["low_confidence"] = True
                    _recompute_hash(open_record)
                else:
                    # No open clause exists at all (continuation with
                    # nothing to attach to, e.g. a single file run in
                    # isolation with no carry_over). Fall back to storing
                    # it as its own clause chunk rather than dropping data.
                    document_sequence += 1
                    chunk_number += 1
                    chunk_id = "-".join(filter(None, [
                        doc_meta.get("document_id", "DOC"),
                        f"p{pdf_page}",
                        f"seq{document_sequence}",
                        f"{document_sequence:03d}",
                    ]))
                    metadata = {
                        **PROJECT_METADATA,
                        **doc_meta,
                        "source_file": source_file,
                        "section_heading": section_heading,
                        "clause_no": "",
                        "clause_no_normalized": "",
                        "parent_clause": "",
                        "hierarchy_level": 0,
                        "heading": "",
                        "chunk_type": "clause",
                        "pdf_page": pdf_page,
                        "printed_page": printed_page,
                        "continues_previous": True,
                        "low_confidence": low_confidence,
                        "chunk_id": chunk_id,
                        "chunk_number": chunk_number,
                        "document_sequence": document_sequence,
                        "chunk_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                        "language": "en",
                        "approval_status": "approved",
                        "stamps": pending_stamps,
                    }
                    pending_stamps = []
                    open_record = ChunkRecord(text=text, metadata=metadata)
                    records.append(open_record)
                    last_record = open_record
                continue

            # --- Empty-text header stubs: not a standalone chunk -----------
            # e.g. clause_no="6.9", heading="TRAINING", text="" -- a section
            # header with no body of its own, immediately followed by its
            # numbered sub-clauses. Per Chapter 7 SS7.4 these are excluded
            # from embedding. Their clause_no/heading don't need to be
            # separately propagated: parent_clause/hierarchy_level for the
            # sub-clauses that follow are already derived purely from each
            # sub-clause's OWN clause_no_normalized, independent of whether
            # this stub has a chunk. `open_record` is deliberately left
            # unchanged so a trailing STAMP/continuation still attaches to
            # the last chunk with real content, not to this empty stub.
            if chunk_type == "clause" and not text.strip():
                continue

            # --- Real chunk: cover_page / index / marginalia / clause -----
            document_sequence += 1
            if chunk_type == "clause":
                chunk_number += 1

            clause_no_normalized = normalize_clause_no(clause_no)

            chunk_id = "-".join(filter(None, [
                doc_meta.get("document_id", "DOC"),
                f"p{pdf_page}",
                f"c{clause_no_normalized}" if clause_no_normalized else f"seq{document_sequence}",
                f"{document_sequence:03d}",
            ]))

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

            metadata = {
                **PROJECT_METADATA,
                **doc_meta,
                "source_file": source_file,
                "section_heading": section_heading,
                "clause_no": clause_no,
                "clause_no_normalized": clause_no_normalized,
                "parent_clause": parent_clause_of(clause_no_normalized),
                "hierarchy_level": hierarchy_level_of(clause_no_normalized),
                "heading": heading,
                "chunk_type": chunk_type,
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "continues_previous": continues_previous,
                "low_confidence": low_confidence,
                "chunk_id": chunk_id,
                "chunk_number": chunk_number,
                "document_sequence": document_sequence,
                "chunk_hash": content_hash,
                "language": "en",
                "approval_status": "approved",
                "stamps": pending_stamps,
            }
            pending_stamps = []

            record = ChunkRecord(text=text, metadata=metadata)
            records.append(record)

            # Any chunk just built is a valid target for a STAMP that
            # trails it (a stamp on a cover page belongs to that cover
            # page, not to some earlier real clause).
            last_record = record

            # Only real, substantive clause chunks stay "open" as a merge
            # target for a later CONTINUATION -- cover_page/index/
            # marginalia chunks are self-contained page furniture, not a
            # clause that a later fragment could be continuing.
            if chunk_type == "clause":
                open_record = record

    return records, open_record
