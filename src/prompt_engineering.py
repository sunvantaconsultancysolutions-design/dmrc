"""
prompt_engineering.py

Chapter 11 -- Gemma 3 Prompt Engineering.

Pure prompt CONSTRUCTION only: no model loading, no inference. This
module takes the merged candidate list already produced by Chapter 9's
hybrid_retriever.hybrid_search() and turns it into the exact prompt
text that gets handed to Gemma 3 in generate_answer.py. Kept separate
from model loading (same split as batch_embed.py/embed_single.py) so
the prompt logic can be built and unit-tested without a GPU or the
~24GB Gemma 3 12B checkpoint in memory.

------------------------------------------------------------------------
11.4 Prompt Components
------------------------------------------------------------------------
The assembled prompt has five parts, each implemented below:
    1. System Prompt        -> SYSTEM_PROMPT
    2. Retrieved Context     -> format_context()
    3. User Query            -> inserted directly into build_prompt()
    4. Response Instructions -> RESPONSE_INSTRUCTIONS
    5. Output Format         -> ANSWER_SECTION_HEADER (model completes from here)
"""

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 11.5 System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Engineering Contract Assistant for the DMRC BE-12 LOT3 & BE-14 LOT3 project.

Answer ONLY using the supplied context below.
Never invent information that is not present in the context.

If the answer is not available in the supplied context, respond with exactly:
"I could not find the requested information in the provided documents."

Always cite, whenever available:
- Clause Number
- Page Number
- BOQ Item Number"""


# ---------------------------------------------------------------------------
# 11.8 Response Instructions
# ---------------------------------------------------------------------------

RESPONSE_INSTRUCTIONS = """Use only the supplied context. Do not assume missing information.
Quote exact values whenever possible (quantities, penalties, durations, thresholds).
Cite the Clause Number, Page Number, and BOQ Item Number (if available) for every claim.
Clearly say so if the requested information is unavailable in the context.
Generate a concise, engineering-style response."""


ANSWER_SECTION_HEADER = "ANSWER"


# ---------------------------------------------------------------------------
# 11.11 Few-Shot Prompting
# ---------------------------------------------------------------------------
# Optional, off by default in build_prompt() unless use_few_shot=True is
# passed explicitly by the caller.
# Kept to a single example -- per 11.16 "context compression" / "maximum
# token limitation", every extra example is retrieved-context budget the
# actual candidates don't get, so this is intentionally minimal.

FEW_SHOT_EXAMPLE = """Question
What is BOQ Item 4.2?

Answer
BOQ Item 4.2 specifies the supply of Cooling Towers with a quantity of 6 Nos.

Source
BOQ Item 4.2"""


# ---------------------------------------------------------------------------
# 11.6 Retrieved Context / 11.10 step 2-4 (dedupe, merge, insert metadata)
# ---------------------------------------------------------------------------

def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate chunk_ids, keeping the first (highest-ranked)
    occurrence. hybrid_retriever.merge_candidates() already dedupes
    across the dense/sparse merge, but this module is written to also
    work standalone against a plain dense- or sparse-only result list
    (e.g. query.py's search() output), so it re-applies the same rule
    defensively rather than assuming the caller already did it.
    """
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(candidate)
    return deduped


def _format_document_name(metadata: Dict[str, Any]) -> Optional[str]:
    """Turns a raw source-file reference into a readable document title.

    If metadata already carries a descriptive title (e.g. from a
    document manifest), that is used as-is. Otherwise the raw filename
    (e.g. "chapter3.json") is cleaned up into something readable
    (e.g. "Chapter 3") purely for display -- the underlying metadata
    and retrieval logic are untouched.
    """
    document_title = metadata.get("document_title")
    if document_title:
        return document_title

    document_name = metadata.get("document_name")
    if not document_name:
        return None

    stem = document_name.rsplit(".", 1)[0] if "." in document_name else document_name
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", stem)
    stem = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", stem)
    stem = stem.strip()
    return stem.title() if stem else document_name


def format_context(candidates: List[Dict[str, Any]]) -> str:
    """Renders the retrieved candidates as numbered "Rank N" blocks
    (11.6) -- these are reranked results (BGE Reranker), so "Rank"
    reflects that ordering more accurately than a generic "Document"
    label. Each block inserts the structural metadata (11.10 step 4:
    clause number, heading, page, BOQ item, document name) alongside
    the chunk's text, in the order: Clause Number, Heading, Page
    Number, BOQ Item Number, Source Document.

    BOQ Item Number is only shown when metadata actually carries an
    item_number field -- this pipeline currently only indexes contract
    clauses (no BOQ parser exists yet), so this stays conditional rather
    than printing an empty "BOQ Item: N/A" on every single result.
    """
    if not candidates:
        return ""

    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        metadata = candidate["metadata"]

        header_parts = []
        clause_no = metadata.get("clause_no")
        if clause_no:
            header_parts.append(f"Clause {clause_no}")
        heading = metadata.get("heading")
        if heading:
            header_parts.append(heading)
        header = " | ".join(header_parts) if header_parts else "Clause (number unavailable)"

        chunk_text = candidate["document"].strip()
        chunk_text = re.sub(r"\n\s*\n+", "\n\n", chunk_text)

        lines = [f"Rank {i}", header, "", chunk_text]

        detail_parts = []
        pdf_page = metadata.get("pdf_page")
        if pdf_page not in (None, ""):
            detail_parts.append(f"Page {pdf_page}")
        item_number = metadata.get("item_number")
        if item_number:
            detail_parts.append(f"BOQ Item {item_number}")
        document_name = _format_document_name(metadata)
        if document_name:
            detail_parts.append(f"Source Document: {document_name}")
        if detail_parts:
            lines.append("")
            lines.append(" | ".join(detail_parts))

        blocks.append("\n".join(lines))

    return "\n\n--------------------------------\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 11.9 Prompt Template / 11.10 Prompt Construction Workflow
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 48

NO_CANDIDATES_CONTEXT_PLACEHOLDER = "(No relevant context was retrieved for this query.)"


def _section(header: str, body: str) -> str:
    """Wraps a body of text in a "==== HEADER ====" block, used for
    every section of the assembled prompt so the template reads the
    same way debuggers/log viewers see it in 11.9.
    """
    return f"{_SEPARATOR}\n{header}\n{_SEPARATOR}\n{body}"


def build_prompt(query: str, candidates: List[Dict[str, Any]], use_few_shot: bool = False) -> str:
    """Assembles the full Gemma 3 prompt from a user query and the
    merged retrieval candidates, following the Chapter 11.10 workflow:

        1. Receive top-ranked documents.      (candidates, as passed in)
        2. Remove duplicate chunks.            -> deduplicate_candidates()
        3. Merge document context.             -> format_context()
        4. Insert metadata.                    -> done inside format_context()
        5. Insert user query.                  -> below
        6. Generate final prompt.              -> return value
        7. Send prompt to Gemma 3.             -> generate_answer.py

    use_few_shot defaults to False (11.11: few-shot prompting is
    optional) so the single example doesn't eat into the retrieved
    context's token budget unless explicitly requested.

    Returns the complete prompt string, ending right after the "ANSWER"
    header so the model's own generation is the completion of the
    template shown in 11.9.
    """
    deduped = deduplicate_candidates(candidates)
    context_block = format_context(deduped) or NO_CANDIDATES_CONTEXT_PLACEHOLDER

    sections = [_section("SYSTEM", SYSTEM_PROMPT)]

    if use_few_shot:
        sections.append(_section("EXAMPLE", FEW_SHOT_EXAMPLE))

    sections.append(_section("CONTEXT", context_block))
    sections.append(_section("QUESTION", query.strip()))
    sections.append(_section("INSTRUCTIONS", RESPONSE_INSTRUCTIONS))
    sections.append(f"{_SEPARATOR}\n{ANSWER_SECTION_HEADER}\n{_SEPARATOR}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 11.14 Handling Missing Information
# ---------------------------------------------------------------------------

NO_CONTEXT_ANSWER = "I could not find the requested information in the provided documents."


def has_usable_context(candidates: List[Dict[str, Any]]) -> bool:
    """True if there is at least one retrieved candidate to answer from.
    generate_answer.py checks this BEFORE calling Gemma 3 at all -- if
    retrieval returned nothing, there is no reason to spend a GPU
    inference call producing text that the system prompt would force
    into the same canned refusal anyway.
    """
    return bool(candidates)
