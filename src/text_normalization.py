"""
text_normalization.py

Normalizes clause text prior to BGE-M3 encoding.

Design constraints (see Chapter 7.6 of the Software Design Report):
  - Unicode normalization (NFKC) to collapse compatibility characters
    (e.g. full-width digits, ligatures) introduced during OCR/transcription.
  - Whitespace cleanup: collapse repeated spaces/newlines from PDF extraction
    artifacts, but preserve paragraph boundaries as single newlines.
  - Clause numbers (e.g. "6.7.2-1"), engineering units (e.g. "415V, 50Hz",
    "kW", "m3/hr") and BOQ identifiers (e.g. "BOQ Item 4.2.1") are NEVER
    stripped or altered — they carry retrieval-relevant meaning.
  - No stop-word removal: BGE-M3 is a transformer-based dense encoder that
    uses full sentence context. Removing stop words breaks grammatical
    structure the model relies on and provides no benefit for a
    transformer encoder (unlike classical TF-IDF/BM25 pipelines).
"""

import re
import unicodedata


def normalize_text(raw_text: str) -> str:
    """Apply Unicode + whitespace normalization while preserving
    clause numbers, units, and BOQ identifiers verbatim.
    """
    if not raw_text:
        return ""

    # 1. Unicode normalization (NFKC): safe for OCR artifacts, does not
    #    touch alphanumeric clause numbers or unit symbols.
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Collapse runs of spaces/tabs (but not newlines yet).
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Collapse 3+ newlines (paragraph breaks) down to a single
    #    blank-line separator; preserve single newlines as-is.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Trim trailing whitespace on each line without touching content.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def build_embedding_input(clause_no: str, heading: str,
                           section_heading: str, text: str) -> str:
    """Constructs the exact string sent to the embedding model.

    Per the Chapter 6 metadata schema (Section 5, "Fields That Should
    Never Be Embedded"), only clause text is embedded, with clause_no /
    heading / section_heading optionally and lightly prepended as
    semantic context. No other metadata field is ever concatenated in.
    """
    prefix_parts = []
    if clause_no:
        prefix_parts.append(f"Clause {clause_no}")
    if heading:
        prefix_parts.append(heading)
    if section_heading:
        prefix_parts.append(section_heading)

    prefix = " | ".join(prefix_parts)
    normalized_text = normalize_text(text)

    if prefix:
        return f"{prefix}\n{normalized_text}"
    return normalized_text
