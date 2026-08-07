"""Semantic chunking: splits cleaned document text into retrieval-sized
chunks along natural boundaries (headings, paragraph breaks, numbered
clauses) rather than blind fixed-width windows — this matters a lot for
contracts, where cutting a clause mid-sentence hurts retrieval quality.

No ML model needed for the base version below (regex + greedy packing).
If you want true embedding-similarity-based splitting later, swap
`split_into_units` to use sentence embeddings + a similarity threshold —
the rest of the pipeline (packing, metadata) doesn't need to change.
"""
import re
from dataclasses import dataclass, field

# Matches common legal-document section starts: "1. ", "1.1 ", "Section 3:",
# "ARTICLE IV", "(a) ", markdown-style "## Heading"
SECTION_BOUNDARY_RE = re.compile(
    r"(?m)^(?=(?:\d+(?:\.\d+)*\.?\s+\S)|(?:Section\s+\d+)|(?:ARTICLE\s+[IVXLC\d]+)|(?:\([a-z]\)\s)|(?:#{1,3}\s))"
)


@dataclass
class Chunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def split_into_units(text: str) -> list[str]:
    """Split on section boundaries first, then further on blank lines within
    any unit that's still too long to be a single coherent chunk."""
    raw_units = [u for u in SECTION_BOUNDARY_RE.split(text) if u.strip()]
    units: list[str] = []
    for unit in raw_units:
        if len(unit) > 2500:
            units.extend(p for p in re.split(r"\n\s*\n", unit) if p.strip())
        else:
            units.append(unit)
    return units


def chunk_document(
    text: str,
    target_chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Greedily packs semantic units into ~target_chunk_size character
    chunks, carrying a small overlap forward so retrieval doesn't lose
    context at chunk boundaries."""
    units = split_into_units(text)
    chunks: list[Chunk] = []
    buffer = ""
    cursor = 0
    buffer_start = 0

    def flush():
        nonlocal buffer
        if buffer.strip():
            chunks.append(
                Chunk(
                    text=buffer.strip(),
                    chunk_index=len(chunks),
                    char_start=buffer_start,
                    char_end=buffer_start + len(buffer),
                    metadata=dict(metadata or {}),
                )
            )
        buffer = ""

    for unit in units:
        if buffer and len(buffer) + len(unit) > target_chunk_size:
            flush()
            # carry the tail of the previous chunk forward for overlap
            overlap_text = chunks[-1].text[-chunk_overlap:] if chunks else ""
            buffer_start = cursor - len(overlap_text)
            buffer = overlap_text
        elif not buffer:
            buffer_start = cursor
        buffer += ("\n" if buffer and not buffer.endswith("\n") else "") + unit
        cursor += len(unit)
    flush()

    return chunks
