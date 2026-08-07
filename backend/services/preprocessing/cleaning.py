"""Cleaning pass that runs before chunking: strips repeated headers/footers
(common in multi-page contract PDFs — "Page X of Y", running titles), and
normalizes whitespace/control characters.
"""
import re
from collections import Counter

PAGE_MARKER_RE = re.compile(r"\[Page \d+\]")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def remove_repeated_headers_footers(pages: list[str], min_repeat_ratio: float = 0.6) -> list[str]:
    """pages: raw text per page. A line that repeats near-identically across
    a large fraction of pages (e.g. a running header/footer) gets dropped.
    """
    if len(pages) < 3:
        return pages  # not enough pages for the repetition heuristic to be reliable

    line_counts = Counter()
    per_page_lines = []
    for page in pages:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        per_page_lines.append(lines)
        for ln in set(lines):  # count each line once per page
            line_counts[ln] += 1

    threshold = max(2, int(len(pages) * min_repeat_ratio))
    boilerplate = {ln for ln, count in line_counts.items() if count >= threshold and len(ln) < 120}

    cleaned_pages = []
    for lines in per_page_lines:
        cleaned_pages.append("\n".join(ln for ln in lines if ln not in boilerplate))
    return cleaned_pages


def clean_document(raw_pages: list[str]) -> str:
    deduped = remove_repeated_headers_footers(raw_pages)
    joined = "\n\n".join(deduped)
    return normalize_whitespace(joined)
