"""PDF text extraction — PyMuPDF (fitz).

Handles both text-layer PDFs and scanned PDFs (the latter returns near-empty
text per page; hand those pages to services/parsing/ocr.py instead).
"""
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageText:
    page_number: int
    text: str
    is_likely_scanned: bool  # heuristic: very little extractable text


def extract_pdf(path: str) -> list[PageText]:
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            # Heuristic: a real text page has plenty of characters per unit
            # area; a scanned page with no OCR layer yields almost nothing.
            is_scanned = len(text.strip()) < 20
            pages.append(PageText(page_number=i + 1, text=text, is_likely_scanned=is_scanned))
    return pages


def extract_pdf_full_text(path: str) -> str:
    """Convenience wrapper: all pages joined, page breaks marked."""
    pages = extract_pdf(path)
    return "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
