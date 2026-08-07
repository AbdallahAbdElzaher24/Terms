"""Key dates & obligations extraction.

The single most commonly requested feature in real contract-review
products (Ironclad, Lexion, Robin AI all lead with this): instead of
making the user read a 40-page contract to find "when do I have to give
notice to avoid auto-renewal," surface that as a structured, sortable
list the moment the document is uploaded.

Deliberately regex-only, matching the pattern already used in
compliance_rules.py — no ML model, no GPU, nothing to download:
  1. A date/duration regex finds every absolute date ("January 1, 2027",
     "01/12/2027") and relative duration ("30 days", "60 days prior") span
     in the document — this is the "where" and gives the exact text.
  2. A keyword window around each match classifies the "what" (renewal
     deadline vs termination notice vs payment due vs plain expiration).
     A wrong "type" label here is a user missing a real deadline, so an
     auditable regex a lawyer can read beats a black-box classifier until
     there's a labeled dataset to train and evaluate one against.
  3. dateutil best-effort parses absolute dates to ISO so a UI can sort/
     alert on them; relative durations ("30 days", "60 days prior") are
     kept as text since they need an anchor date (contract start/signing)
     this module doesn't have.

Runs on plain text, no model download, nothing to fail to load — this
module has no optional dependency of its own (dateutil is a core
requirement, already used elsewhere in the app).
"""
import re
from dataclasses import dataclass

# (obligation_type, display label, keyword patterns to look for in the
# window around a date/duration match). Order matters — first match wins,
# so more specific categories (renewal, termination) come before the
# generic "expiration" catch-all.
_CATEGORY_RULES = [
    (
        "renewal_deadline",
        "Auto-renewal / renewal notice deadline",
        [r"auto.?renew", r"renew(al|s|ed)?", r"non.?renewal", r"successive (one|two|1|2)?.?year"],
    ),
    (
        "termination_notice",
        "Termination notice period",
        [r"notice of termination", r"terminat(e|ion)", r"prior written notice", r"days.? notice"],
    ),
    (
        "payment_due",
        "Payment / invoice due date",
        [r"payment", r"invoice", r"due (within|date)", r"net\s?\d+", r"fee(s)? (are|is) due"],
    ),
    (
        "expiration",
        "Contract expiration / term end",
        [r"expir(e|ation|es)", r"term of this agreement", r"effective (until|through)",
         r"(remain|shall remain|continue) in (full force and )?effect (until|through)"],
    ),
]

# Absolute dates: "January 1, 2027" / "1 January 2027" / "01/12/2027" / "2027-01-12".
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"          # January 1, 2027
    rf"|\d{{1,2}}\s+{_MONTH}\s+\d{{4}}"                 # 1 January 2027
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"                        # 2027-01-12 (ISO)
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\b",               # 01/12/2027
    re.IGNORECASE,
)

# Relative durations: "30 days", "60 days prior", "12 months", "2 years".
_DURATION_RE = re.compile(
    r"\b\d+\s*(?:day|days|month|months|year|years|week|weeks)\b(?:\s+(?:prior|in advance|notice))?",
    re.IGNORECASE,
)


@dataclass
class DateMatch:
    text: str
    start: int
    end: int


@dataclass
class Obligation:
    obligation_type: str  # "renewal_deadline" | "termination_notice" | "payment_due" | "expiration" | "other"
    label: str  # human-readable category
    raw_text: str  # the exact date/duration span found, e.g. "60 days" or "January 1, 2027"
    parsed_date: str | None  # ISO date string if raw_text parsed as an absolute date, else None (relative duration)
    context: str  # surrounding sentence/snippet, for the user to verify against the source
    confidence: float  # fixed at 1.0 — regex matches are exact, not a probabilistic score


def _find_date_and_duration_spans(text: str) -> list[DateMatch]:
    matches = [DateMatch(m.group(0), m.start(), m.end()) for m in _DATE_RE.finditer(text)]
    matches += [DateMatch(m.group(0), m.start(), m.end()) for m in _DURATION_RE.finditer(text)]
    return matches


def _classify_context(window_text: str) -> tuple[str, str]:
    window_lower = window_text.lower()
    for obligation_type, label, patterns in _CATEGORY_RULES:
        if any(re.search(pat, window_lower) for pat in patterns):
            return obligation_type, label
    return "other", "Other date/deadline"


def _try_parse_date(raw_text: str) -> str | None:
    try:
        from dateutil import parser as date_parser

        dt = date_parser.parse(raw_text, fuzzy=False, default=None)
        return dt.date().isoformat()
    except Exception:
        return None  # relative durations ("30 days"), malformed spans, or unparseable text — fine, keep raw_text only


def _sentence_bounded_window(text: str, start: int, end: int, max_radius: int = 200) -> str:
    """Character window around [start, end), trimmed to stop at the nearest
    sentence boundary on each side rather than a hard char cutoff.

    A plain char window bleeds into the *next* sentence when a match sits
    near the end of one (e.g. "...in effect until January 1, 2027.\\nPayment
    is due within 30 days...") — the word "Payment" from the unrelated next
    sentence then leaks into the classification window and mislabels an
    expiration date as a payment due date. Caught by a unit test during
    development; fixed by cutting at ". " / ".\\n" boundaries.

    Deliberately does NOT split on bare newlines — contracts wrap lines
    mid-sentence constantly, and a bare "\\n" is not a sentence boundary.
    """
    lo = max(0, start - max_radius)
    hi = min(len(text), end + max_radius)
    left, right = text[lo:start], text[end:hi]

    last_boundary = None
    for m in re.finditer(r"\.\s", left):
        last_boundary = m.end()
    if last_boundary is not None:
        left = left[last_boundary:]

    m = re.search(r"\.\s", right)
    if m:
        right = right[: m.end()]

    return (left + text[start:end] + right).strip()


def extract_obligations(document_text: str) -> list[Obligation]:
    """Returns obligations sorted with dated (parseable) ones first, most
    relevant categories (renewal/termination) before generic ones."""
    matches = _find_date_and_duration_spans(document_text)

    obligations: list[Obligation] = []
    seen: set[str] = set()
    for m in matches:
        key = f"{m.text.strip().lower()}::{m.start}"
        if key in seen:
            continue
        seen.add(key)

        window = _sentence_bounded_window(document_text, m.start, m.end)
        obligation_type, label = _classify_context(window)

        obligations.append(
            Obligation(
                obligation_type=obligation_type,
                label=label,
                raw_text=m.text,
                parsed_date=_try_parse_date(m.text),
                context=window.strip(),
                confidence=1.0,
            )
        )

    priority = {"renewal_deadline": 0, "termination_notice": 1, "payment_due": 2, "expiration": 3, "other": 4}
    obligations.sort(key=lambda o: (o.parsed_date is None, priority.get(o.obligation_type, 9)))
    return obligations
