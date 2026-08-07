"""Rule engine — GDPR / CCPA / missing-clause checks.

Deliberately not ML-based: these are yes/no compliance checks a lawyer can
audit and you can extend without retraining anything. Runs on the full
cleaned document text (post services/preprocessing/cleaning.py) plus,
optionally, the clause-classification labels from
services/legal_ai/legalbert_classifier.py — pass those in when you have them
for higher-precision "missing clause" checks (label-based) rather than the
keyword fallback used here.
"""
import re
from dataclasses import dataclass


@dataclass
class ComplianceFinding:
    rule_id: str
    regulation: str  # "GDPR" | "CCPA" | "General"
    severity: str  # "info" | "warning" | "critical"
    title: str
    detail: str
    present: bool  # True if the required element WAS found (informational), False if missing (a gap)


# Each rule: id, regulation, severity, title, one or more regex patterns
# (case-insensitive) whose presence counts as "found". Keep patterns broad —
# false positives here are far cheaper than a lawyer relying on a missed gap.
_RULES = [
    dict(
        rule_id="gdpr_data_subject_rights",
        regulation="GDPR",
        severity="critical",
        title="Data subject rights (access, deletion, portability)",
        patterns=[r"right to (access|erasure|be forgotten|data portability|rectification)"],
    ),
    dict(
        rule_id="gdpr_legal_basis",
        regulation="GDPR",
        severity="warning",
        title="Legal basis for processing stated",
        patterns=[r"(legal basis|legitimate interest|lawful basis) for processing"],
    ),
    dict(
        rule_id="gdpr_dpo_contact",
        regulation="GDPR",
        severity="info",
        title="Data Protection Officer / privacy contact listed",
        patterns=[r"data protection officer", r"privacy (officer|contact|inquiries)"],
    ),
    dict(
        rule_id="gdpr_international_transfer",
        regulation="GDPR",
        severity="warning",
        title="International data transfer safeguards mentioned",
        patterns=[r"standard contractual clauses", r"adequacy decision", r"cross-border transfer"],
    ),
    dict(
        rule_id="ccpa_opt_out_sale",
        regulation="CCPA",
        severity="critical",
        title="'Do Not Sell My Personal Information' / opt-out of sale",
        patterns=[r"do not sell", r"opt.?out of (the )?sale"],
    ),
    dict(
        rule_id="ccpa_categories_collected",
        regulation="CCPA",
        severity="warning",
        title="Categories of personal information collected disclosed",
        patterns=[r"categories of (personal information|personal data) (we|that we) collect"],
    ),
    dict(
        rule_id="ccpa_non_discrimination",
        regulation="CCPA",
        severity="info",
        title="Non-discrimination for exercising privacy rights",
        patterns=[r"will not discriminate", r"non.?discrimination"],
    ),
    dict(
        rule_id="general_governing_law",
        regulation="General",
        severity="warning",
        title="Governing law clause",
        patterns=[r"governed by (the laws of|and construed)"],
    ),
    dict(
        rule_id="general_termination",
        regulation="General",
        severity="warning",
        title="Termination clause",
        patterns=[r"terminat(e|ion)", r"either party may (terminate|cancel)"],
    ),
    dict(
        rule_id="general_liability_limitation",
        regulation="General",
        severity="info",
        title="Limitation of liability clause",
        patterns=[r"limitation of liability", r"shall not be liable"],
    ),
    dict(
        rule_id="general_dispute_resolution",
        regulation="General",
        severity="info",
        title="Dispute resolution / arbitration clause",
        patterns=[r"arbitration", r"dispute resolution"],
    ),
]


def check_compliance(document_text: str) -> list[ComplianceFinding]:
    text_lower = document_text.lower()
    findings = []
    for rule in _RULES:
        present = any(re.search(pat, text_lower, re.IGNORECASE) for pat in rule["patterns"])
        findings.append(
            ComplianceFinding(
                rule_id=rule["rule_id"],
                regulation=rule["regulation"],
                severity=rule["severity"] if not present else "info",
                title=rule["title"],
                detail=("Found in document." if present else "Not found — likely missing from this document."),
                present=present,
            )
        )
    return findings


def summarize_gaps(findings: list[ComplianceFinding]) -> list[ComplianceFinding]:
    """Just the missing ones, worst severity first — this is what you'd
    actually surface to a user as "here's what's missing"."""
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    gaps = [f for f in findings if not f.present]
    return sorted(gaps, key=lambda f: severity_rank.get(f.severity, 3))
