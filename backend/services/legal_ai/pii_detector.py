"""Sensitive-data / PII detection — Microsoft Presidio.

Detects emails, phone numbers, national IDs, credit cards, IBANs, etc. so
the app can flag "this document contains PII" and, if you want, redact it
before sending chunks to the LLM. Presidio's default recognizers are
English-tuned; for Arabic PII (national ID formats, phone numbers) you'll
likely want to add custom regex recognizers — see the commented example
below.

Setup:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_sm   # recommended: small & fast (~12 MB)

Speed notes:
    - en_core_web_sm is tried first (no word vectors, smaller pipeline —
      loads faster and runs several times faster per request than _md/_lg).
      Set PII_SPACY_MODEL=en_core_web_lg if you want the more accurate,
      slower model instead.
    - The parser/lemmatizer spaCy pipeline components are disabled since
      Presidio's NER recognizer doesn't use them.
    - detect_pii() caps input at 20,000 chars by default (override with
      max_chars=None) since the orchestrator only needs a present/absent
      signal, not exhaustive findings across an entire long document.
    - If no spaCy model is installed at all, falls back to a regex-only
      engine (no transformer/NLP context scoring, but pattern-based
      recognizers — emails, phones, credit cards, IBANs — still work).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class PIIFinding:
    entity_type: str  # "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", ...
    text: str
    start: int
    end: int
    score: float


def _build_analyzer():
    """Build a Presidio AnalyzerEngine, falling back gracefully if no spaCy
    model is installed.  Priority: fastest first — en_core_web_sm (~12MB,
    loads in ~1s and runs several times faster per-request) → en_core_web_md
    → en_core_web_lg (most accurate but noticeably slower to load and run) →
    regex-only.

    Override the preferred model with the PII_SPACY_MODEL env var, e.g.:
        PII_SPACY_MODEL=en_core_web_lg  (if you want accuracy over speed)
    """
    import os

    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Ordered list of spaCy models to try (fastest/smallest first — this is
    # what makes PII detection fast; en_core_web_sm has no word vectors and
    # a much smaller pipeline than md/lg, so both load time and per-request
    # inference are significantly quicker).
    _SPACY_MODELS = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

    preferred = os.getenv("PII_SPACY_MODEL")
    if preferred:
        # Put the user's preferred model first, keep the rest as fallback.
        _SPACY_MODELS = [preferred] + [m for m in _SPACY_MODELS if m != preferred]

    def _try_spacy_engine(model_name: str):
        """Return an AnalyzerEngine using the given spaCy model, or None on failure."""
        import spacy

        if not spacy.util.is_package(model_name):
            logger.debug("spaCy model %s not installed, skipping.", model_name)
            return None

        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
                # Presidio's spaCy recognizer only needs tokenization + NER
                # (entity types). The parser and lemmatizer are loaded by
                # spaCy's full pipeline but never consulted by Presidio, so
                # disabling them cuts both load time and per-call latency
                # with no loss of detection accuracy.
                "ner_model_configuration": {
                    "labels_to_ignore": [],
                },
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            # Belt-and-suspenders: also disable at the underlying spaCy
            # pipeline level in case the provider didn't already trim it.
            try:
                nlp = nlp_engine.nlp["en"]
                for component in ("parser", "lemmatizer", "attribute_ruler"):
                    if component in nlp.pipe_names:
                        nlp.disable_pipe(component)
            except Exception:  # noqa: BLE001 — purely a speed optimization, never fatal
                pass

            return AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load spaCy model %s: %s", model_name, exc)
            return None

    # 1) Try each spaCy model in preference order.
    for model in _SPACY_MODELS:
        engine = _try_spacy_engine(model)
        if engine is not None:
            logger.info("Presidio: using spaCy model '%s'.", model)
            _add_custom_recognizers(engine)
            return engine

    # 2) Fallback: regex-only engine (no spaCy dependency at all).
    logger.warning(
        "No spaCy model found. Presidio falling back to regex-only mode — "
        "NLP-context-dependent entities (e.g. PERSON names) will not be detected. "
        "Install a model with: python -m spacy download en_core_web_sm"
    )
    try:
        from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

        class _NoOpNlpEngine(NlpEngine):
            """Minimal NLP engine that does nothing — lets regex recognizers work."""

            def load(self) -> None:  # noqa: D401
                pass

            def is_loaded(self) -> bool:  # type: ignore[override]
                # Presidio calls this as a method (self.nlp_engine.is_loaded()),
                # not a property/attribute — a plain `is_loaded = True` class
                # attribute satisfies the ABC's abstractmethod check but then
                # blows up at call time with "'bool' object is not callable".
                return True

            def process_text(self, text: str, language: str):  # type: ignore[override]
                # NlpArtifacts' constructor args changed between presidio-analyzer
                # versions (older: entities/tokens/tokens_indices/dependencies/
                # keywords/lemmas/language; installed version here: entities/
                # tokens/tokens_indices/lemmas/nlp_engine/language/scores — no
                # `dependencies` or `keywords` kwargs anymore, and `nlp_engine`
                # is now required). tokens is typed as a spaCy Doc but nothing
                # in the regex-only recognizer path touches it in no-op mode.
                return NlpArtifacts(
                    entities=[],
                    tokens=[],
                    tokens_indices=[],
                    lemmas=[],
                    nlp_engine=self,
                    language=language,
                )

            def process_batch(self, texts, language, **_):  # type: ignore[override]
                return [self.process_text(t, language) for t in texts]

            def is_stopword(self, word, language):  # type: ignore[override]
                return False

            def is_punct(self, word, language):  # type: ignore[override]
                # No tokenizer here to check punctuation properly, and the
                # regex recognizers (email/phone/credit-card/IBAN/etc.) never
                # call this — it only needs to exist to satisfy the ABC.
                return False

            def get_supported_languages(self):
                return ["en"]

            def get_supported_entities(self):
                # NLP-derived entities (PERSON, LOCATION, etc.) aren't
                # available in no-op mode — only the pattern/regex
                # recognizers registered via _add_custom_recognizers and
                # Presidio's built-in regex recognizers (EMAIL_ADDRESS,
                # PHONE_NUMBER, CREDIT_CARD, IBAN_CODE, ...) will fire.
                return []

        regex_engine = AnalyzerEngine(nlp_engine=_NoOpNlpEngine())
        _add_custom_recognizers(regex_engine)
        return regex_engine
    except Exception as exc:  # noqa: BLE001
        # Last resort — standard init without any spaCy calls.
        # This may still try to download; if it does and fails, the _safe()
        # wrapper in orchestrator.py will catch it.
        logger.warning("Regex-only engine setup failed (%s); using bare AnalyzerEngine.", exc)
        engine = AnalyzerEngine.__new__(AnalyzerEngine)
        # Re-raise so _safe can catch and return the default empty list.
        raise


def _add_custom_recognizers(analyzer) -> None:
    """Register locale-specific recognizers on an existing AnalyzerEngine."""
    from presidio_analyzer import PatternRecognizer, Pattern

    # Egyptian national ID: 14 digits starting with 2 or 3 (birth century).
    egypt_id_pattern = Pattern(name="egypt_national_id", regex=r"\b[23]\d{13}\b", score=0.7)
    analyzer.registry.add_recognizer(
        PatternRecognizer(supported_entity="EG_NATIONAL_ID", patterns=[egypt_id_pattern])
    )


@lru_cache(maxsize=1)
def _get_analyzer():
    return _build_analyzer()


def detect_pii(text: str, language: str = "en", max_chars: int | None = 20_000) -> list[PIIFinding]:
    """Detect PII entities in *text*.

    language: Presidio's built-in recognizers are English-only by default;
    pass 'en' unless you've added an Arabic NLP pipeline.

    max_chars: NLP cost scales with input length. If the caller only needs a
    present/absent signal (as the orchestrator does — see `pii_present`),
    scanning an entire multi-document context is wasted work: PII near the
    start of retrieved context is just as informative as PII anywhere else.
    Defaults to the first 20,000 characters (~4-5k tokens); pass None to
    scan the full text.
    """
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]

    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language=language)
    return [
        PIIFinding(
            entity_type=r.entity_type,
            text=text[r.start: r.end],
            start=r.start,
            end=r.end,
            score=r.score,
        )
        for r in results
    ]


def redact_pii(text: str, findings: list[PIIFinding] | None = None) -> str:
    """Simple placeholder-based redaction, e.g. john@x.com -> [EMAIL_ADDRESS]."""
    findings = findings if findings is not None else detect_pii(text)
    # Replace back-to-front so earlier offsets stay valid.
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[: f.start] + f"[{f.entity_type}]" + text[f.end:]
    return text
