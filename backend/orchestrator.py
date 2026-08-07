"""AI Orchestrator — the node in the diagram between the API layer and every
model/service. Two entry points:

  process_document(...)    — Input Pipeline -> Preprocessing -> RAG indexing
                              (call once per uploaded file)
  build_answer_context(...) — RAG retrieval -> Legal AI (NER/classify/risk/PII)
                              -> rule engine -> Prompt Builder
                              (call once per chat message)

The chat-turn path is modelled as a LangGraph StateGraph (see _build_chat_graph):
every pipeline stage is a graph node, and the conditional edges encode which
stages can be skipped (no document attached, embeddings failed, nothing
retrieved). The routing is explicit — you can read the whole pipeline off the
graph edges instead of following nested ifs, and adding a new stage (e.g. a
query-expansion or answer-evaluation step) is one node + one edge.

Each stage is wrapped in try/except via _safe so a missing model (you haven't
run a notebook / downloaded weights yet) degrades gracefully instead of
crashing the whole request — you'll see it noted in the response rather than
a 500. Node state merges warnings/timings via reducers, so the graph output
is the same shape build_answer_context used to assemble by hand.

Both entry points also record, per stage, how long it took and whether it
degraded (see _safe's `timings` param) — this is what lets you see "which
stage is actually slow" instead of just "the request took 2.3s" and
guessing.
"""
import concurrent.futures
import logging
import math
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from services.legal_ai.compliance_rules import check_compliance, summarize_gaps
from services.preprocessing.chunking import chunk_document
from services.preprocessing.cleaning import clean_document, normalize_whitespace
from services.prompt_builder import PromptContext, build_system_prompt
from services.rag import vector_store

logger = logging.getLogger("orchestrator")

# Below these, a label is too unreliable to state as fact to the LLM (which
# has no way to independently sanity-check "HIGH risk" or "NDA" — it will
# just repeat whatever it's told). Previously every classifier output was
# fed into the prompt verbatim regardless of confidence, so a 28%-confidence
# "high risk" guess and a 96%-confidence one looked identical to the model
# generating the user-facing answer. Tune these per-model once you have a
# held-out eval set (see scripts/train_legalbert_classifiers.py) — these are
# reasonable starting points, not measured optima.
CONFIDENCE_THRESHOLDS = {
    "clause_type": 0.5,  # classify_clause_types already thresholds internally at 0.5
}


def _safe(fn, *args, warnings: list[str], label: str, default=None,
          timings: dict[str, float] | None = None, **kwargs):
    """Runs fn(), catching model-not-ready/model-failed errors into
    `warnings` instead of raising, and — if a `timings` dict is passed —
    records how long the call took (successful or not) under `label`. This
    is the one place stage latency gets measured, so every stage that goes
    through _safe automatically shows up in the timing breakdown for free.

    NOTE: SystemExit is also caught here because spaCy's download helper calls
    sys.exit(1) on failure (e.g. when a model is missing and the network is
    unavailable).  SystemExit inherits from BaseException, not Exception, so a
    bare `except Exception` would miss it and let it propagate as an unhandled
    crash.  We treat it the same as any other model-setup failure: log a
    warning and return the default value so the rest of the pipeline continues.
    """
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as e:
        warnings.append(f"{label}: model not set up yet ({e})")
        return default
    except SystemExit as e:  # spaCy cli calls sys.exit() on download failure
        logger.warning("%s failed with SystemExit(%s) — likely a missing NLP model", label, e.code)
        warnings.append(f"{label}: model not available (run the appropriate download command)")
        return default
    except Exception as e:  # noqa: BLE001 — this is an intentional graceful-degrade boundary
        logger.exception("%s failed", label)
        warnings.append(f"{label}: failed ({e})")
        return default
    finally:
        if timings is not None:
            timings[label] = round((time.perf_counter() - start) * 1000, 1)


@dataclass
class ProcessResult:
    document_id: str
    num_chunks: int
    compliance_gap_titles: list[str]
    warnings: list[str]
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


def process_document(document_id: str, raw_pages: list[str], language_hint: str = "en") -> ProcessResult:
    """raw_pages: plain text per page/section, already extracted by whichever
    parser matched the file type (pdf_parser / docx_parser / ocr / speech_to_text)."""
    warnings: list[str] = []
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    t0 = time.perf_counter()
    cleaned = clean_document(raw_pages)
    chunks = chunk_document(cleaned, metadata={"document_id": document_id})
    chunk_texts = [c.text for c in chunks]
    timings["Cleaning + chunking"] = round((time.perf_counter() - t0) * 1000, 1)

    # Embeddings — needs sentence-transformers (see services/rag/embeddings.py)
    embeddings = _safe(
        lambda: __import__("services.rag.embeddings", fromlist=["embed_texts"]).embed_texts(chunk_texts),
        warnings=warnings,
        label="Embeddings",
        timings=timings,
    )
    if embeddings is not None:
        vector_store.delete_document(document_id)
        vector_store.upsert_chunks(document_id, chunk_texts, embeddings, metadata=[c.metadata for c in chunks])

    t0 = time.perf_counter()
    findings = check_compliance(cleaned)
    gaps = summarize_gaps(findings)
    timings["Compliance check"] = round((time.perf_counter() - t0) * 1000, 1)

    total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
    # Completeness proxy: fraction of stages that actually ran rather than
    # degrading — the only "quality" signal available at upload time, since
    # there's no question/answer yet to grade groundedness against.
    quality_score = round(max(0.0, 1.0 - len(warnings) / 2), 3)  # 2 = number of _safe-guarded stages above

    return ProcessResult(
        document_id=document_id,
        num_chunks=len(chunks),
        compliance_gap_titles=[g.title for g in gaps],
        warnings=warnings,
        stage_timings_ms=timings,
        total_latency_ms=total_latency_ms,
    )


@dataclass
class AnswerContext:
    system_prompt: str
    warnings: list[str]
    citations: list[dict] = field(default_factory=list)
    # Each citation: {"id": 1-based rank, "text": chunk excerpt, "score": reranker score (0.0 if reranker degraded)}.
    # Same retrieved chunks that go into system_prompt below — surfaced separately so
    # the frontend can show "this answer is based on clause X" instead of the user
    # having to trust an unverifiable LLM claim. Previously retrieved_texts was only
    # ever baked into the prompt string and never returned to the caller at all.
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    quality_score: float = 1.0
    # Heuristic 0-1 observability signal, NOT a measured answer-quality metric —
    # there's no ground truth at request time to compare against. It blends how
    # many pipeline stages degraded, whether anything was actually retrieved,
    # and how confident the reranker/risk classifier were. Useful for spotting
    # "this answer is on shakier ground than usual", not for grading correctness.


def _compute_quality_score(warnings: list[str], attempted_stages: int, citations: list[dict]) -> float:
    if attempted_stages == 0:
        return 1.0  # no document context requested — nothing could have degraded
    stage_health = max(0.0, 1.0 - len(warnings) / attempted_stages)
    grounding = 1.0 if citations else 0.5  # retrieved something vs. answering ungrounded
    # citations[0]["score"] is the cross-encoder reranker's raw logit (see
    # reranker.py), NOT a 0-1 probability — it's unbounded and routinely
    # negative for a poor (query, chunk) match. Squash it through a sigmoid
    # before folding it into the weighted blend below, otherwise a bad match
    # (e.g. -1.6) can single-handedly drag the whole quality score negative.
    raw_rerank_score = citations[0]["score"] if citations else 0.0  # 0.0 logit -> neutral 0.5 after sigmoid
    top_rerank_score = 1.0 / (1.0 + math.exp(-raw_rerank_score))
    return round(
        max(0.0, min(1.0, 0.5 * stage_health + 0.3 * grounding + 0.2 * top_rerank_score)), 3
    )


# ------------------------------------------------------------------ graph --


class ChatTurnState(TypedDict):
    """State flowing through the chat-turn graph. Reducers:
    - warnings / attempted_stages accumulate across nodes (operator.add).
    - timings is a per-node {label: ms} dict merged into one (operator.or_)."""

    question: str
    document_id: str | None
    language: str

    warnings: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], operator.or_]
    attempted_stages: Annotated[int, operator.add]

    query_embedding: object | None  # np.ndarray or None
    candidate_texts: list[str]
    retrieved_texts: list[str]
    citations: list[dict]

    entities: list[dict]
    # NOTE: this is contract-*risk* classification (Safe/Standard, Unilateral
    # Termination, Unlimited Liability, Non-Compete — see
    # services/legal_ai/legalbert_classifier.py), NOT CUAD-style clause-*type*
    # tagging (Auto-Renewal, Arbitration, ...). Kept as a separate field from
    # any future clause-type classifier so prompt_builder.py never mislabels
    # a risk flag as a "clause type" to the LLM.
    risk_flags: list[dict]  # [{"label": ..., "confidence": ...}, ...]
    pii_present: bool
    compliance_gap_titles: list[str]

    system_prompt: str
    quality_score: float


def _node_normalize_question(state: ChatTurnState) -> dict:
    # Same defensive whitespace/control-character normalization used on
    # uploaded documents (see services/preprocessing/cleaning.py) — cheap,
    # and pasted questions can carry the same invisible-character/CRLF mess
    # a copy-pasted PDF clause can. The heavier cleaning step (repeated
    # header/footer removal) only makes sense across multiple pages of a
    # document, so it's not relevant to a single short chat message.
    return {"question": normalize_whitespace(state["question"])}


def _node_embed_query(state: ChatTurnState) -> dict:
    """Produces the query embedding. A plain chat turn (no document_id) or a
    failed embedding both leave query_embedding=None, which the router after
    this node uses to skip straight to prompt building."""
    if not state["document_id"]:
        return {"query_embedding": None}

    warnings, timings = [], {}
    query_embedding = _safe(
        lambda: __import__("services.rag.embeddings", fromlist=["embed_query"]).embed_query(state["question"]),
        warnings=warnings,
        label="Query embedding",
        timings=timings,
    )
    return {
        "query_embedding": query_embedding,
        "warnings": warnings,
        "timings": timings,
        "attempted_stages": 1,
    }


def _node_retrieve(state: ChatTurnState) -> dict:
    """Dense+BM25 hybrid retrieval with a dense-only fallback if the hybrid
    module itself fails to run (rather than just BM25 inside it, which
    hybrid_search already handles on its own)."""
    warnings, timings = [], {}
    candidates = _safe(
        lambda: __import__(
            "services.rag.hybrid_search", fromlist=["hybrid_retrieve"]
        ).hybrid_retrieve(state["query_embedding"], state["question"], state["document_id"], top_k=20),
        warnings=warnings,
        label="Hybrid retrieval (dense+BM25)",
        default=None,
        timings=timings,
    )
    if candidates is None:
        candidates_df = vector_store.similarity_search(
            state["query_embedding"], top_k=20, document_id=state["document_id"]
        )
        candidates = candidates_df["text"].tolist() if not candidates_df.empty else []
    return {
        "candidate_texts": candidates,
        "warnings": warnings,
        "timings": timings,
        "attempted_stages": 1,
    }


def _node_rerank(state: ChatTurnState) -> dict:
    """Cross-encoder rerank of the candidate pool down to the top 5. If the
    reranker degrades, keep the top-5 candidates with score 0.0 so the LLM
    still gets grounded context."""
    warnings, timings = [], {}
    reranked = _safe(
        lambda: __import__("services.rag.reranker", fromlist=["rerank"]).rerank(
            state["question"], state["candidate_texts"], top_k=5
        ),
        warnings=warnings,
        label="Reranking",
        default=[(i, t, 0.0) for i, t in enumerate(state["candidate_texts"][:5])],
        timings=timings,
    )
    retrieved_texts = [text for _, text, _ in reranked]
    citations = [
        {
            "id": i + 1,
            "text": (text[:300] + "…") if len(text) > 300 else text,
            "score": round(float(score), 3),
        }
        for i, (_, text, score) in enumerate(reranked)
    ]
    return {
        "retrieved_texts": retrieved_texts,
        "citations": citations,
        "warnings": warnings,
        "timings": timings,
        "attempted_stages": 1,
    }


def _node_analyze(state: ChatTurnState) -> dict:
    """Risk classification and PII detection run on the same joined
    context concurrently (onnxruntime inference releases the GIL during
    compute), cutting wall-clock time to the slowest single stage."""
    joined_context = " ".join(state["retrieved_texts"])
    warnings, timings = [], {}

    def _run_risk():
        # Despite the module name, this is contract-*risk* classification
        # (Safe/Standard, Unilateral Termination, Unlimited Liability,
        # Non-Compete), not CUAD-style clause-*type* tagging — see
        # services/legal_ai/legalbert_classifier.py's module docstring.
        return _safe(
            lambda: __import__(
                "services.legal_ai.legalbert_classifier", fromlist=["classify_clause_types"]
            ).classify_clause_types(joined_context),
            warnings=warnings, label="Risk classification (Legal-BERT+DeBERTa ensemble)", default=[], timings=timings,
        )

    def _run_pii():
        return _safe(
            lambda: __import__("services.legal_ai.pii_detector", fromlist=["detect_pii"]).detect_pii(
                joined_context
            ),
            warnings=warnings, label="PII detection (Presidio)", default=[], timings=timings,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="legal-ai-stage") as pool:
        fut_risk = pool.submit(_run_risk)
        fut_pii = pool.submit(_run_pii)
        risk_results = fut_risk.result()
        pii_findings = fut_pii.result()

    risk_flags = [{"label": r.label, "confidence": round(r.confidence, 3)} for r in (risk_results or [])]

    return {
        "entities": [],
        "risk_flags": risk_flags,
        "pii_present": bool(pii_findings),
        "warnings": warnings,
        "timings": timings,
        "attempted_stages": 2,
    }


def _node_compliance(state: ChatTurnState) -> dict:
    """Rule-engine GDPR/CCPA/missing-clause checks on the retrieved context
    (see services/legal_ai/compliance_rules.py)."""
    joined_context = " ".join(state["retrieved_texts"])
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    gaps = summarize_gaps(check_compliance(joined_context))
    compliance_gap_titles = [g.title for g in gaps if g.severity in ("critical", "warning")]
    timings["Compliance check"] = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "compliance_gap_titles": compliance_gap_titles,
        "timings": timings,
        "attempted_stages": 1,
    }


def _node_build_prompt(state: ChatTurnState) -> dict:
    """Assembles PromptContext from whatever the earlier nodes produced and
    renders the final system prompt. Also the only place the heuristic
    quality score is computed — it needs the complete picture (warnings,
    stages attempted, citations, risk confidence), so it lives at the end."""
    ctx = PromptContext(
        retrieved_chunks=state["retrieved_texts"],
        risk_flags=state["risk_flags"],
        entities=state["entities"],
        compliance_gaps=state["compliance_gap_titles"],
        pii_present=state["pii_present"],
        language=state["language"],
    )
    quality_score = _compute_quality_score(
        state["warnings"], state["attempted_stages"], state["citations"]
    )
    return {
        "system_prompt": build_system_prompt(ctx),
        "quality_score": quality_score,
    }


def _route_after_embed(state: ChatTurnState) -> str:
    # No document attached, or embedding failed — nothing to retrieve against.
    if state["document_id"] and state["query_embedding"] is not None:
        return "retrieve"
    return "build_prompt"


def _route_after_retrieve(state: ChatTurnState) -> str:
    return "rerank" if state["candidate_texts"] else "build_prompt"


def _route_after_rerank(state: ChatTurnState) -> str:
    return "analyze" if state["retrieved_texts"] else "build_prompt"


def _build_chat_graph():
    """The chat-turn pipeline as a LangGraph StateGraph:
    START -> normalize -> embed -> (retrieve -> rerank -> analyze -> compliance)
                                -> build_prompt -> END

    Conditional edges express the short-circuits (plain chat with no document,
    failed embedding, empty retrieval) explicitly instead of nested ifs."""
    graph = StateGraph(ChatTurnState)

    graph.add_node("normalize_question", _node_normalize_question)
    graph.add_node("embed_query", _node_embed_query)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("rerank", _node_rerank)
    graph.add_node("analyze", _node_analyze)
    graph.add_node("compliance", _node_compliance)
    graph.add_node("build_prompt", _node_build_prompt)

    graph.add_edge(START, "normalize_question")
    graph.add_edge("normalize_question", "embed_query")
    graph.add_conditional_edges(
        "embed_query", _route_after_embed, ["retrieve", "build_prompt"]
    )
    graph.add_conditional_edges(
        "retrieve", _route_after_retrieve, ["rerank", "build_prompt"]
    )
    graph.add_conditional_edges(
        "rerank", _route_after_rerank, ["analyze", "build_prompt"]
    )
    graph.add_edge("analyze", "compliance")
    graph.add_edge("compliance", "build_prompt")
    graph.add_edge("build_prompt", END)

    return graph.compile()


# Compiled once at import; nodes are plain functions, so everything stays
# monkeypatchable in tests (patching services.rag.embeddings.embed_query etc.
# still affects what the nodes see at invoke-time).
_chat_graph = _build_chat_graph()


def build_answer_context(
    document_id: str | None,
    question: str,
    language: str = "en",
) -> AnswerContext:
    """Runs the LangGraph chat-turn pipeline and returns the assembled
    context (system prompt + warnings + citations + timings + quality score)
    that main.py streams the LLM answer with."""
    pipeline_start = time.perf_counter()

    state = _chat_graph.invoke(
        {
            "question": question,
            "document_id": document_id,
            "language": language,
            "warnings": [],
            "timings": {},
            "attempted_stages": 0,
            "query_embedding": None,
            "candidate_texts": [],
            "retrieved_texts": [],
            "citations": [],
            "entities": [],
            "risk_flags": [],
            "pii_present": False,
            "compliance_gap_titles": [],
            "system_prompt": "",
            "quality_score": 1.0,
        }
    )

    total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)

    return AnswerContext(
        system_prompt=state["system_prompt"],
        warnings=state["warnings"],
        citations=state["citations"],
        stage_timings_ms=state["timings"],
        total_latency_ms=total_latency_ms,
        quality_score=state["quality_score"],
    )
