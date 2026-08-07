"""BM25 keyword search — complements the dense embedding search in
vector_store.py.

Dense embeddings (see services/rag/embeddings.py) are strong on semantic/paraphrase matches
("what happens if I want to cancel early?" -> the auto-renewal clause,
even with no words in common) but they're known to underperform on exact
legal text: clause numbers ("Section 5.2(b)"), defined terms, party names,
dollar figures, statute citations. Ask "what does section 9.3 say" and a
purely dense retriever can miss the one chunk that literally contains
"9.3" if nothing about it is semantically close to the question.

BM25 is the classic sparse/lexical ranking answer to exactly that gap —
pure Python, no model download, no GPU, negligible latency. It's meant to
run alongside the dense retriever (see hybrid_search.py), not replace it.

Uses `rank_bm25` (~150 lines, pure Python, no C extension). The index is
rebuilt per-query from the same on-disk chunk store vector_store.py reads
from — fine at the "a few hundred thousand chunks" scale vector_store.py
already targets; persist the index instead of rebuilding it per call if
that changes.
"""
import re

from services.rag import vector_store

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# document_id -> (cache_version_at_build_time, BM25Okapi, texts). Rebuilding
# the index (tokenizing every chunk) on every single chat turn was pure
# waste once a document's chunks stop changing — this reuses the index as
# long as vector_store's cache version for that document hasn't moved
# (i.e. nothing was upserted/deleted since we built it).
_bm25_cache: dict[str, tuple[int, "object", list[str]]] = {}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_search(query: str, document_id: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Returns [(chunk_text, score), ...] sorted best match first, scores
    above 0 only. Empty list if the document has no indexed chunks, or if
    rank_bm25 isn't installed (raises ImportError — callers should catch
    this and fall back to dense-only, see hybrid_search.py)."""
    from rank_bm25 import BM25Okapi  # imported lazily: importing this module shouldn't require rank_bm25 unless this is actually called

    version = vector_store.get_cache_version(document_id)
    cached = _bm25_cache.get(document_id)
    if cached and cached[0] == version:
        _, bm25, texts = cached
    else:
        df = vector_store.get_document_chunks(document_id)
        if df.empty:
            return []
        texts = df["text"].tolist()
        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        _bm25_cache[document_id] = (version, bm25, texts)

    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(texts, scores), key=lambda pair: pair[1], reverse=True)
    return [(text, float(score)) for text, score in ranked[:top_k] if score > 0]
