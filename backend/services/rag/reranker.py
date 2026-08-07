"""Reranking — cross-encoder/mmarco-mMiniLMv2-L12-H384-v1.

Embedding similarity search (vector_store.similarity_search) is fast but
approximate; the reranker cross-encodes (query, chunk) pairs directly and
is much more accurate at picking the true top-K. Standard RAG pattern:
retrieve ~20-50 candidates by embedding similarity, then rerank down to the
3-5 you actually put in the prompt.

Picked over heavier options like BAAI/bge-reranker-base (~1.1GB) for a
lighter local footprint — this is a 12-layer MiniLM cross-encoder trained
on mMARCO (multilingual, incl. Arabic), small enough to load fast on CPU.
Swap MODEL_NAME to "BAAI/bge-reranker-base" or "BAAI/bge-reranker-large" if
you want a higher accuracy ceiling and have the disk/RAM/latency budget
for it.

First call downloads the model (~470MB) — needs internet once, then runs
fully offline.
"""
from functools import lru_cache

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder  # imported lazily — downloads weights on first use

    return CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: list[str], top_k: int = 5) -> list[tuple[int, str, float]]:
    """Returns [(original_index, text, score), ...] sorted best-first,
    truncated to top_k."""
    model = _get_model()
    pairs = [[query, c] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(range(len(candidates)), candidates, scores), key=lambda x: x[2], reverse=True)
    return ranked[:top_k]
