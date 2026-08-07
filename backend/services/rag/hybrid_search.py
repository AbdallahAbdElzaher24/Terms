"""Hybrid retrieval — fuses the dense retriever (vector_store.py) and the
sparse/keyword retriever (keyword_search.py) before candidates go to the
cross-encoder reranker (reranker.py).

Why fuse instead of picking one: dense embeddings miss exact terms
(clause numbers, defined terms, names); BM25 misses paraphrase/semantic
matches. Neither is a superset of the other, so this is standard modern
RAG practice — retrieve with both, merge the two rankings, then let the
(more expensive, more accurate) cross-encoder reranker do the fine-grained
sort over the merged candidate pool.

Uses Reciprocal Rank Fusion (RRF) rather than trying to normalize and
average the two raw score scales directly — dense scores are cosine
similarities in [-1, 1], BM25 scores are unbounded term-frequency scores,
and naively averaging them would let whichever one happens to have larger
numbers dominate. RRF sidesteps that by only using each result's *rank*
(position) in its own list, which is scale-free by construction.
"""
import logging

from services.rag import keyword_search, vector_store

logger = logging.getLogger("rag.hybrid_search")


def hybrid_retrieve(
    query_embedding,
    query_text: str,
    document_id: str,
    top_k: int = 20,
    rrf_k: int = 60,  # standard RRF constant — dampens the impact of rank 1 vs rank 2 at the top of each list
) -> list[str]:
    """Returns up to top_k chunk texts, best-first, fused across both
    retrievers. Falls back to dense-only if rank_bm25 isn't installed or
    the keyword search otherwise fails — this must never be the reason a
    chat turn 500s, since the dense retriever alone is still a functioning
    RAG pipeline on its own (that's all this app had until this feature)."""
    dense_df = vector_store.similarity_search(query_embedding, top_k=top_k, document_id=document_id)
    dense_ranking = dense_df["text"].tolist() if not dense_df.empty else []

    try:
        bm25_ranking = [text for text, _score in keyword_search.bm25_search(query_text, document_id, top_k=top_k)]
    except ImportError:
        logger.info("rank_bm25 not installed — hybrid retrieval degraded to dense-only")
        bm25_ranking = []
    except Exception:  # noqa: BLE001 — keyword search is an enhancement, never a hard dependency for chat to work
        logger.exception("BM25 search failed — falling back to dense-only for this query")
        bm25_ranking = []

    if not bm25_ranking:
        return dense_ranking

    # RRF: each chunk's fused score is the sum, across the rankings it
    # appears in, of 1/(rrf_k + rank). A chunk ranked highly by *both*
    # retrievers rises above one ranked #1 by only one of them.
    fused_scores: dict[str, float] = {}
    for ranking in (dense_ranking, bm25_ranking):
        for rank, text in enumerate(ranking):
            fused_scores[text] = fused_scores.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)

    return sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
