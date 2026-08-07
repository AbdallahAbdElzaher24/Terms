"""Embeddings storage — one Parquet file per document under
backend/storage/embeddings/. Plain pyarrow/pandas, no extra service or
database to run; everything lives on local disk next to the rest of this
app's storage.

Similarity search is done in-process with numpy cosine similarity after
loading a document's file (fine up to a few hundred thousand chunks per
document; swap for a proper ANN index — e.g. Chroma or FAISS — if a single
document's chunk count grows well past that).
"""
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE_DIR = Path(__file__).parent.parent.parent / "storage" / "embeddings"

# In-process cache: document_id -> chunk rows (text/embedding/metadata).
# similarity_search() and get_document_chunks() are called at least once per
# chat turn for the same document_id, and previously each of those calls
# re-read the file from disk + rebuilt a numpy matrix from scratch — real
# cost on a multi-turn conversation. _cache_version lets keyword_search.py
# (a separate module) know when its own derived BM25 index is stale without
# a circular import back into that module.
_chunk_cache: dict[str, "pd.DataFrame"] = {}
_cache_version: dict[str, int] = {}


def get_cache_version(document_id: str) -> int:
    return _cache_version.get(document_id, 0)


def _invalidate(document_id: str) -> None:
    _chunk_cache.pop(document_id, None)
    _cache_version[document_id] = _cache_version.get(document_id, 0) + 1


def _path(document_id: str) -> Path:
    return STORAGE_DIR / f"{document_id}.parquet"


def upsert_chunks(
    document_id: str,
    chunk_texts: list[str],
    embeddings: np.ndarray,  # shape (n_chunks, embedding_dim)
    metadata: list[dict] | None = None,
) -> None:
    """Writes (overwrites) the chunk rows for a document. Call
    delete_document first if you just want to remove a document; this
    function always replaces that document's file wholesale — simplest
    correct option for a local single-user app where a document is
    re-processed as a whole, never patched row-by-row."""
    n = len(chunk_texts)
    metadata = metadata or [{} for _ in range(n)]
    df = pd.DataFrame(
        {
            "document_id": [document_id] * n,
            "chunk_index": list(range(n)),
            "text": chunk_texts,
            "embedding": [emb.astype(np.float32).tolist() for emb in embeddings],
            "metadata": [str(m) for m in metadata],
        }
    )
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(document_id), index=False)
    _invalidate(document_id)


def delete_document(document_id: str) -> None:
    _path(document_id).unlink(missing_ok=True)
    _invalidate(document_id)


def _load(document_id: str | None = None) -> pd.DataFrame:
    """Loads a document's chunk rows. Cached per document_id (see
    _chunk_cache above) — a chat conversation calls this repeatedly for the
    same document_id and the underlying data doesn't change between calls,
    so after the first hit this is a dict lookup instead of a disk read +
    pandas parse.

    document_id=None (used only by tests/edge cases) returns an empty
    frame — this store is always scoped to one document per file, there is
    no "load everything" table to scan."""
    empty = pd.DataFrame(columns=["document_id", "chunk_index", "text", "embedding", "metadata"])
    if document_id is None:
        return empty
    if document_id in _chunk_cache:
        return _chunk_cache[document_id]

    path = _path(document_id)
    df = pd.read_parquet(path) if path.exists() else empty
    _chunk_cache[document_id] = df
    return df


def get_document_chunks(document_id: str) -> pd.DataFrame:
    """Public accessor for a document's chunk rows (text + metadata, no
    embedding-similarity math) — used by keyword_search.py to build a BM25
    index from the same source of truth similarity_search reads from."""
    return _load(document_id)


def similarity_search(query_embedding: np.ndarray, top_k: int = 20, document_id: str | None = None) -> pd.DataFrame:
    """Cosine similarity over a document's chunks. Returns a DataFrame
    sorted by score, descending, with a `score` column."""
    df = _load(document_id)
    if df.empty:
        return df.assign(score=[])

    matrix = np.stack(df["embedding"].apply(np.array).to_numpy())
    q = query_embedding.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    scores = m_norm @ q_norm

    df = df.assign(score=scores).sort_values("score", ascending=False)
    return df.head(top_k).reset_index(drop=True)
