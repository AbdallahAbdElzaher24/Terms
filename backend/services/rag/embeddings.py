"""Embeddings — paraphrase-multilingual-MiniLM-L12-v2 via sentence-transformers.

Multilingual (handles Arabic and English in the same vector space, which
matters here) and small — picked over heavier options like BAAI/bge-m3
(~2.2GB) specifically to keep the local footprint light while still
covering 50+ languages including Arabic. Swap MODEL_NAME to "BAAI/bge-m3"
if you want the larger model's higher ceiling (longer max input, slightly
better retrieval quality) and have the disk/RAM/latency budget for it.

First call downloads the model (~470MB) from Hugging Face — needs internet
once, then runs fully offline.
"""
from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # imported lazily — downloads weights on first use

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Returns shape (len(texts), 384) float32 array, L2-normalized so
    cosine similarity == dot product."""
    model = _get_model()
    embeddings = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
