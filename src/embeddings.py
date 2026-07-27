"""The one and only embedding model load site.

Three independent ``SentenceTransformer()`` constructions (ingestion, pipeline,
eval) is how the "query and corpus embedded by the same model" invariant
silently breaks.  There is exactly one here, lazily constructed, and every
other module imports it.

Both entry points call ``encode(..., normalize_embeddings=True)``.  With
L2-normalised vectors and a table configured ``metric="cosine"``, LanceDB's
``_distance`` is cosine distance in ``[0, 2]`` and ``similarity = 1 - distance``
is genuine cosine similarity.  Skipping normalisation at *either* end
(all-MiniLM-L6-v2 does not normalise by default) corrupts the score scale
without raising.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from src.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point
    from sentence_transformers import SentenceTransformer


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when a store was written with a different embedding model.

    Named deliberately: the alternative is an opaque Arrow dimension error at
    first query, long after the cause.
    """


class _EmbeddingService:
    """Lazily-initialised singleton wrapper around a SentenceTransformer."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._model_name: str = get_settings().embedding_model_name

    # -- lifecycle ---------------------------------------------------------
    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Imported here, not at module scope: importing torch costs
                    # ~2s and several hundred MB, which we do not want to pay in
                    # `cost_analysis.py` or in tests that never embed anything.
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- public contract ---------------------------------------------------
    @property
    def model_name(self) -> str:
        """Model identifier, available without paying the load cost."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Output dimensionality (384 for all-MiniLM-L6-v2). Forces a load."""
        model = self._ensure_loaded()
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension;
        # accept either so the pin can move without an edit here.
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed a batch of documents. Returns ``(n, dim)`` float32, L2-normed.

        Batching amortises per-call framework overhead and yields larger GEMMs;
        it does *not* amortise model load, which happens once above.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            batch_size=batch_size or get_settings().embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns ``(dim,)`` float32, L2-normalised.

        Deliberately routed through the same ``encode`` call as ingestion so
        the two can never diverge in normalisation or pooling.
        """
        return self.embed_texts([text], batch_size=1)[0]


#: Process-wide singleton. Import this, never construct SentenceTransformer.
embedding_service = _EmbeddingService()


def assert_model_matches(stored_model_name: str, stored_dim: int) -> None:
    """Fail loudly when an existing store was built with another model.

    Called on table open.  Cosine similarity between vectors from two different
    models is meaningless but numerically well-formed, so nothing downstream
    would ever notice.
    """
    if stored_model_name != embedding_service.model_name:
        raise EmbeddingModelMismatchError(
            f"Vector store was written with embedding model "
            f"{stored_model_name!r} but the current configuration uses "
            f"{embedding_service.model_name!r}. Vectors from different models "
            f"are not comparable. Either set EMBEDDING_MODEL_NAME="
            f"{stored_model_name!r} or delete the store and re-ingest."
        )
    if stored_dim != embedding_service.dim:
        raise EmbeddingModelMismatchError(
            f"Vector store dimensionality is {stored_dim} but the current "
            f"embedding model produces {embedding_service.dim}. Delete "
            f"VECTOR_STORE_PATH and re-ingest."
        )
