import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID
from .EmbeddingCalculator import EmbeddingCalculator


logger = logging.getLogger(__name__)


class CodeRankCalculator(EmbeddingCalculator):
    """
    EmbeddingCalculator implementation backed by Qwen3-Embedding-0.6B.

    Returns float32 embeddings as bytes (little-endian).
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._model = None
        self._device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen3 embedding model."""
        try:
            if self._device:
                self._model = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True, device=self._device)
            else:
                self._model = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True)
            logger.info("Loaded embedding model from %s", EMBEDDING_MODEL_ID)
        except Exception as e:
            msg = f"Failed to load embedding model: {EMBEDDING_MODEL_ID}"
            logger.error("%s; error: %r", msg, e)
            raise RuntimeError(msg) from e

    @property
    def dimensions(self) -> int:
        """Return the native embedding dimension for Qwen3-Embedding-0.6B."""
        return EMBEDDING_DIMENSIONS

    def calculate(self, chunk: str) -> bytes:
        """
        Compute an embedding for the given text and return it as float32 bytes.
        """
        vec = self._model.encode(chunk, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimension {arr.shape[0]} for {EMBEDDING_MODEL_ID}; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return arr.tobytes()

    def calculate_batch(self, chunks: list[str]) -> list[bytes]:
        vecs = self._model.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=len(chunks),
            show_progress_bar=False,
        )
        results = []

        for vec in vecs:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] != EMBEDDING_DIMENSIONS:
                raise ValueError(f"Unexpected embedding dimension {arr.shape[0]}")
            results.append(arr.tobytes())
        return results
