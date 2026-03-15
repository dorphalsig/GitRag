import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID

logger = logging.getLogger(__name__)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n, minimum 32."""
    if n <= 32:
        return 32
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class EmbeddingCalculator:
    """
    EmbeddingCalculator implementation backed by Qwen3-Embedding-0.6B.

    Returns float32 embeddings as bytes (little-endian).
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen3 embedding model."""
        try:
            # Attempt ONNX if on CPU
            if self._device == "cpu" or self._device is None:
                try:
                    self._model = SentenceTransformer(
                        EMBEDDING_MODEL_ID,
                        trust_remote_code=True,
                        device="cpu",
                        backend="onnx"
                    )
                    logger.info("Loaded embedding model with ONNX backend")
                except Exception as e:
                    logger.warning("Failed to load model with ONNX backend, falling back to PyTorch: %r", e)
                    self._model = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True, device="cpu")
            elif self._device:
                self._model = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True, device=self._device)
            else:
                self._model = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True)

            assert self._model is not None
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
        assert self._model is not None
        vec = self._model.encode(chunk, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimension {arr.shape[0]} for {EMBEDDING_MODEL_ID}; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return arr.tobytes()

    def calculate_batch(self, chunks: list[str]) -> list[bytes]:
        assert self._model is not None
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
