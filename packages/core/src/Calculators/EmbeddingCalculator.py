import logging
from typing import Optional, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID, EMBEDDING_BACKEND

logger = logging.getLogger(__name__)


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
        try:
            if self._device == "cpu" or self._device is None:
                try:
                    self._model = SentenceTransformer(
                        EMBEDDING_MODEL_ID,
                        trust_remote_code=True,
                        device="cpu",
                        backend=EMBEDDING_BACKEND,
                    )
                    logger.info("Loaded embedding model with ONNX backend")
                except Exception as e:
                    logger.warning(f"{EMBEDDING_BACKEND} backend failed, falling back to PyTorch: %r", e)
                    self._model = SentenceTransformer(
                        EMBEDDING_MODEL_ID,
                        trust_remote_code=True,
                        device="cpu",
                    )
            elif self._device:
                self._model = SentenceTransformer(
                    EMBEDDING_MODEL_ID,
                    trust_remote_code=True,
                    device=self._device,
                )
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

    def calculate_batch(self, chunks: Iterable[str]) -> Iterable[bytes]:
        assert self._model is not None
        materialized_chunks = list(chunks)
        if not materialized_chunks: return []

        vecs = self._model.encode(
            materialized_chunks,
            batch_size=len(materialized_chunks),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if vecs[0].shape[0] != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding dimension {vecs[0].shape[0]} for {EMBEDDING_MODEL_ID}; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )

        return map(lambda x: np.asarray(x, dtype=np.float32).reshape(-1), vecs)
