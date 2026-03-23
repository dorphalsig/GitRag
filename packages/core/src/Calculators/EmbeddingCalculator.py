import logging
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID

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
        """Load the Qwen3 embedding model."""
        try:
            # Attempt ONNX if on CPU
            if self._device == "cpu" or self._device is None:
                try:
                    kwargs = {
                        "model_id_or_path": EMBEDDING_MODEL_ID,
                        "trust_remote_code": True,
                        "device": "cpu",
                        "backend": "onnx",
                    }
                    if os.path.exists(os.path.join(EMBEDDING_MODEL_ID, "model_optimized.onnx")):
                        kwargs["model_kwargs"] = {"file_name": "model_optimized.onnx"}

                    self._model = SentenceTransformer(**kwargs)
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
        if not chunks:
            return []
        batch_size = max(1, min(EMBEDDING_BATCH_SIZE, len(chunks)))
        results: list[bytes] = []
        for start in range(0, len(chunks), batch_size):
            window = chunks[start : start + batch_size]
            vecs = self._model.encode(
                window,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            for vec in vecs:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.shape[0] != EMBEDDING_DIMENSIONS:
                    raise ValueError(f"Unexpected embedding dimension {arr.shape[0]}")
                results.append(arr.tobytes())
        return results
