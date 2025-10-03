import logging
from typing import Optional

import numpy as np

from EmbeddingCalculator import EmbeddingCalculator

try:
    # SentenceTransformers is the supported way per model card
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise ImportError("sentence-transformers is required for CodeRankCalculator") from e


logger = logging.getLogger(__name__)


class CodeRankCalculator(EmbeddingCalculator):
    """
    EmbeddingCalculator implementation backed by CodeRankEmbed.

    Loads the CodeRankEmbed model (primary: nomic-ai/CodeRankEmbed, fallback: cornst
    ack/CodeRankEmbed as referenced in the official model card), and returns
    float32 embeddings as bytes (little-endian).

    Methods are kept small; dimension size is determined on first encode and cached.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self._model = None
        self._dim: Optional[int] = None
        self._device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load the model with a deterministic fallback; log which repo succeeded."""
        # Primary path (the official card is hosted under nomic-ai)
        # Fallback to cornst... because the card demonstrates that in usage.
        tried = []
        for repo in ("nomic-ai/CodeRankEmbed", "cornstack/CodeRankEmbed"):
            try:
                if self._device:
                    self._model = SentenceTransformer(repo, trust_remote_code=True, device=self._device)
                else:
                    self._model = SentenceTransformer(repo, trust_remote_code=True)
                logger.info("Loaded CodeRankEmbed from %s", repo)
                return
            except Exception as e:
                tried.append((repo, repr(e)))
        msg = "Failed to load CodeRankEmbed. Tried: " + ", ".join([r for r, _ in tried])
        logger.error("%s; last error: %s", msg, tried[-1][1] if tried else "N/A")
        raise RuntimeError(msg)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension (lazy-initialized)."""
        if self._dim is None:
            # Determine by encoding a minimal string
            vec = self._model.encode([" "], normalize_embeddings=True)
            self._dim = int(vec.shape[-1])
            logger.info("CodeRankEmbed dimension detected: %d", self._dim)
        return self._dim

    def calculate(self, chunk: str) -> bytes:
        """
        Compute a code embedding for the given text and return it as float32 bytes.

        The CodeRankEmbed card requires a special prefix only for *queries*.
        For code documents (our chunks), we encode directly.
        """
        # single-text encode returns 1 x D array
        vec = self._model.encode([chunk], normalize_embeddings=True)
        if self._dim is None:
            self._dim = int(vec.shape[-1])
            logger.info("CodeRankEmbed dimension detected: %d", self._dim)
        # Ensure float32 little-endian bytes
        arr = np.asarray(vec[0], dtype=np.float32)
        return arr.tobytes()
