"""Hybrid retrieval orchestration with optional Qwen3 reranking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker.Chunk import Chunk
from Persistence.Persist import PersistenceAdapter
from constants import (
    DEFAULT_INITIAL_RETRIEVAL_LIMIT,
    MAX_INITIAL_RETRIEVAL_LIMIT,
    DEFAULT_RERANK_TASK_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    RERANK_BATCH_SIZE,
    RETRIEVAL_QUERY_PREFIX, EMBEDDING_BACKEND,
)

LOG4J_FORMAT = "%(asctime)s %(levelname)-5s %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG4J_FORMAT,
    datefmt=DATE_FORMAT
)
logger = logging.getLogger("indexer")


class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[Chunk]) -> List[float]: ...


class Qwen3Reranker:
    """Reranker backed by `Qwen/Qwen3-Reranker-0.6B`.

    The heavy model objects are cached at class scope so multiple retrievers
    can share one loaded model instance.
    """

    _model: Any = None
    _lock = Lock()

    def __init__(
            self,
            model_name: str = DEFAULT_RERANKER_MODEL,
            task_instruction: str | None = None
    ) -> None:
        self._model_name = model_name
        self._task_instruction = task_instruction or DEFAULT_RERANK_TASK_INSTRUCTION
        self._batch_size = max(1, RERANK_BATCH_SIZE)
        self._ensure_loaded(model_name=model_name)

    @classmethod
    def _ensure_loaded(cls, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        if cls._model is not None and cls._loaded_model_name == model_name:
            return
        with cls._lock:
            if cls._model is not None and cls._loaded_model_name == model_name:
                return
            try:
                cls._model = CrossEncoder(
                    model_name,
                    trust_remote_code=True,
                    device="cpu",
                    backend=EMBEDDING_BACKEND,
                )
                cls._loaded_model_name = model_name
            except Exception as e:
                logger.warning("ONNX/OPENVINO backend failed for reranker, falling back to PyTorch: %r", e)
                cls._model = CrossEncoder(model_name, trust_remote_code=True, device="cpu")
                cls._loaded_model_name = model_name

    def score(self, query: str, candidates: Sequence[Chunk]) -> List[float]:
        if not candidates:
            return []
        pairs = [
            (f"Instruct: {self._task_instruction}\nQuery: {query}", c.chunk)
            for c in candidates
        ]
        scores = self.__class__._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


@dataclass
class Retriever:
    persistence: PersistenceAdapter
    embedding_calculator: EmbeddingCalculator
    reranker: Optional[Reranker] = None
    initial_limit: int = DEFAULT_INITIAL_RETRIEVAL_LIMIT

    def retrieve(self, query: str, *, top_k: int = 10, repo: str | None = None, branch: str | None = None, ) \
            -> List[Chunk]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        query_embedding = self._calculate_query_embedding(RETRIEVAL_QUERY_PREFIX + normalized_query)
        initial_limit = max(top_k, min(max(self.initial_limit, top_k), MAX_INITIAL_RETRIEVAL_LIMIT))
        candidates = self.persistence.search(
            query_embedding=query_embedding,
            query_text=normalized_query,
            limit=initial_limit,
            repo=repo,
            branch=branch,
        )
        if not candidates:
            return []

        if self.reranker is None:
            return candidates[:top_k]

        scores = self.reranker.score(normalized_query, candidates)
        if len(scores) != len(candidates):
            raise ValueError(f"Reranker returned mismatched score length: {len(scores)} vs {len(candidates)}")
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    def _calculate_query_embedding(self, query: str) -> np.ndarray:
        raw = self.embedding_calculator.calculate(query)
        if isinstance(raw, (bytearray, memoryview)):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            raise TypeError(f"EmbeddingCalculator.calculate must return bytes, got {type(raw)}")

        return np.frombuffer(raw, dtype=np.float32)
