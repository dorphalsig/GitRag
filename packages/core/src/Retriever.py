"""Hybrid retrieval orchestration with optional Qwen3 reranking."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np
from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker.Chunk import Chunk
from Persistence.Persist import PersistenceAdapter
from constants import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_INITIAL_RETRIEVAL_LIMIT,
    MAX_INITIAL_RETRIEVAL_LIMIT,
    DEFAULT_RERANK_TASK_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    RERANK_BATCH_SIZE,
    RETRIEVAL_QUERY_PREFIX,
)

class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[Chunk]) -> List[float]: ...


class Qwen3Reranker:
    """Reranker backed by `Qwen/Qwen3-Reranker-0.6B`.

    The heavy model objects are cached at class scope so multiple retrievers
    can share one loaded model instance.
    """

    _model: Any = None
    _tokenizer: Any = None
    _load_error: Optional[Exception] = None
    _loaded_model_name: Optional[str] = None
    _loaded_attn_implementation: Optional[str] = None
    _load_error_model_name: Optional[str] = None
    _load_error_attn_implementation: Optional[str] = None
    _lock = Lock()

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        task_instruction: str | None = None,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    ) -> None:
        self._model_name = model_name
        self._task_instruction = task_instruction or DEFAULT_RERANK_TASK_INSTRUCTION
        self._attn_implementation = attn_implementation
        self._batch_size = max(1, RERANK_BATCH_SIZE)
        self._ensure_loaded(model_name=model_name, attn_implementation=attn_implementation)

    @classmethod
    def _ensure_loaded(
        cls,
        model_name: str = DEFAULT_RERANKER_MODEL,
        attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    ) -> None:
        if (
            cls._model is not None
            and cls._tokenizer is not None
            and cls._loaded_model_name == model_name
            and cls._loaded_attn_implementation == attn_implementation
        ):
            return
        with cls._lock:
            if (
                cls._model is not None
                and cls._tokenizer is not None
                and cls._loaded_model_name == model_name
                and cls._loaded_attn_implementation == attn_implementation
            ):
                return
            if (
                cls._load_error is not None
                and cls._load_error_model_name == model_name
                and cls._load_error_attn_implementation == attn_implementation
            ):
                raise RuntimeError("Qwen3 reranker unavailable") from cls._load_error
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                cls._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                cls._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                )
                cls._model.eval()
                cls._loaded_model_name = model_name
                cls._loaded_attn_implementation = attn_implementation
                cls._load_error = None
                cls._load_error_model_name = None
                cls._load_error_attn_implementation = None
            except Exception as exc:  # pragma: no cover - environment-dependent
                cls._load_error = exc
                cls._load_error_model_name = model_name
                cls._load_error_attn_implementation = attn_implementation
                raise RuntimeError("Failed to load Qwen3 reranker model") from exc

    def score(self, query: str, candidates: Sequence[Chunk]) -> List[float]:
        if not candidates:
            return []
        model = self.__class__._model
        tokenizer = self.__class__._tokenizer
        if model is None or tokenizer is None:
            return [0.0 for _ in candidates]

        pairs = [
            (
                f"Instruct: {self._task_instruction}\nQuery: {query}",
                candidate.chunk,
            )
            for candidate in candidates
        ]
        import torch

        all_scores: list[float] = []
        batch_size = max(1, int(getattr(self, "_batch_size", RERANK_BATCH_SIZE)))
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.inference_mode():
                outputs = model(**encoded)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                all_scores.extend([0.0 for _ in batch_pairs])
                continue
            flat = logits.squeeze(-1).detach().cpu().tolist()
            if isinstance(flat, float):
                all_scores.append(float(flat))
            else:
                all_scores.extend(float(v) for v in flat)
        return all_scores


@dataclass
class Retriever:
    persistence: PersistenceAdapter
    embedding_calculator: EmbeddingCalculator
    reranker: Optional[Reranker] = None
    initial_limit: int = DEFAULT_INITIAL_RETRIEVAL_LIMIT

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]:
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
