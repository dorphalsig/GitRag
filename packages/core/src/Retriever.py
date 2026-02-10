"""Hybrid retrieval orchestration with optional Qwen3 reranking."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Optional, Protocol, Sequence

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunk import Chunk
from Persist import PersistenceAdapter


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
    _loaded_model_name: str | None = None
    _loaded_attn_implementation: str | None = None
    _lock = Lock()

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        task_instruction: str | None = None,
        attn_implementation: str = "eager",
    ) -> None:
        self._model_name = model_name
        self._task_instruction = task_instruction or (
            "Given a user query and a code/document chunk, output a relevance score for retrieval reranking."
        )
        self._attn_implementation = attn_implementation
        self._ensure_loaded(model_name=model_name, attn_implementation=attn_implementation)

    @classmethod
    def _ensure_loaded(
        cls,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        attn_implementation: str = "eager",
    ) -> None:
        requested = (model_name, attn_implementation)
        cached = (cls._loaded_model_name, cls._loaded_attn_implementation)
        if cls._model is not None and cls._tokenizer is not None and cached == requested:
            return
        with cls._lock:
            cached = (cls._loaded_model_name, cls._loaded_attn_implementation)
            if cls._model is not None and cls._tokenizer is not None and cached == requested:
                return
            if cls._load_error is not None and cached == requested:
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
            except Exception as exc:  # pragma: no cover - environment-dependent
                cls._load_error = exc
                cls._loaded_model_name = model_name
                cls._loaded_attn_implementation = attn_implementation
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
        encoded = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        import torch

        with torch.no_grad():
            outputs = model(**encoded)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            return [0.0 for _ in candidates]
        flat = logits.squeeze(-1).detach().cpu().tolist()
        if isinstance(flat, float):
            return [flat]
        return [float(v) for v in flat]


@dataclass
class Retriever:
    persistence: PersistenceAdapter
    embedding_calculator: EmbeddingCalculator
    reranker: Optional[Reranker] = None
    initial_limit: int = 50

    def retrieve(self, query: str, *, top_k: int = 10) -> List[Chunk]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        query_embedding = self._calculate_query_embedding(normalized_query)
        candidates = self.persistence.search(
            query_embedding=query_embedding,
            query_text=normalized_query,
            limit=max(self.initial_limit, top_k),
        )
        if not candidates:
            return []

        if self.reranker is None:
            return candidates[:top_k]

        scores = self.reranker.score(normalized_query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned mismatched score length")
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    def _calculate_query_embedding(self, query: str) -> List[float]:
        raw = self.embedding_calculator.calculate(query)
        if isinstance(raw, (bytearray, memoryview)):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            raise TypeError("EmbeddingCalculator.calculate must return bytes")

        vals = array("f")
        vals.frombytes(raw)
        return list(vals)
