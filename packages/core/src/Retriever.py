"""Hybrid retrieval orchestration with optional Qwen3 reranking."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import numpy as np
from array import array
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker.Chunk import Chunk
from ComponentLoader import load_components
from Persistence.Persist import PersistenceAdapter
from constants import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_INITIAL_RETRIEVAL_LIMIT,
    DEFAULT_RERANK_TASK_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
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
                backend = os.getenv("RETRIEVAL_OPTIMIZER", "torch").lower()
                if backend == "onnx":
                    from sentence_transformers import CrossEncoder
                    cls._model = CrossEncoder(model_name, trust_remote_code=True, backend="onnx")
                    cls._tokenizer = cls._model.tokenizer
                else:
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
        try:
            from sentence_transformers import CrossEncoder
            if isinstance(model, CrossEncoder):
                preds = model.predict(pairs)
                if isinstance(preds, (float, np.float32, np.float64, int)):
                    return [float(preds)]
                return [float(v) for v in preds]
        except ImportError:
            pass

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
        candidates = self.persistence.search(
            query_embedding=query_embedding,
            query_text=normalized_query,
            limit=max(self.initial_limit, top_k),
            repo=repo,
            branch=branch,
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


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Query the retrieval pipeline.")
    parser.add_argument("query", nargs="?", help="The search query.")
    parser.add_argument("--repo", help="Optional repository filter.")
    parser.add_argument("--branch", help="Optional branch filter.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--server", action="store_true", help="Start the MCP server (SSE).")
    parser.add_argument("--port", type=int, default=7860, help="Port for the MCP server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the MCP server.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("retriever")

    if not args.server and not args.query:
        parser.print_help()
        sys.exit(1)

    calc, persist = load_components(args.repo)
    reranker = Qwen3Reranker()
    retriever = Retriever(persist, calc, reranker=reranker)

    if args.server:
        logger.info("Starting MCP Server on %s:%d", args.host, args.port)
        try:
            from gitrag_mcp_server.server import create_mcp_server
        except ImportError as e:
            logger.error("Failed to import MCP server: %s", e)
            sys.exit(1)

        mcp = create_mcp_server(retriever=retriever)
        mcp.run(transport="sse", port=args.port, host=args.host)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    results = retriever.retrieve(args.query, top_k=args.top_k, repo=args.repo, branch=args.branch)
    output = []
    for chunk in results:
        payload = asdict(chunk)
        if "embeddings" in payload:
            payload["embeddings"] = None
        output.append(payload)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
