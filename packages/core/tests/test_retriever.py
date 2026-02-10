from __future__ import annotations

from array import array
from types import SimpleNamespace
from unittest import mock

import pytest

from Chunk import Chunk
from Retriever import Qwen3Reranker, Retriever


class StubEmbeddingCalculator:
    def __init__(self, values: list[float]):
        self.values = values
        self.calls: list[str] = []

    def calculate(self, chunk: str) -> bytes:
        self.calls.append(chunk)
        return array("f", self.values).tobytes()


class StubPersistence:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self.calls = []

    def persist_batch(self, chunks):
        return None

    def delete_batch(self, paths):
        return None

    def search(self, query_embedding, query_text, limit=10):
        self.calls.append({"embedding": query_embedding, "query_text": query_text, "limit": limit})
        return list(self._chunks)


class StubReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls = []

    def score(self, query: str, candidates):
        self.calls.append((query, [c.path for c in candidates]))
        return list(self.scores)


def _chunk(path: str, chunk_text: str) -> Chunk:
    c = Chunk(
        chunk=chunk_text,
        repo="repo",
        path=path,
        language="python",
        start_rc=(0, 0),
        end_rc=(0, len(chunk_text)),
        start_bytes=0,
        end_bytes=len(chunk_text.encode("utf-8")),
    )
    object.__setattr__(c, "embeddings", array("f", [0.1, 0.2]).tobytes())
    return c


def test_retriever_uses_search_and_reranks() -> None:
    chunks = [_chunk("a.py", "alpha"), _chunk("b.py", "beta")]
    persistence = StubPersistence(chunks)
    embed = StubEmbeddingCalculator([1.0, 0.0])
    reranker = StubReranker([0.1, 0.9])

    retriever = Retriever(
        persistence=persistence,
        embedding_calculator=embed,
        reranker=reranker,
        initial_limit=50,
    )

    out = retriever.retrieve("find beta", top_k=1)

    assert [c.path for c in out] == ["b.py"]
    assert persistence.calls[0]["limit"] == 50
    assert persistence.calls[0]["query_text"] == "find beta"
    assert embed.calls == ["find beta"]
    assert reranker.calls[0][0] == "find beta"


def test_retriever_without_reranker_returns_first_k() -> None:
    chunks = [_chunk("a.py", "alpha"), _chunk("b.py", "beta")]
    retriever = Retriever(
        persistence=StubPersistence(chunks),
        embedding_calculator=StubEmbeddingCalculator([1.0, 0.0]),
        reranker=None,
        initial_limit=5,
    )

    out = retriever.retrieve("q", top_k=1)
    assert [c.path for c in out] == ["a.py"]


def test_retriever_empty_query_returns_empty() -> None:
    retriever = Retriever(
        persistence=StubPersistence([_chunk("a.py", "alpha")]),
        embedding_calculator=StubEmbeddingCalculator([1.0]),
    )
    assert retriever.retrieve("   ") == []


def test_retriever_raises_on_mismatched_rerank_scores() -> None:
    chunks = [_chunk("a.py", "alpha"), _chunk("b.py", "beta")]
    retriever = Retriever(
        persistence=StubPersistence(chunks),
        embedding_calculator=StubEmbeddingCalculator([1.0, 0.0]),
        reranker=StubReranker([0.2]),
    )
    with pytest.raises(ValueError, match="mismatched score length"):
        retriever.retrieve("q")


def test_retriever_calculate_query_embedding_requires_bytes() -> None:
    class BadEmbedding:
        def calculate(self, chunk: str):
            return [1.0, 2.0]

    retriever = Retriever(
        persistence=StubPersistence([]),
        embedding_calculator=BadEmbedding(),
    )
    with pytest.raises(TypeError, match="must return bytes"):
        retriever.retrieve("q")


def test_qwen3_reranker_singleton_loads_model_once() -> None:
    fake_tokenizer = mock.MagicMock()
    fake_model = mock.MagicMock()
    fake_model.eval = mock.MagicMock()
    fake_logits = mock.MagicMock()
    fake_logits.squeeze.return_value.detach.return_value.cpu.return_value.tolist.return_value = [0.7]
    fake_model.return_value = SimpleNamespace(logits=fake_logits)

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer) as tok_loader, mock.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model
    ) as model_loader:
        Qwen3Reranker._model = None
        Qwen3Reranker._tokenizer = None
        Qwen3Reranker._load_error = None
        Qwen3Reranker._loaded_model_name = None
        Qwen3Reranker._loaded_attn_implementation = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None

        r1 = Qwen3Reranker()
        r2 = Qwen3Reranker()
        scores = r1.score("query", [_chunk("a.py", "alpha")])

    assert tok_loader.call_count == 1
    assert model_loader.call_count == 1
    assert model_loader.call_args.kwargs["attn_implementation"] == "eager"
    assert fake_model.eval.call_count == 1
    assert r2 is not None
    assert scores == [0.7]


def test_qwen3_reranker_returns_zeros_when_logits_missing() -> None:
    reranker = Qwen3Reranker.__new__(Qwen3Reranker)
    reranker._task_instruction = "task"
    Qwen3Reranker._model = mock.MagicMock(return_value=SimpleNamespace())
    Qwen3Reranker._tokenizer = mock.MagicMock(return_value={"input_ids": [1]})

    scores = reranker.score("q", [_chunk("a.py", "alpha")])
    assert scores == [0.0]


def test_qwen3_reranker_load_failure_sets_error() -> None:
    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("boom")):
        Qwen3Reranker._model = None
        Qwen3Reranker._tokenizer = None
        Qwen3Reranker._load_error = None
        Qwen3Reranker._loaded_model_name = None
        Qwen3Reranker._loaded_attn_implementation = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None
        with pytest.raises(RuntimeError, match="Failed to load"):
            Qwen3Reranker()


def test_qwen3_reranker_forwards_attention_implementation() -> None:
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=mock.MagicMock()), mock.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock.MagicMock(eval=mock.MagicMock())
    ) as model_loader:
        Qwen3Reranker._model = None
        Qwen3Reranker._tokenizer = None
        Qwen3Reranker._load_error = None
        Qwen3Reranker._loaded_model_name = None
        Qwen3Reranker._loaded_attn_implementation = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None

        Qwen3Reranker(attn_implementation="sdpa")

    assert model_loader.call_args.kwargs["attn_implementation"] == "sdpa"


def test_qwen3_reranker_reloads_when_model_configuration_changes() -> None:
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=mock.MagicMock()) as tok_loader, mock.patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock.MagicMock(eval=mock.MagicMock()),
    ) as model_loader:
        Qwen3Reranker._model = None
        Qwen3Reranker._tokenizer = None
        Qwen3Reranker._load_error = None
        Qwen3Reranker._loaded_model_name = None
        Qwen3Reranker._loaded_attn_implementation = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None

        Qwen3Reranker(model_name="model-a", attn_implementation="eager")
        Qwen3Reranker(model_name="model-b", attn_implementation="sdpa")

    assert tok_loader.call_count == 2
    assert model_loader.call_count == 2
    assert model_loader.call_args.kwargs["attn_implementation"] == "sdpa"
    assert model_loader.call_args.args[0] == "model-b"
