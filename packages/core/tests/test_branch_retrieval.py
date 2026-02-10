from __future__ import annotations

from array import array

from Chunk import Chunk
from Retriever import Retriever


class StubEmbeddingCalculator:
    def calculate(self, _: str) -> bytes:
        return array("f", [1.0, 0.0]).tobytes()


class SpyPersistence:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def persist_batch(self, chunks):
        return None

    def delete_batch(self, paths):
        return None

    def search(self, query_embedding, query_text, limit=10, repo=None, branch=None):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "query_text": query_text,
                "limit": limit,
                "repo": repo,
                "branch": branch,
            }
        )
        return [
            Chunk(
                chunk="hit",
                repo=repo or "repo",
                branch=branch,
                path="src/a.py",
                language="python",
                start_rc=(0, 0),
                end_rc=(0, 3),
                start_bytes=0,
                end_bytes=3,
            )
        ]


def test_retrieve_forwards_repo_and_branch_filters() -> None:
    persistence = SpyPersistence()
    retriever = Retriever(
        persistence=persistence,
        embedding_calculator=StubEmbeddingCalculator(),
        initial_limit=7,
    )

    out = retriever.retrieve("query", top_k=3, repo="my-repo", branch="main")

    assert len(out) == 1
    assert persistence.calls[0]["repo"] == "my-repo"
    assert persistence.calls[0]["branch"] == "main"
    assert persistence.calls[0]["query_text"] == "query"


def test_retrieve_defaults_repo_and_branch_to_none() -> None:
    persistence = SpyPersistence()
    retriever = Retriever(
        persistence=persistence,
        embedding_calculator=StubEmbeddingCalculator(),
    )

    out = retriever.retrieve("query")

    assert len(out) == 1
    assert persistence.calls[0]["repo"] is None
    assert persistence.calls[0]["branch"] is None


from types import SimpleNamespace
from unittest import mock

import pytest

from Retriever import Qwen3Reranker


class StubReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def score(self, query: str, candidates):
        return list(self.scores)


def test_retrieve_with_reranker_and_mismatch_error() -> None:
    class TwoHitPersistence(SpyPersistence):
        def search(self, query_embedding, query_text, limit=10, repo=None, branch=None):
            super().search(query_embedding, query_text, limit=limit, repo=repo, branch=branch)
            return [
                Chunk(chunk="a", repo="r", path="a.py", language="python", start_rc=(0, 0), end_rc=(0, 1), start_bytes=0, end_bytes=1),
                Chunk(chunk="b", repo="r", path="b.py", language="python", start_rc=(0, 0), end_rc=(0, 1), start_bytes=0, end_bytes=1),
            ]

    persistence = TwoHitPersistence()
    retriever = Retriever(
        persistence=persistence,
        embedding_calculator=StubEmbeddingCalculator(),
        reranker=StubReranker([0.1]),
        initial_limit=2,
    )
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=2)


def test_retrieve_empty_query_short_circuits() -> None:
    persistence = SpyPersistence()
    retriever = Retriever(persistence=persistence, embedding_calculator=StubEmbeddingCalculator())
    assert retriever.retrieve("   ") == []
    assert persistence.calls == []


def test_qwen3_reranker_singleton_load_and_score() -> None:
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
        scores = r1.score("query", [Chunk(chunk="a", repo="r", path="a.py", language="python", start_rc=(0, 0), end_rc=(0, 1), start_bytes=0, end_bytes=1)])

    assert tok_loader.call_count == 1
    assert model_loader.call_count == 1
    assert r2 is not None
    assert scores == [0.7]


def test_qwen3_reranker_load_failure_is_wrapped() -> None:
    with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("boom")):
        Qwen3Reranker._model = None
        Qwen3Reranker._tokenizer = None
        Qwen3Reranker._load_error = None
        Qwen3Reranker._loaded_model_name = None
        Qwen3Reranker._loaded_attn_implementation = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None

        with pytest.raises(RuntimeError, match="Failed to load Qwen3 reranker model"):
            Qwen3Reranker(model_name="broken-model")
