import numpy as np
import pytest

import src.Calculators.CodeRankCalculator as calculator_module


class FakeSentenceTransformer:
    def __init__(self, model_id, trust_remote_code=True, device=None):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.device = device

    def encode(self, text, normalize_embeddings=True):
        assert normalize_embeddings is True
        assert isinstance(text, str)
        return np.ones(1024, dtype=np.float32)


class BadDimSentenceTransformer(FakeSentenceTransformer):
    def encode(self, text, normalize_embeddings=True):
        return np.ones(10, dtype=np.float32)


class FailingSentenceTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_qwen_embedding_returns_1024_dimension_vector(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FakeSentenceTransformer)

    calc = calculator_module.CodeRankCalculator()
    raw = calc.calculate("def hello():\n    return 'world'\n")
    vec = np.frombuffer(raw, dtype=np.float32)

    assert calc.dimensions == 1024
    assert len(vec) == 1024


def test_initializes_with_expected_model_id(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FakeSentenceTransformer)

    calc = calculator_module.CodeRankCalculator(device="cpu")

    assert calc._model.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert calc._model.device == "cpu"


def test_raises_runtime_error_when_model_fails_to_load(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FailingSentenceTransformer)

    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        calculator_module.CodeRankCalculator()


def test_raises_when_embedding_dimension_is_unexpected(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", BadDimSentenceTransformer)

    calc = calculator_module.CodeRankCalculator()
    with pytest.raises(ValueError, match="Unexpected embedding dimension"):
        calc.calculate("x")
