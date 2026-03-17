import numpy as np
import pytest

import Calculators.EmbeddingCalculator as calculator_module


class FakeSentenceTransformer:
    def __init__(self, model_id, trust_remote_code=True, device=None, **kwargs):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.kwargs = kwargs

    def encode(self, text, normalize_embeddings=True, batch_size=None, show_progress_bar=None):
        assert normalize_embeddings is True
        if isinstance(text, str):
            return np.ones(calculator_module.EMBEDDING_DIMENSIONS, dtype=np.float32)
        elif isinstance(text, list):
            return [np.ones(calculator_module.EMBEDDING_DIMENSIONS, dtype=np.float32) for _ in text]
        else:
            raise TypeError("text must be str or list")


class OnnxFallbackSentenceTransformer:
    def __init__(self, model_id, trust_remote_code=True, device=None, backend=None):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.backend = backend
        if backend == "onnx":
            raise RuntimeError("onnx failed")

    def encode(self, text, normalize_embeddings=True, **kwargs):
        return np.ones(calculator_module.EMBEDDING_DIMENSIONS, dtype=np.float32)


class BadDimSentenceTransformer(FakeSentenceTransformer):
    def encode(self, text, normalize_embeddings=True, batch_size=None, show_progress_bar=None):
        return np.ones(10, dtype=np.float32)


class FailingSentenceTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_embedding_returns_expected_dimension_vector(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FakeSentenceTransformer)

    calc = calculator_module.EmbeddingCalculator()
    raw = calc.calculate("def hello():\n    return 'world'\n")
    vec = np.frombuffer(raw, dtype=np.float32)

    assert calc.dimensions == calculator_module.EMBEDDING_DIMENSIONS
    assert len(vec) == calculator_module.EMBEDDING_DIMENSIONS


def test_initializes_with_expected_model_id(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FakeSentenceTransformer)

    calc = calculator_module.EmbeddingCalculator(device="cpu")

    assert calc._model is not None
    assert calc._model.model_id == calculator_module.EMBEDDING_MODEL_ID
    assert calc._model.device == "cpu"


def test_raises_runtime_error_when_model_fails_to_load(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", FailingSentenceTransformer)

    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        calculator_module.EmbeddingCalculator()


def test_raises_when_embedding_dimension_is_unexpected(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", BadDimSentenceTransformer)

    calc = calculator_module.EmbeddingCalculator()
    with pytest.raises(ValueError, match="Unexpected embedding dimension"):
        calc.calculate("x")


def test_falls_back_to_pytorch_when_onnx_fails(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", OnnxFallbackSentenceTransformer)

    # Should not raise - falls back to PyTorch
    calc = calculator_module.EmbeddingCalculator(device="cpu")

    assert calc._model is not None
    raw = calc.calculate("test")
    assert len(np.frombuffer(raw, dtype=np.float32)) == calculator_module.EMBEDDING_DIMENSIONS


class SeqLengthTrackingSentenceTransformer:
    """Tracks max_seq_length changes."""
    def __init__(self, model_id, trust_remote_code=True, device=None, backend=None, **kwargs):
        self.model_id = model_id
        self.max_seq_length = 1024
        self.seq_length_history = []

        class MockTokenizer:
            def encode(self, text):
                return [0] * (len(text) // 4 + 1)
            
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    return {"input_ids": [self.encode(texts)]}
                return {"input_ids": [self.encode(t) for t in texts]}
        self.tokenizer = MockTokenizer()

    def encode(self, texts, normalize_embeddings=True, batch_size=None, show_progress_bar=False):
        self.seq_length_history.append(self.max_seq_length)
        if isinstance(texts, list):
            return [np.ones(calculator_module.EMBEDDING_DIMENSIONS, dtype=np.float32) for _ in texts]
        return np.ones(calculator_module.EMBEDDING_DIMENSIONS, dtype=np.float32)


def test_dynamic_seq_length_adjusts_for_batch(monkeypatch):
    monkeypatch.setattr(calculator_module, "SentenceTransformer", SeqLengthTrackingSentenceTransformer)

    calc = calculator_module.EmbeddingCalculator()

    # Short chunks - should use smaller seq length
    short_chunks = ["hello", "world", "test"]
    calc.calculate_batch(short_chunks)

    # Seq length should be reduced (not 1024)
    assert calc._model is not None
    assert calc._model.seq_length_history[-1] < 1024
