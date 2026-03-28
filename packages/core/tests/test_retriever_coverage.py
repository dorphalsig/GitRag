import pytest
from unittest.mock import MagicMock, patch

from Retriever import Qwen3Reranker, Retriever
from constants import DEFAULT_RERANKER_MODEL, DEFAULT_ATTN_IMPLEMENTATION
from Chunker.Chunk import Chunk

def test_qwen3_reranker_ensure_loaded_lock_fast_path():
    reranker = Qwen3Reranker()
    # It's already loaded, calling it again should hit the fast path
    Qwen3Reranker._ensure_loaded()

def test_qwen3_reranker_ensure_loaded_error_path():
    Qwen3Reranker._load_error = Exception("test error")
    Qwen3Reranker._load_error_model_name = DEFAULT_RERANKER_MODEL
    Qwen3Reranker._load_error_attn_implementation = DEFAULT_ATTN_IMPLEMENTATION

    # Clear cached model state if any
    old_model = Qwen3Reranker._model
    old_tokenizer = Qwen3Reranker._tokenizer
    Qwen3Reranker._model = None
    Qwen3Reranker._tokenizer = None

    try:
        with pytest.raises(RuntimeError, match="Qwen3 reranker unavailable"):
            Qwen3Reranker._ensure_loaded()
    finally:
        # Reset
        Qwen3Reranker._load_error = None
        Qwen3Reranker._load_error_model_name = None
        Qwen3Reranker._load_error_attn_implementation = None
        Qwen3Reranker._model = old_model
        Qwen3Reranker._tokenizer = old_tokenizer

def test_qwen3_reranker_score_fallback_logic():
    # Test fallback logic in score when it's not a CrossEncoder
    reranker = Qwen3Reranker()
    Qwen3Reranker._model = MagicMock()
    Qwen3Reranker._tokenizer = MagicMock()

    # Mock tokenizer to return some encoded dummy
    Qwen3Reranker._tokenizer.return_value = {"input_ids": [1, 2, 3]}

    mock_outputs = MagicMock()
    mock_logits = MagicMock()
    mock_logits.squeeze.return_value.detach.return_value.cpu.return_value.tolist.return_value = [0.9, 0.1]
    mock_outputs.logits = mock_logits
    Qwen3Reranker._model.return_value = mock_outputs

    c1 = Chunk(chunk="c1", repo="r", path="p", language="l", start_rc=(1,0), end_rc=(2,0), start_bytes=0, end_bytes=1)
    c2 = Chunk(chunk="c2", repo="r", path="p", language="l", start_rc=(1,0), end_rc=(2,0), start_bytes=0, end_bytes=1)

    scores = reranker.score("query", [c1, c2])
    assert scores == [0.9, 0.1]

    # Float fallback
    mock_logits.squeeze.return_value.detach.return_value.cpu.return_value.tolist.return_value = 0.8
    scores = reranker.score("query", [c1])
    assert scores == [0.8]

    # Logits is None fallback
    mock_outputs.logits = None
    scores = reranker.score("query", [c1, c2])
    assert scores == [0.0, 0.0]

def test_retriever_calculate_embedding_bytearray():
    mock_calc = MagicMock()
    # return bytearray instead of bytes
    mock_calc.calculate.return_value = bytearray(b"\x00\x00\x00\x00")
    retriever = Retriever(MagicMock(), mock_calc)
    res = retriever._calculate_query_embedding("test")
    assert len(res) == 1

def test_retriever_calculate_embedding_invalid_type():
    mock_calc = MagicMock()
    # return str instead of bytes
    mock_calc.calculate.return_value = "invalid"
    retriever = Retriever(MagicMock(), mock_calc)
    with pytest.raises(TypeError, match="must return bytes"):
        retriever._calculate_query_embedding("test")
