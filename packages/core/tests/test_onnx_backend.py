import os
from unittest.mock import MagicMock, patch

import pytest

from Calculators.EmbeddingCalculator import EmbeddingCalculator


def test_embedding_calculator_onnx():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "onnx"}), \
         patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:

        mock_instance = mock_st.return_value
        calc = EmbeddingCalculator()

        mock_st.assert_called_once()
        args, kwargs = mock_st.call_args
        assert kwargs.get("backend") == "onnx"


def test_reranker_onnx():
    from Retriever import Qwen3Reranker
    # Force reload of class variables for testing
    Qwen3Reranker._model = None
    Qwen3Reranker._tokenizer = None

    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "onnx"}), \
         patch("sentence_transformers.CrossEncoder") as mock_ce:

        mock_instance = mock_ce.return_value
        mock_instance.tokenizer = MagicMock()

        # We need to patch the lock to avoid actual loading issues if any
        with patch("Retriever.Lock"):
            reranker = Qwen3Reranker()

        mock_ce.assert_called_once()
        args, kwargs = mock_ce.call_args
        assert kwargs.get("backend") == "onnx"
