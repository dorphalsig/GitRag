import os
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID, MAX_SEQ_LENGTH

def test_embedding_calculator_init_torch():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            
            calc = EmbeddingCalculator(device="cpu")
            
            mock_st.assert_called_once_with(EMBEDDING_MODEL_ID, trust_remote_code=True, device="cpu")
            assert mock_instance.max_seq_length == MAX_SEQ_LENGTH

def test_embedding_calculator_init_error():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model load failed")
            
            with pytest.raises(RuntimeError, match="Failed to load embedding model"):
                EmbeddingCalculator()

def test_embedding_calculator_dimensions():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            calc = EmbeddingCalculator()
            assert calc.dimensions == EMBEDDING_DIMENSIONS

def test_embedding_calculator_calculate():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            
            # Mock a valid embedding
            valid_embedding = np.random.rand(EMBEDDING_DIMENSIONS).astype(np.float32)
            mock_instance.encode.return_value = valid_embedding
            
            calc = EmbeddingCalculator()
            result = calc.calculate("test chunk")
            
            assert isinstance(result, bytes)
            # Verify numpy array conversion
            restored = np.frombuffer(result, dtype=np.float32)
            np.testing.assert_array_equal(restored, valid_embedding)
            mock_instance.encode.assert_called_once_with("test chunk", normalize_embeddings=True)

def test_embedding_calculator_calculate_wrong_dim():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            
            # Mock an invalid embedding (wrong dimensions)
            invalid_embedding = np.random.rand(EMBEDDING_DIMENSIONS - 1).astype(np.float32)
            mock_instance.encode.return_value = invalid_embedding
            
            calc = EmbeddingCalculator()
            with pytest.raises(ValueError, match="Unexpected embedding dimension"):
                calc.calculate("test chunk")

def test_embedding_calculator_calculate_batch():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            
            # Mock a valid batch of embeddings
            valid_embeddings = [
                np.random.rand(EMBEDDING_DIMENSIONS).astype(np.float32),
                np.random.rand(EMBEDDING_DIMENSIONS).astype(np.float32)
            ]
            mock_instance.encode.return_value = valid_embeddings
            
            calc = EmbeddingCalculator()
            results = calc.calculate_batch(["chunk 1", "chunk 2"])
            
            assert len(results) == 2
            assert all(isinstance(r, bytes) for r in results)
            
            mock_instance.encode.assert_called_once_with(
                ["chunk 1", "chunk 2"],
                normalize_embeddings=True,
                batch_size=2,
                show_progress_bar=False
            )

def test_embedding_calculator_calculate_batch_wrong_dim():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("Calculators.EmbeddingCalculator.SentenceTransformer") as mock_st:
            mock_instance = mock_st.return_value
            
            # Mock an invalid embedding in batch
            invalid_embeddings = [
                np.random.rand(EMBEDDING_DIMENSIONS - 1).astype(np.float32)
            ]
            mock_instance.encode.return_value = invalid_embeddings
            
            calc = EmbeddingCalculator()
            with pytest.raises(ValueError, match="Unexpected embedding dimension"):
                calc.calculate_batch(["test chunk"])