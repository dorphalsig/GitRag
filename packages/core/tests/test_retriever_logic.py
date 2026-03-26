import os
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from Chunker.Chunk import Chunk
from Retriever import Qwen3Reranker, Retriever

def test_reranker_initialization_torch():
    with patch.dict(os.environ, {"RETRIEVAL_OPTIMIZER": "torch"}, clear=True):
        with patch("transformers.AutoTokenizer") as mock_tokenizer, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model:
            
            Qwen3Reranker._model = None
            Qwen3Reranker._tokenizer = None
            
            reranker = Qwen3Reranker(model_name="test-model")
            
            mock_tokenizer.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)
            mock_model.from_pretrained.assert_called_once()

def test_reranker_score_empty_candidates():
    reranker = Qwen3Reranker()
    scores = reranker.score("query", [])
    assert scores == []

def test_reranker_score_not_loaded():
    Qwen3Reranker._model = None
    Qwen3Reranker._tokenizer = None
    
    # We mock _ensure_loaded so it doesn't load a real model!
    with patch.object(Qwen3Reranker, "_ensure_loaded"):
        reranker = Qwen3Reranker()
        scores = reranker.score("query", [MagicMock()])
        assert scores == [0.0]

def test_reranker_score_cross_encoder():
    Qwen3Reranker._model = MagicMock()
    Qwen3Reranker._tokenizer = MagicMock()
    
    class DummyCrossEncoder:
        def predict(self, pairs):
            return np.array([0.5, 0.8])
            
    Qwen3Reranker._model = DummyCrossEncoder()
    
    with patch("Retriever.Qwen3Reranker._ensure_loaded"):
        # We need to make sure the Retriever uses the real CrossEncoder class for isinstance
        reranker = Qwen3Reranker()
        
        candidates = [MagicMock(chunk="c1"), MagicMock(chunk="c2")]
        
        # We need to patch the sentence_transformers import inside score to use our dummy class
        with patch("sentence_transformers.CrossEncoder", DummyCrossEncoder):
            scores = reranker.score("query", candidates)
            assert scores == [0.5, 0.8]

def test_retriever_empty_query():
    persist = MagicMock()
    calc = MagicMock()
    retriever = Retriever(persist, calc)
    
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

def test_retriever_no_candidates():
    persist = MagicMock()
    calc = MagicMock()
    calc.calculate.return_value = b'0000'
    
    persist.search.return_value = []
    
    retriever = Retriever(persist, calc)
    assert retriever.retrieve("query") == []

def test_retriever_no_reranker():
    persist = MagicMock()
    calc = MagicMock()
    calc.calculate.return_value = b'0000'
    
    c1, c2 = MagicMock(), MagicMock()
    persist.search.return_value = [c1, c2]
    
    retriever = Retriever(persist, calc, reranker=None)
    results = retriever.retrieve("query", top_k=1)
    
    assert len(results) == 1
    assert results[0] == c1

def test_retriever_with_reranker():
    persist = MagicMock()
    calc = MagicMock()
    calc.calculate.return_value = b'0000'
    
    c1, c2 = MagicMock(), MagicMock()
    persist.search.return_value = [c1, c2]
    
    reranker = MagicMock()
    reranker.score.return_value = [0.1, 0.9]
    
    retriever = Retriever(persist, calc, reranker=reranker)
    results = retriever.retrieve("query", top_k=2)
    
    assert len(results) == 2
    assert results[0] == c2
    assert results[1] == c1

def test_retriever_mismatched_scores():
    persist = MagicMock()
    calc = MagicMock()
    calc.calculate.return_value = b'0000'
    
    persist.search.return_value = [MagicMock()]
    
    reranker = MagicMock()
    reranker.score.return_value = []
    
    retriever = Retriever(persist, calc, reranker=reranker)
    with pytest.raises(ValueError, match="Reranker returned mismatched score"):
        retriever.retrieve("query")
