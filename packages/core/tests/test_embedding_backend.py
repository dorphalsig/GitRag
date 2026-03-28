import pytest
from Calculators.EmbeddingCalculator import EmbeddingCalculator


@pytest.mark.manual
def test_cpu_embedding_uses_onnx_backend():
    """
    One-time manual check: confirms CPU embedding loads with ONNX backend.

    Run with:  pytest -m manual --no-cov
    Not run in standard CI.

    Requires optimum[onnxruntime] installed (included via sentence-transformers[onnx]).
    """
    calc = EmbeddingCalculator.__new__(EmbeddingCalculator)
    calc._model = None
    calc._device = "cpu"
    EmbeddingCalculator._load_model(calc)

    assert calc._model is not None, "Model failed to load"

    backend = getattr(calc._model, "backend", "")
    model_type = str(type(calc._model)).lower()
    assert "onnx" in backend.lower() or "onnx" in model_type, (
        f"Expected ONNX backend. "
        f"Got: type={type(calc._model)}, backend={backend!r}. "
        "Ensure 'optimum[onnxruntime]' is installed."
    )
