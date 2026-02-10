import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import refactor_imports


def _stub_indexer_deps() -> None:
    if "numpy" not in sys.modules:
        fake_np = types.SimpleNamespace(asarray=lambda data, dtype=None: data, float32="float32")
        sys.modules["numpy"] = fake_np

    if "sentence_transformers" not in sys.modules:
        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, normalize_embeddings=True):
                return [[0.0] * 3 for _ in texts]

        sys.modules["sentence_transformers"] = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )

    if "chunker" not in sys.modules:
        sys.modules["chunker"] = types.SimpleNamespace(chunk_file=lambda path, repo: [])

    if "text_detection" not in sys.modules:
        class StubBinaryDetector:
            def __init__(self, *args, **kwargs):
                pass

            def is_binary(self, path):
                return False

        sys.modules["text_detection"] = types.SimpleNamespace(BinaryDetector=StubBinaryDetector)


def test_import_indexer_module():
    _stub_indexer_deps()
    indexer = refactor_imports.import_indexer_module()
    assert hasattr(indexer, "_collect_changes")


def test_import_persist_and_initialize_libsql_adapter():
    engine = MagicMock()
    connect_ctx = MagicMock()
    connect_ctx.__enter__.return_value = MagicMock()
    connect_ctx.__exit__.return_value = False
    engine.connect.return_value = connect_ctx

    adapter = refactor_imports.initialize_libsql_adapter(
        database_url="libsql://example",
        auth_token="token",
        dim=4,
        engine=engine,
    )
    assert adapter is not None


def test_import_persist_can_initialize_postgres_adapter():
    engine = MagicMock()
    begin_ctx = MagicMock()
    begin_ctx.__enter__.return_value = MagicMock()
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx

    adapter = refactor_imports.initialize_postgres_adapter(
        url="postgresql://localhost/db",
        dim=8,
        engine=engine,
    )
    assert adapter is not None
