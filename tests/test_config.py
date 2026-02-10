import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if "sentence_transformers" not in sys.modules:
    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            return [0.0] * 1024

    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer
    )

if "chunker" not in sys.modules:
    sys.modules["chunker"] = types.SimpleNamespace(chunk_file=lambda path, repo: [])

if "text_detection" not in sys.modules:
    class _StubBinaryDetector:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["text_detection"] = types.SimpleNamespace(BinaryDetector=_StubBinaryDetector)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Indexer  # type: ignore


class DBConfigResolutionTests(unittest.TestCase):
    def test_libsql_uses_database_url_when_present(self):
        with mock.patch.dict(
            "os.environ",
            {
                "DB_PROVIDER": "libsql",
                "DATABASE_URL": "libsql://new-style",
                "DB_AUTH_TOKEN": "token-1",
                "LIBSQL_TABLE": "chunks_custom",
                "LIBSQL_FTS_TABLE": "chunks_custom_fts",
            },
            clear=True,
        ):
            cfg = Indexer._resolve_db_cfg()

        self.assertEqual(cfg.provider, "libsql")
        self.assertEqual(cfg.url, "libsql://new-style")
        self.assertEqual(cfg.auth_token, "token-1")
        self.assertEqual(cfg.table_map["chunks"], "chunks_custom")
        self.assertEqual(cfg.table_map["chunks_fts"], "chunks_custom_fts")

    def test_libsql_falls_back_to_legacy_turso_variables(self):
        with mock.patch.dict(
            "os.environ",
            {
                "DB_PROVIDER": "libsql",
                "TURSO_DATABASE_URL": "libsql://legacy",
                "TURSO_AUTH_TOKEN": "legacy-token",
            },
            clear=True,
        ):
            cfg = Indexer._resolve_db_cfg()

        self.assertEqual(cfg.provider, "libsql")
        self.assertEqual(cfg.url, "libsql://legacy")
        self.assertEqual(cfg.auth_token, "legacy-token")
        self.assertEqual(cfg.table_map["chunks"], "chunks")
        self.assertEqual(cfg.table_map["chunks_fts"], "chunks_fts")

    def test_postgres_reads_database_url_and_provider(self):
        with mock.patch.dict(
            "os.environ",
            {
                "DB_PROVIDER": "postgres",
                "DATABASE_URL": "postgresql://user:pass@localhost:5432/app",
            },
            clear=True,
        ):
            cfg = Indexer._resolve_db_cfg()

        self.assertEqual(cfg.provider, "postgres")
        self.assertEqual(cfg.url, "postgresql://user:pass@localhost:5432/app")
        self.assertEqual(cfg.table_map, {})


if __name__ == "__main__":
    unittest.main()
