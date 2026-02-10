import sys
import unittest
from array import array
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Chunk import Chunk  # type: ignore
from Persist import DBConfig, create_persistence_adapter  # type: ignore
from PersistPostgres import PersistInPostgres  # type: ignore


class PostgresPersistTests(unittest.TestCase):
    def _fake_engine(self):
        engine = mock.MagicMock()
        conn = mock.MagicMock()
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = conn
        ctx.__exit__.return_value = False
        engine.begin.return_value = ctx
        return engine, conn

    def _chunk(self) -> Chunk:
        c = Chunk(
            chunk="print('hello postgres')",
            repo="r",
            path="p.py",
            language="python",
            start_rc=(0, 0),
            end_rc=(0, 1),
            start_bytes=0,
            end_bytes=1,
        )
        vals = array("f", [0.25] * 1024)
        object.__setattr__(c, "embeddings", vals.tobytes())
        return c

    def test_bootstrap_executes_vector_extension_and_schema_ddl(self):
        engine, conn = self._fake_engine()
        cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})

        PersistInPostgres(cfg=cfg, dim=1024, engine=engine)

        sql_texts = [str(call.args[0]) for call in conn.execute.call_args_list]
        self.assertTrue(any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in sql_texts))
        self.assertTrue(any("embedding vector(1024)" in s for s in sql_texts))
        self.assertTrue(any("search_vector tsvector" in s for s in sql_texts))
        self.assertTrue(any("USING GIN (search_vector)" in s for s in sql_texts))

    def test_persist_batch_emits_upsert_with_tsvector_and_vector_param(self):
        engine, conn = self._fake_engine()
        cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
        adapter = PersistInPostgres(cfg=cfg, dim=1024, engine=engine)
        conn.execute.reset_mock()  # ignore bootstrap DDL calls

        chunk = self._chunk()
        adapter.persist_batch([chunk])

        self.assertEqual(conn.execute.call_count, 1)
        stmt = conn.execute.call_args.args[0]
        params = conn.execute.call_args.args[1]
        sql = str(stmt)
        self.assertIn("VALUES (:id, :path, :repo, :chunk, :embedding, to_tsvector('english', :chunk))", sql)
        self.assertIn("ON CONFLICT(id) DO UPDATE", sql)
        self.assertEqual(params["path"], "p.py")
        self.assertEqual(len(params["embedding"]), 1024)
        self.assertIsInstance(params["embedding"][0], float)

    def test_delete_batch_uses_path_expanding_parameter(self):
        engine, conn = self._fake_engine()
        cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
        adapter = PersistInPostgres(cfg=cfg, dim=1024, engine=engine)
        conn.execute.reset_mock()

        adapter.delete_batch(["a.py", "b.py"])

        stmt = conn.execute.call_args.args[0]
        params = conn.execute.call_args.args[1]
        self.assertIn("DELETE FROM chunks WHERE path IN", str(stmt))
        self.assertEqual(params["paths"], ("a.py", "b.py"))

    def test_registry_can_create_postgres_adapter(self):
        engine, _ = self._fake_engine()
        cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
        adapter = create_persistence_adapter("postgres", cfg=cfg, dim=1024, engine=engine)
        self.assertIsInstance(adapter, PersistInPostgres)

    def test_normalized_url_prefers_postgresql_psycopg2(self):
        engine, _ = self._fake_engine()
        cfg = DBConfig(provider="postgres", url="postgres://u:p@h:5432/db", table_map={})
        adapter = PersistInPostgres(cfg=cfg, dim=1024, engine=engine)
        self.assertEqual(adapter._normalized_url(), "postgresql+psycopg2://u:p@h:5432/db")

    def test_build_engine_uses_normalized_url(self):
        cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
        with mock.patch("PersistPostgres.create_engine") as mocked_create_engine:
            mocked_create_engine.return_value = mock.MagicMock()
            adapter = PersistInPostgres(cfg=cfg, dim=1024, engine_factory=lambda: mock.MagicMock())
            adapter._build_engine()
        called_url = mocked_create_engine.call_args.args[0]
        self.assertEqual(called_url, "postgresql+psycopg2://localhost/db")


if __name__ == "__main__":
    unittest.main()
