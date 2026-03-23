import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from Chunker.Chunk import Chunk  # type: ignore
from Persistence.Persist import LibsqlConfig, create_persistence_adapter  # type: ignore


def _build_chunk(repo: str, path: str) -> Chunk:
    chunk = Chunk(
        chunk="print('hello libsql')\n",
        repo=repo,
        path=path,
        language="python",
        start_rc=(1, 0),
        end_rc=(1, 18),
        start_bytes=0,
        end_bytes=18,
    )
    vec = array("f", [0.0] * 768)
    vec[0] = 1.0
    object.__setattr__(chunk, "embeddings", vec.tobytes())
    return chunk

class LibsqlPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._db_path = Path(cls._tmpdir.name) / "libsql-test.sqlite3"
        cls._database_url = "libsql://local-test-db"
        cls._auth_token = None
        cls._engine = create_engine(f"sqlite+pysqlite:///{cls._db_path}", future=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._engine.dispose()
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.repo = f"tests/libsql/{uuid4().hex}"
        self.path = "src/test_chunk.py"
        cfg = LibsqlConfig.from_parts(
            database_url=self._database_url,
            auth_token=self._auth_token,
        )
        self.adapter = create_persistence_adapter(
            "libsql",
            cfg=cfg,
            dim=768,
            engine=self._engine,
        )

    def tearDown(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM chunks WHERE repo = :repo"), {"repo": self.repo})
            conn.execute(text("DELETE FROM chunks_fts WHERE id LIKE :pattern"), {"pattern": f"{self.repo}::%"})

    def test_persist_batch_inserts_rows_and_embeddings(self) -> None:
        chunk = _build_chunk(self.repo, self.path)
        self.adapter.persist_batch([chunk])

        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT repo, path, chunk, length(embedding) AS emb_len FROM chunks WHERE id = :id"
                ),
                {"id": chunk.id()},
            ).mappings().one_or_none()
            self.assertIsNotNone(row)
            self.assertEqual(row["repo"], chunk.repo)
            self.assertEqual(row["path"], chunk.path)
            self.assertEqual(row["chunk"], chunk.chunk)
            self.assertEqual(row["emb_len"], len(chunk.embeddings))

            fts_row = conn.execute(
                text("SELECT chunk FROM chunks_fts WHERE id = :id"),
                {"id": chunk.id()},
            ).mappings().one_or_none()
            self.assertIsNotNone(fts_row)
            self.assertEqual(fts_row["chunk"], chunk.chunk)

    def test_persist_batch_requires_embeddings(self) -> None:
        chunk = _build_chunk(self.repo, self.path)
        object.__setattr__(chunk, "embeddings", None)
        with self.assertRaisesRegex(ValueError, "missing embeddings"):
            self.adapter.persist_batch([chunk])

    def test_delete_batch_removes_chunks(self) -> None:
        chunk = _build_chunk(self.repo, self.path)
        self.adapter.persist_batch([chunk])
        self.adapter.delete_batch([chunk.path])

        with self._engine.connect() as conn:
            remaining = conn.execute(
                text("SELECT COUNT(*) AS c FROM chunks WHERE repo = :repo"),
                {"repo": self.repo},
            ).scalar_one()
            self.assertEqual(remaining, 0)
            fts_remaining = conn.execute(
                text("SELECT COUNT(*) AS c FROM chunks_fts WHERE id = :id"),
                {"id": chunk.id()},
            ).scalar_one()
            self.assertEqual(fts_remaining, 0)

    def test_search_includes_vector_only_candidates(self) -> None:
        keyword_chunk = _build_chunk(self.repo, "src/keyword.py")
        object.__setattr__(keyword_chunk, "chunk", "keywordmatch")
        vector_chunk = _build_chunk(self.repo, "src/vector.py")
        vec = array("f", [0.0] * 768)
        vec[1] = 1.0
        object.__setattr__(vector_chunk, "embeddings", vec.tobytes())
        object.__setattr__(vector_chunk, "chunk", "different text")
        self.adapter.persist_batch([keyword_chunk, vector_chunk])

        query_vec = array("f", [0.0] * 768)
        query_vec[1] = 1.0
        out = self.adapter.search(query_vec, "keywordmatch", limit=2, repo=self.repo)

        paths = {chunk.path for chunk in out}
        self.assertIn("src/keyword.py", paths)
        self.assertIn("src/vector.py", paths)

    def test_search_handles_special_characters_in_fts_query(self) -> None:
        chunk = _build_chunk(self.repo, "src/special.py")
        self.adapter.persist_batch([chunk])
        query_vec = array("f", [0.0] * 768)
        query_vec[0] = 1.0
        out = self.adapter.search(query_vec, 'foo-(bar) "baz"*', limit=1, repo=self.repo)
        self.assertIsInstance(out, list)


class GetIndexedPathsTests(unittest.TestCase):
    """Unit tests for PersistInLibsql.get_indexed_paths using mocks (no real DB)."""

    def _make_adapter(self, mock_engine):
        """Build a PersistInLibsql with a fully mocked engine bypassing bootstrap."""
        from Persistence.Persist import LibsqlConfig, PersistInLibsql  # type: ignore

        cfg = LibsqlConfig.from_parts(database_url="libsql://localhost:8080")
        with patch.object(PersistInLibsql, "_bootstrap"):
            adapter = PersistInLibsql(cfg=cfg, dim=768, engine=mock_engine)
        return adapter

    def _make_engine_returning(self, rows):
        """Return a mock engine whose connect() context manager yields rows."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = rows

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_ctx
        # begin() is called by _bootstrap (patched away), but mock it anyway
        mock_begin_ctx = MagicMock()
        mock_begin_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_begin_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_begin_ctx
        return mock_engine

    def test_returns_empty_set_when_no_chunks(self):
        """get_indexed_paths returns an empty set when the table has no rows."""
        mock_engine = self._make_engine_returning([])
        adapter = self._make_adapter(mock_engine)

        result = adapter.get_indexed_paths()

        self.assertIsInstance(result, set)
        self.assertEqual(result, set())

    def test_returns_paths_from_chunks(self):
        """get_indexed_paths returns the distinct paths present in the chunks table."""
        rows = [("src/foo.py",), ("src/bar.py",), ("README.md",)]
        mock_engine = self._make_engine_returning(rows)
        adapter = self._make_adapter(mock_engine)

        result = adapter.get_indexed_paths()

        self.assertIsInstance(result, set)
        self.assertEqual(result, {"src/foo.py", "src/bar.py", "README.md"})

    def test_queries_with_distinct(self):
        """get_indexed_paths issues a DISTINCT query against the chunks table."""
        mock_engine = self._make_engine_returning([])
        adapter = self._make_adapter(mock_engine)

        adapter.get_indexed_paths()

        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        call_args = mock_conn.execute.call_args
        executed_sql = str(call_args[0][0])
        self.assertIn("DISTINCT", executed_sql.upper())
        self.assertIn("path", executed_sql.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
