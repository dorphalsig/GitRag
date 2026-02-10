import os
import unittest
from array import array
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from Chunk import Chunk  # type: ignore
from Persist import LibsqlConfig, create_persistence_adapter  # type: ignore


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise unittest.SkipTest(f"{name} env variable required for libSQL persistence tests")
    return value


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


def _bootstrap_schema(engine) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          repo TEXT NOT NULL,
          branch TEXT,
          path TEXT NOT NULL,
          language TEXT NOT NULL,
          start_row INTEGER NOT NULL,
          start_col INTEGER NOT NULL,
          end_row INTEGER NOT NULL,
          end_col INTEGER NOT NULL,
          start_bytes INTEGER NOT NULL,
          end_bytes INTEGER NOT NULL,
          chunk TEXT NOT NULL,
          status TEXT NOT NULL,
          mutation_id TEXT,
          embedding BLOB NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks(repo)",
        "CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path)",
        "CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks(repo, path)",
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id, chunk)",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


class LibsqlPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._database_url = _required_env("TURSO_DATABASE_URL")
        cls._auth_token = _required_env("TURSO_AUTH_TOKEN")
        cls._engine = create_engine(
            f"sqlite+{cls._database_url}?secure=true",
            future=True,
            connect_args={"auth_token": cls._auth_token},
        )
        _bootstrap_schema(cls._engine)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._engine.dispose()

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
