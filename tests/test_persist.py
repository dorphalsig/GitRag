import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Persist import PersistInVectorize, PersistConfig, create_persistence_adapter  # type: ignore
from persistence_registry import (
    register_persistence_adapter,
    unregister_persistence_adapter,
)
from Chunk import Chunk  # type: ignore
import chunker  # type: ignore
class StubDatabase:
    def __init__(self):
        self.queries = []
        self.select_result = []

    def query(self, account_id, database_id, sql, params=None):
        self.queries.append((sql.strip(), tuple(params) if params else None))
        if sql.strip().startswith("SELECT id"):
            return types.SimpleNamespace(result=self.select_result)
        return types.SimpleNamespace(result=[])


class StubMetadataIndex:
    def __init__(self, parent):
        self._parent = parent

    def list(self, name, account_id):
        return types.SimpleNamespace(
            indexes=[{"property_name": m} for m in self._parent.metadata]
        )

    def create(self, name, account_id, index_type, property_name):
        if property_name not in self._parent.metadata:
            self._parent.metadata.append(property_name)
        return types.SimpleNamespace()


class StubVectorizeIndexes:
    def __init__(self):
        self.created = False
        self.metadata: list[str] = []
        self.upserts: list[str] = []
        self.deleted: list[list[str]] = []
        self.processed = ""
        self.metadata_index = StubMetadataIndex(self)

    def get(self, name, account_id):
        if not self.created:
            raise Exception("missing")
        return types.SimpleNamespace()

    def create(self, account_id, name, config):
        self.created = True
        return types.SimpleNamespace()

    def upsert(self, name, account_id, body, unparsable_behavior):
        self.upserts.append(body)
        self.processed = "m1"
        return types.SimpleNamespace(mutation_id="m1")

    def info(self, name, account_id):
        return types.SimpleNamespace(processed_up_to_mutation=self.processed)

    def delete_by_ids(self, name, account_id, ids):
        self.deleted.append(list(ids))
        return types.SimpleNamespace()


class StubCloudflare:
    def __init__(self):
        self.d1 = types.SimpleNamespace(database=StubDatabase())
        self.vectorize = types.SimpleNamespace(indexes=StubVectorizeIndexes())


class PersistTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PersistConfig(account_id="acct", vectorize_index="idx", d1_database_id="db")
        self.client = StubCloudflare()
        self.persist = PersistInVectorize(cfg=self.cfg, dim=3, client=self.client)

    def test_persist_batch_runs_setup_and_commits(self):
        fixture = Path(ROOT) / "tests" / "fixtures" / "Markdown.md"
        source_chunks = chunker.chunk_file(str(fixture), repo="repo")
        self.assertTrue(source_chunks)

        class StubCalc:
            def calculate(self, text: str) -> bytes:
                return text.encode("utf-8")[:8]

        calc = StubCalc()
        for sc in source_chunks:
            sc.calculate_embeddings(calc)
        chunk = source_chunks[0]
        self.persist.persist_batch([chunk])

        indexes = self.client.vectorize.indexes
        self.assertTrue(indexes.created)
        self.assertCountEqual(indexes.metadata, ["repo", "path", "language"])
        stored = indexes.upserts[-1]
        self.assertIsNotNone(stored)
        vectors = stored.get("vectors") if isinstance(stored, dict) else None
        self.assertIsInstance(vectors, list)
        self.assertTrue(vectors)
        entry = vectors[0]
        self.assertIn("id", entry)
        self.assertIn("values", entry)

        queries = self.client.d1.database.queries
        self.assertTrue(any("INSERT INTO" in sql for sql, _ in queries))
        self.assertTrue(any("UPDATE" in sql for sql, _ in queries))
        self.assertTrue(
            any("INSERT INTO chunks_fts" in sql for sql, _ in queries),
            "Expected FTS insert",
        )

    def test_delete_batch_deletes_ids(self):
        chunk_id = "abc"
        self.client.d1.database.select_result = [{"id": chunk_id}]
        self.persist.delete_batch(["file.txt"])
        self.assertEqual(self.client.vectorize.indexes.deleted[-1], [chunk_id])
        delete_queries = [sql for sql, _ in self.client.d1.database.queries if sql.startswith("DELETE")]
        self.assertTrue(delete_queries)
        self.assertTrue(
            any("chunks_fts" in sql for sql in delete_queries),
            "Expected FTS deletes",
        )

    def test_create_persistence_adapter_aliases(self):
        adapter = create_persistence_adapter("cloudflare", cfg=self.cfg, dim=3, client=self.client)
        self.assertIsInstance(adapter, PersistInVectorize)
        with self.assertRaises(ValueError):
            create_persistence_adapter("unknown", cfg=self.cfg, dim=3, client=self.client)

    def test_register_custom_adapter(self):
        created = {}

        class DummyAdapter:
            def __init__(self, *, cfg: PersistConfig, dim: int, **kwargs):
                created["cfg"] = cfg
                created["dim"] = dim
                created["kwargs"] = kwargs

            def persist_batch(self, chunks):
                created["persisted"] = list(chunks)

            def delete_batch(self, paths):
                created["deleted"] = list(paths)

        def factory(*, cfg: PersistConfig, dim: int, **kwargs):
            return DummyAdapter(cfg=cfg, dim=dim, **kwargs)

        register_persistence_adapter("dummy", factory)
        try:
            adapter = create_persistence_adapter("dummy", cfg=self.cfg, dim=7, client=None, extra="value")
            self.assertIsInstance(adapter, DummyAdapter)
            self.assertEqual(created["cfg"], self.cfg)
            self.assertEqual(created["dim"], 7)
            self.assertEqual(created["kwargs"].get("extra"), "value")
            fixture = Path(ROOT) / "tests" / "fixtures" / "Markdown.md"
            adapter.persist_batch([chunker.chunk_file(str(fixture), repo="repo")[0]])
            self.assertIn("persisted", created)
            adapter.delete_batch(["foo"])
            self.assertEqual(created.get("deleted"), ["foo"])
        finally:
            unregister_persistence_adapter("dummy")

    def test_real_cloudflare_persistence_optional(self):
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        vectorize_index = os.getenv("CLOUDFLARE_VECTORIZE_INDEX")
        d1_database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
        token = os.getenv("CLOUDFLARE_API_TOKEN")
        dim_env = os.getenv("GITRAG_EMBED_DIM")
        missing = []
        if not account:
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not vectorize_index:
            missing.append("CLOUDFLARE_VECTORIZE_INDEX")
        if not d1_database_id:
            missing.append("CLOUDFLARE_D1_DATABASE_ID")
        if not token:
            missing.append("CLOUDFLARE_API_TOKEN")
        if not dim_env:
            missing.append("GITRAG_EMBED_DIM")
        self.assertFalse(
            missing,
            "Missing environment variables for Cloudflare persistence test: " + ", ".join(missing),
        )

        dim = int(dim_env)

        fixture = Path(ROOT) / "tests" / "fixtures" / "Markdown.md"
        chunk = chunker.chunk_file(str(fixture), repo="repo")[0]

        class FixedCalc:
            def calculate(self, text: str) -> bytes:
                return b"\x00" * (dim * 4)

        calc = FixedCalc()
        chunk.calculate_embeddings(calc)

        adapter = create_persistence_adapter(
            "cloudflare",
            cfg=PersistConfig(
                account_id=account,
                vectorize_index=vectorize_index,
                d1_database_id=d1_database_id,
            ),
            dim=dim,
        )

        adapter.persist_batch([chunk])


if __name__ == "__main__":
    unittest.main()
