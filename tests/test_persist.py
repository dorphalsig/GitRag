import json
import os
import sys
import types
import unittest
from typing import Optional

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
import requests


class StubDatabase:
    def __init__(self):
        self.queries = []
        self.select_result = []

    def query(self, account_id, database_id, sql, params=None):
        self.queries.append((sql.strip(), tuple(params) if params else None))
        if sql.strip().startswith("SELECT id"):
            return types.SimpleNamespace(result=self.select_result)
        return types.SimpleNamespace(result=[])


class StubCloudflare:
    def __init__(self):
        self.d1 = types.SimpleNamespace(database=StubDatabase())


class PersistTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PersistConfig(account_id="acct", vectorize_index="idx", d1_database_id="db")
        self.client = StubCloudflare()
        self.persist = PersistInVectorize(
            cfg=self.cfg,
            dim=3,
            client=self.client,
            api_token="stub",
        )
        self._index_exists = False
        self._metadata: list[str] = []
        self._deleted_ids: list[str] = []
        self._upsert_payload: bytes | None = None
        self._last_mutation: str | None = None

        def _raise_http(status: int, path: str):
            resp = requests.Response()
            resp.status_code = status
            resp.url = path
            raise requests.exceptions.HTTPError(response=resp)

        def fake_get(instance: PersistInVectorize, path: str):
            if path.endswith(f"/indexes/{self.cfg.vectorize_index}"):
                if not self._index_exists:
                    _raise_http(404, path)
                return {"result": {}}
            if path.endswith("/metadata-indexes"):
                return {"result": {"indexes": [{"property_name": m} for m in self._metadata]}}
            if path.endswith("/info"):
                processed = self._last_mutation or ""
                return {"result": {"processed_up_to_mutation": processed}}
            return {}

        def fake_post(
            instance: PersistInVectorize,
            path: str,
            *,
            json_data: Optional[dict] = None,
            content: Optional[bytes] = None,
            headers: Optional[dict] = None,
        ):
            if path.endswith("/indexes"):
                self._index_exists = True
                return {"result": {"name": self.cfg.vectorize_index}}
            if path.endswith("/metadata-indexes"):
                prop = (json_data or {}).get("property_name")
                if prop and prop not in self._metadata:
                    self._metadata.append(prop)
                return {"result": {}}
            if path.endswith("/upsert"):
                self._upsert_payload = content
                self._last_mutation = "m1"
                return {"mutation_id": "m1"}
            if path.endswith("/delete_by_ids"):
                ids = (json_data or {}).get("ids") or []
                self._deleted_ids = list(ids)
                return {"result": {}}
            return {"result": {}}

        self.persist._vectorize_get = types.MethodType(fake_get, self.persist)
        self.persist._vectorize_post = types.MethodType(fake_post, self.persist)

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

        self.assertTrue(self._index_exists)
        self.assertCountEqual(self._metadata, ["repo", "path", "language"])
        stored = self._upsert_payload
        self.assertIsNotNone(stored)
        payload = stored.decode("utf-8").strip()
        self.assertIn("\"id\"", payload)
        self.assertIn("\"values\"", payload)

        queries = self.client.d1.database.queries
        self.assertTrue(any("INSERT INTO" in sql for sql, _ in queries))
        self.assertTrue(any("UPDATE" in sql for sql, _ in queries))

    def test_delete_batch_deletes_ids(self):
        chunk_id = "abc"
        self.client.d1.database.select_result = [{"id": chunk_id}]
        self.persist.delete_batch(["file.txt"])
        self.assertEqual(self._deleted_ids, [chunk_id])
        delete_queries = [sql for sql, _ in self.client.d1.database.queries if sql.startswith("DELETE")]
        self.assertTrue(delete_queries)

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
        if missing:
            self.skipTest(
                "Missing environment variables for Cloudflare persistence test: "
                + ", ".join(missing)
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
