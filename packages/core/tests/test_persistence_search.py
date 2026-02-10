from __future__ import annotations

from array import array
from unittest import mock

from sqlalchemy import create_engine, text

from Chunk import Chunk
from Persist import DBConfig, LibsqlConfig, PersistInLibsql
from PersistPostgres import PersistInPostgres


def _chunk(*, repo: str, path: str, body: str, vec: list[float]) -> Chunk:
    c = Chunk(
        chunk=body,
        repo=repo,
        path=path,
        language="python",
        start_rc=(0, 0),
        end_rc=(0, len(body)),
        start_bytes=0,
        end_bytes=len(body.encode("utf-8")),
    )
    object.__setattr__(c, "embeddings", array("f", vec).tobytes())
    return c


def _bootstrap_libsql_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE chunks (
              id TEXT PRIMARY KEY,
              repo TEXT NOT NULL,
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
            """
        ))
        conn.execute(text("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, chunk)"))


def test_libsql_search_returns_ranked_chunks_and_is_consistent_with_insert() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = LibsqlConfig.from_parts(database_url="file:memdb1")
    adapter = PersistInLibsql(cfg=cfg, dim=4, engine=engine)

    c1 = _chunk(repo="r", path="a.py", body="alpha keyword", vec=[1.0, 0.0, 0.0, 0.0])
    c2 = _chunk(repo="r", path="b.py", body="keyword beta", vec=[0.2, 0.9, 0.0, 0.0])
    adapter.persist_batch([c1, c2])

    out = adapter.search(query_embedding=[1.0, 0.0, 0.0, 0.0], query_text="keyword", limit=2)

    assert len(out) == 2
    assert out[0].path == "a.py"
    assert {chunk.path for chunk in out} == {"a.py", "b.py"}


def test_libsql_search_falls_back_to_like_when_fts_errors() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = DBConfig(provider="libsql", url="file:memdb2", table_map={"chunks": "chunks", "chunks_fts": "chunks_fts"})
    adapter = PersistInLibsql(cfg=cfg, dim=2, engine=engine)

    c1 = _chunk(repo="r", path="k.py", body="needle haystack", vec=[0.0, 1.0])
    adapter.persist_batch([c1])

    with mock.patch.object(adapter, "_keyword_search", side_effect=None) as spy:
        # force fallback path by calling the real method with query that still matches LIKE
        rows = PersistInLibsql._keyword_search(adapter, query="needle", limit=5)
        assert rows
        assert spy.call_count == 0

    # Monkeypatch engine connect execute to raise on MATCH and then continue for LIKE.
    real_connect = engine.connect

    class WrapperConn:
        def __init__(self, conn):
            self._conn = conn
            self._raised = False

        def execute(self, statement, params=None):
            sql = str(statement)
            if "MATCH" in sql and not self._raised:
                self._raised = True
                raise RuntimeError("fts unavailable")
            return self._conn.execute(statement, params or {})

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._conn.__exit__(exc_type, exc, tb)

    with mock.patch.object(engine, "connect", side_effect=lambda: WrapperConn(real_connect())):
        results = adapter.search(query_embedding=[0.0, 1.0], query_text="needle", limit=1)

    assert len(results) == 1
    assert results[0].path == "k.py"


def test_postgres_search_executes_hybrid_query_and_maps_chunks() -> None:
    engine = mock.MagicMock()
    begin_ctx = mock.MagicMock()
    begin_ctx.__enter__.return_value = mock.MagicMock()
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx

    connect_ctx = mock.MagicMock()
    fake_row = {
        "id": "id1",
        "path": "x.py",
        "repo": "repo",
        "chunk": "keyword hit",
        "embedding": [1.0, 0.0],
    }
    connect_ctx.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = [fake_row]
    connect_ctx.__exit__.return_value = False
    engine.connect.return_value = connect_ctx

    cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
    adapter = PersistInPostgres(cfg=cfg, dim=2, engine=engine)

    results = adapter.search(query_embedding=[1.0, 0.0], query_text="keyword", limit=3)

    assert len(results) == 1
    assert results[0].path == "x.py"
    called_sql = str(connect_ctx.__enter__.return_value.execute.call_args.args[0])
    assert "hybrid_rank" in called_sql
    params = connect_ctx.__enter__.return_value.execute.call_args.args[1]
    assert params["limit"] == 3
    assert params["query_text"] == "keyword"


def test_libsql_delete_batch_deletes_chunks_and_fts_rows() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = LibsqlConfig.from_parts(database_url="file:memdb3")
    adapter = PersistInLibsql(cfg=cfg, dim=2, engine=engine)

    c1 = _chunk(repo="r", path="to-delete.py", body="delete me", vec=[1.0, 0.0])
    adapter.persist_batch([c1])
    adapter.delete_batch(["to-delete.py"])

    with engine.connect() as conn:
        remaining = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
        fts_remaining = conn.execute(text("SELECT COUNT(*) FROM chunks_fts")).scalar_one()
    assert remaining == 0
    assert fts_remaining == 0


def test_libsql_vector_similarity_and_row_conversion_helpers() -> None:
    sim = PersistInLibsql._vector_similarity([1.0, 0.0], array("f", [1.0, 0.0]).tobytes())
    assert sim > 0.99
    assert PersistInLibsql._vector_similarity([], None) == 0.0

    row = {
        "repo": "repo",
        "path": "x.py",
        "chunk": "hello",
        "language": "python",
        "start_row": 1,
        "start_col": 2,
        "end_row": 1,
        "end_col": 7,
        "start_bytes": 0,
        "end_bytes": 5,
        "embedding": array("f", [0.1, 0.2]).tobytes(),
    }
    chunk = PersistInLibsql._row_to_chunk(row)
    assert chunk.path == "x.py"
    assert chunk.language == "python"


def test_libsql_persist_batch_rejects_missing_embeddings() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = LibsqlConfig.from_parts(database_url="file:memdb4")
    adapter = PersistInLibsql(cfg=cfg, dim=2, engine=engine)

    c1 = Chunk(
        chunk="no embed",
        repo="r",
        path="e.py",
        language="python",
        start_rc=(0, 0),
        end_rc=(0, 8),
        start_bytes=0,
        end_bytes=8,
    )

    try:
        adapter.persist_batch([c1])
        raised = False
    except ValueError:
        raised = True
    assert raised
