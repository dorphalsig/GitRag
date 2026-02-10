from __future__ import annotations

from array import array
from unittest import mock

import pytest
from sqlalchemy import create_engine, text

from Chunk import Chunk
from Persist import DBConfig, LibsqlConfig, PersistInLibsql
from PersistPostgres import PersistInPostgres


def _bootstrap_libsql_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE chunks (
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
            """
        ))
        conn.execute(text("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, chunk)"))


def _chunk(*, repo: str, path: str, body: str, branch: str | None, vec: list[float]) -> Chunk:
    c = Chunk(
        chunk=body,
        repo=repo,
        branch=branch,
        path=path,
        language="python",
        start_rc=(0, 0),
        end_rc=(0, len(body)),
        start_bytes=0,
        end_bytes=len(body.encode("utf-8")),
    )
    object.__setattr__(c, "embeddings", array("f", vec).tobytes())
    return c


def test_chunk_id_changes_across_branches() -> None:
    main_chunk = _chunk(repo="org/repo", branch="main", path="a.py", body="print('x')", vec=[1.0, 0.0])
    dev_chunk = _chunk(repo="org/repo", branch="dev", path="a.py", body="print('x')", vec=[1.0, 0.0])
    none_chunk = _chunk(repo="org/repo", branch=None, path="a.py", body="print('x')", vec=[1.0, 0.0])

    assert main_chunk.id() != dev_chunk.id()
    assert main_chunk.id() != none_chunk.id()


def test_persist_and_search_with_repo_and_branch_filters() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = LibsqlConfig.from_parts(database_url="file:branchdb")
    adapter = PersistInLibsql(cfg=cfg, dim=2, engine=engine)

    chunks = [
        _chunk(repo="org/repo", branch="main", path="main.py", body="keyword alpha", vec=[1.0, 0.0]),
        _chunk(repo="org/repo", branch="dev", path="dev.py", body="keyword beta", vec=[1.0, 0.0]),
        _chunk(repo="org/repo", branch=None, path="detached.py", body="keyword detached", vec=[1.0, 0.0]),
        _chunk(repo="org/other", branch="main", path="other.py", body="keyword gamma", vec=[1.0, 0.0]),
    ]
    adapter.persist_batch(chunks)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT branch FROM chunks WHERE path = :path"), {"path": "main.py"}).mappings().one()
        null_row = conn.execute(text("SELECT branch FROM chunks WHERE path = :path"), {"path": "detached.py"}).mappings().one()
    assert row["branch"] == "main"
    assert null_row["branch"] is None

    by_repo_branch = adapter.search([1.0, 0.0], "keyword", limit=10, repo="org/repo", branch="main")
    assert [c.path for c in by_repo_branch] == ["main.py"]

    by_repo_only = adapter.search([1.0, 0.0], "keyword", limit=10, repo="org/repo")
    assert {c.path for c in by_repo_only} == {"main.py", "dev.py", "detached.py"}

    by_wrong_repo = adapter.search([1.0, 0.0], "keyword", limit=10, repo="missing/repo")
    assert by_wrong_repo == []

    by_wrong_branch = adapter.search([1.0, 0.0], "keyword", limit=10, repo="org/repo", branch="release")
    assert by_wrong_branch == []


def test_postgres_persist_and_search_include_branch_filters() -> None:
    engine = mock.MagicMock()

    begin_ctx = mock.MagicMock()
    begin_conn = mock.MagicMock()
    begin_ctx.__enter__.return_value = begin_conn
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx

    connect_ctx = mock.MagicMock()
    connect_conn = mock.MagicMock()
    connect_ctx.__enter__.return_value = connect_conn
    connect_ctx.__exit__.return_value = False
    engine.connect.return_value = connect_ctx

    fake_row = {
        "id": "id1",
        "path": "a.py",
        "repo": "org/repo",
        "branch": "main",
        "chunk": "keyword",
        "embedding": [1.0, 0.0],
    }
    connect_conn.execute.return_value.mappings.return_value.all.return_value = [fake_row]

    cfg = DBConfig(provider="postgres", url="postgresql://localhost/db", table_map={})
    adapter = PersistInPostgres(cfg=cfg, dim=2, engine=engine)

    c = _chunk(repo="org/repo", branch="main", path="a.py", body="keyword", vec=[1.0, 0.0])
    adapter.persist_batch([c])
    results = adapter.search([1.0, 0.0], "keyword", limit=5, repo="org/repo", branch="main")

    assert len(results) == 1
    assert results[0].branch == "main"

    search_sql = str(connect_conn.execute.call_args.args[0])
    search_params = connect_conn.execute.call_args.args[1]
    assert "branch = :branch" in search_sql
    assert "repo = :repo" in search_sql
    assert search_params["repo"] == "org/repo"
    assert search_params["branch"] == "main"

    persist_params = begin_conn.execute.call_args_list[-1].args[1]
    assert persist_params["branch"] == "main"


def test_chunk_calculate_embeddings_variants() -> None:
    class GoodCalc:
        def calculate(self, _: str):
            return memoryview(array("f", [1.0, 2.0]).tobytes())

    class BadCalc:
        def calculate(self, _: str):
            return "bad"

    c = Chunk(
        chunk="body",
        repo="r",
        path="p.py",
        language="python",
        start_rc=(0, 0),
        end_rc=(0, 4),
        start_bytes=0,
        end_bytes=4,
    )
    c.calculate_embeddings(GoodCalc())
    assert isinstance(c.embeddings, bytes)

    with pytest.raises(TypeError):
        c.calculate_embeddings(BadCalc())


def test_libsql_delete_close_and_vector_similarity_edges() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    _bootstrap_libsql_schema(engine)
    cfg = LibsqlConfig.from_parts(database_url="file:branchdb2")
    adapter = PersistInLibsql(cfg=cfg, dim=2, engine=engine)

    adapter.delete_batch([])
    adapter.close()

    c = _chunk(repo="org/repo", branch="main", path="x.py", body="x", vec=[1.0, 0.0])
    adapter.persist_batch([c])
    adapter.delete_batch(["x.py"])

    assert PersistInLibsql._vector_similarity([], None) == 0.0
    assert PersistInLibsql._vector_similarity([0.0, 0.0], array("f", [1.0, 0.0]).tobytes()) == 0.0


def test_postgres_normalized_url_and_delete_batch() -> None:
    engine = mock.MagicMock()
    begin_ctx = mock.MagicMock()
    begin_conn = mock.MagicMock()
    begin_ctx.__enter__.return_value = begin_conn
    begin_ctx.__exit__.return_value = False
    engine.begin.return_value = begin_ctx

    cfg = DBConfig(provider="postgres", url="postgres://localhost/db", table_map={})
    adapter = PersistInPostgres(cfg=cfg, dim=2, engine=engine)

    assert adapter._normalized_url().startswith("postgresql+psycopg2://")
    adapter.delete_batch([])
    adapter.delete_batch(["a.py"])
    assert begin_conn.execute.call_count >= 3
