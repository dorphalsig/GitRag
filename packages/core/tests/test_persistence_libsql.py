import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import text
from Persistence.Persist import PersistInLibsql, DBConfig, LibsqlConfig, create_persistence_adapter
from Chunker.Chunk import Chunk

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn

def test_libsql_config_from_parts():
    cfg = LibsqlConfig.from_parts(database_url="libsql://test", auth_token="token", table="my_chunks")
    assert cfg.provider == "libsql"
    assert cfg.url == "libsql://test"
    assert cfg.auth_token == "token"
    assert cfg.table == "my_chunks"
    assert cfg.fts_table == "my_chunks_fts"
    assert cfg.database_url == "libsql://test"
    assert cfg.resolved_fts_table == "my_chunks_fts"

def test_libsql_persist_batch(mock_engine):
    engine, conn = mock_engine
    cfg = LibsqlConfig.from_parts(database_url="libsql://test")
    # Mocking _bootstrap to avoid DDL execution
    with patch.object(PersistInLibsql, "_bootstrap"):
        persist = PersistInLibsql(cfg=cfg, engine=engine)

        chunk = Chunk(
            chunk="test content",
            repo="repo",
            path="file.py",
            language="python",
            start_rc=(1, 0),
            end_rc=(2, 0),
            start_bytes=0,
            end_bytes=12,
            embeddings=b"0" * 4096
        )

        persist.persist_batch([chunk])

        # Verify execute called for insert and FTS refresh
        # 1. conn.execute(insert_stmt, params)
        # 2. conn.execute(delete_fts, params)
        # 3. conn.execute(insert_fts, params)
        assert conn.execute.call_count >= 3

def test_libsql_delete_batch(mock_engine):
    engine, conn = mock_engine
    cfg = LibsqlConfig.from_parts(database_url="libsql://test")
    with patch.object(PersistInLibsql, "_bootstrap"):
        persist = PersistInLibsql(cfg=cfg, engine=engine)
        persist.delete_batch(["path1", "path2"])
        assert conn.execute.call_count == 2

def test_libsql_close(mock_engine):
    engine, conn = mock_engine
    cfg = LibsqlConfig.from_parts(database_url="libsql://test")
    with patch.object(PersistInLibsql, "_bootstrap"):
        persist = PersistInLibsql(cfg=cfg, engine=engine)
        persist.close()
        # engine.dispose should NOT be called because engine was passed in (owns_engine=False)
        assert engine.dispose.call_count == 0

def test_create_persistence_adapter_unsupported():
    cfg = DBConfig(provider="invalid", url="...")
    with pytest.raises(ValueError, match="Unsupported persistence adapter"):
        create_persistence_adapter("invalid", cfg=cfg)

def test_libsql_invalid_provider():
    cfg = DBConfig(provider="postgres", url="...")
    with pytest.raises(TypeError, match="cfg.provider must be 'libsql'"):
        PersistInLibsql(cfg=cfg)

def test_libsql_persist_batch_missing_embeddings(mock_engine):
    engine, conn = mock_engine
    cfg = LibsqlConfig.from_parts(database_url="libsql://test")
    with patch.object(PersistInLibsql, "_bootstrap"):
        persist = PersistInLibsql(cfg=cfg, engine=engine)
        chunk = Chunk(chunk="c", repo="r", path="p", language="l", start_rc=(0,0), end_rc=(0,0), start_bytes=0, end_bytes=0)
        with pytest.raises(ValueError, match="missing embeddings"):
            persist.persist_batch([chunk])
