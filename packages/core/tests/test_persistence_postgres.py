import pytest
from unittest.mock import MagicMock, patch
from Persistence.PersistPostgres import PersistInPostgres
from Persistence.Persist import DBConfig
from Chunker.Chunk import Chunk

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    engine.connect.return_value.__enter__.return_value = conn
    return engine, conn

def test_postgres_persist_batch(mock_engine):
    engine, conn = mock_engine
    cfg = DBConfig(provider="postgres", url="postgres://localhost/db", table_map={"chunks": "tbl"})

    with patch.object(PersistInPostgres, "_bootstrap"):
        persist = PersistInPostgres(cfg=cfg, engine=engine)

        chunk = Chunk(
            chunk="test content",
            repo="repo",
            path="file.py",
            language="python",
            start_rc=(1, 0),
            end_rc=(2, 0),
            start_bytes=0,
            end_bytes=12,
            embeddings=b"0" * 4096 # 1024 floats * 4 bytes
        )

        persist.persist_batch([chunk])
        assert conn.execute.call_count == 1

def test_postgres_delete_batch(mock_engine):
    engine, conn = mock_engine
    cfg = DBConfig(provider="postgres", url="postgres://localhost/db", table_map={"chunks": "tbl"})
    with patch.object(PersistInPostgres, "_bootstrap"):
        persist = PersistInPostgres(cfg=cfg, engine=engine)
        persist.delete_batch(["path1"])
        assert conn.execute.call_count == 1

def test_postgres_invalid_provider():
    cfg = DBConfig(provider="libsql", url="...")
    with pytest.raises(TypeError, match="cfg.provider must be 'postgres'"):
        PersistInPostgres(cfg=cfg)

def test_postgres_url_normalization():
    with patch.object(PersistInPostgres, "_bootstrap"):
        engine = MagicMock()
        cfg1 = DBConfig(provider="postgres", url="postgres://localhost/db")
        p1 = PersistInPostgres(cfg=cfg1, engine=engine)
        assert p1._normalized_url() == "postgresql+psycopg2://localhost/db"

        cfg2 = DBConfig(provider="postgres", url="postgresql://localhost/db")
        p2 = PersistInPostgres(cfg=cfg2, engine=engine)
        assert p2._normalized_url() == "postgresql+psycopg2://localhost/db"

        cfg3 = DBConfig(provider="postgres", url="postgresql+other://localhost/db")
        p3 = PersistInPostgres(cfg=cfg3, engine=engine)
        assert p3._normalized_url() == "postgresql+other://localhost/db"

def test_postgres_decode_embedding_mismatch():
    engine = MagicMock()
    cfg = DBConfig(provider="postgres", url="postgres://localhost/db")
    with patch.object(PersistInPostgres, "_bootstrap"):
        persist = PersistInPostgres(cfg=cfg, engine=engine, dim=1024)
        with pytest.raises(ValueError, match="Embedding size mismatch"):
            persist._decode_embedding(b"0" * 10)
