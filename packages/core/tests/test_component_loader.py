import os
from unittest.mock import MagicMock, patch
import pytest

from ComponentLoader import _env_value, resolve_db_cfg, load_components
from Persistence.Persist import DBConfig, LibsqlConfig

def test_env_value():
    with patch.dict(os.environ, {"TEST_KEY": " test_value "}):
        assert _env_value("TEST_KEY") == "test_value"
    assert _env_value("NONEXISTENT") == ""

def test_resolve_db_cfg_libsql_success():
    with patch.dict(os.environ, {
        "DB_PROVIDER": "libsql",
        "DATABASE_URL": "sqlite:///:memory:",
        "DB_AUTH_TOKEN": "token",
        "LIBSQL_TABLE": "my_table",
        "LIBSQL_FTS_TABLE": "my_fts_table"
    }):
        cfg = resolve_db_cfg()
        assert isinstance(cfg, LibsqlConfig)
        assert cfg.provider == "libsql"
        assert cfg.url == "sqlite:///:memory:"
        assert cfg.auth_token == "token"
        assert cfg.table == "my_table"
        assert cfg.fts_table == "my_fts_table"

def test_resolve_db_cfg_libsql_fallback_turso():
    with patch.dict(os.environ, {
        "TURSO_DATABASE_URL": "libsql://test.turso.io",
        "TURSO_AUTH_TOKEN": "turso_token"
    }, clear=True):
        cfg = resolve_db_cfg()
        assert isinstance(cfg, LibsqlConfig)
        assert cfg.url == "libsql://test.turso.io"
        assert cfg.auth_token == "turso_token"

def test_resolve_db_cfg_libsql_missing_url():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="TURSO_DATABASE_URL for libsql"):
            resolve_db_cfg()

def test_resolve_db_cfg_postgres():
    with patch.dict(os.environ, {
        "DB_PROVIDER": "postgres",
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
    }, clear=True):
        cfg = resolve_db_cfg()
        assert not isinstance(cfg, LibsqlConfig)
        assert isinstance(cfg, DBConfig)
        assert cfg.provider == "postgres"
        assert cfg.url == "postgresql://user:pass@localhost/db"

def test_resolve_db_cfg_postgres_missing_url():
    with patch.dict(os.environ, {"DB_PROVIDER": "postgres"}, clear=True):
        with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
            resolve_db_cfg()

@patch("ComponentLoader.EmbeddingCalculator")
@patch("ComponentLoader.create_persistence_adapter")
@patch("ComponentLoader.resolve_db_cfg")
def test_load_components(mock_resolve, mock_create_persist, mock_calc_class):
    mock_calc = mock_calc_class.return_value
    mock_calc.dimensions = 1024

    mock_cfg = MagicMock()
    mock_cfg.provider = "postgres"
    mock_resolve.return_value = mock_cfg

    mock_persist = mock_create_persist.return_value

    calc, persist = load_components("test-repo")

    assert calc == mock_calc
    assert persist == mock_persist
    mock_create_persist.assert_called_once_with("postgres", cfg=mock_cfg, dim=1024)
