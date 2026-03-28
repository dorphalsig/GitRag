"""Shared component loading logic for Indexer and Retriever."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Persistence.Persist import DBConfig, LibsqlConfig, PersistenceAdapter, create_persistence_adapter
from constants import DEFAULT_DB_PROVIDER, DEFAULT_TABLE_NAME

def _env_value(name: str) -> str:
    return (os.environ.get(name) or "").strip()

def resolve_db_cfg() -> DBConfig:
    provider = (_env_value("DB_PROVIDER") or DEFAULT_DB_PROVIDER).lower()
    database_url = _env_value("DATABASE_URL")
    if not database_url and provider == "libsql":
        database_url = _env_value("TURSO_DATABASE_URL")

    if not database_url:
        if provider == "libsql":
            raise RuntimeError("DATABASE_URL (or TURSO_DATABASE_URL for libsql) is required")
        raise RuntimeError(f"DATABASE_URL is required for provider '{provider}'")

    auth_token = _env_value("DB_AUTH_TOKEN") or _env_value("TURSO_AUTH_TOKEN") or None
    if provider == "libsql":
        table = _env_value("LIBSQL_TABLE") or DEFAULT_TABLE_NAME
        fts_table = _env_value("LIBSQL_FTS_TABLE") or None
        return LibsqlConfig.from_parts(
            database_url=database_url,
            auth_token=auth_token,
            table=table,
            fts_table=fts_table,
        )

    return DBConfig(provider=provider, url=database_url, auth_token=auth_token, table_map={})

def load_components() -> Tuple[EmbeddingCalculator, PersistenceAdapter]:
    """Initialize the embedding calculator and persistence layer."""
    calc = EmbeddingCalculator()
    cfg = resolve_db_cfg()
    persist = create_persistence_adapter(cfg.provider, cfg=cfg, dim=calc.dimensions)
    return calc, persist
