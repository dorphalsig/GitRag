"""Helpers for validating import and initialization flows after monorepo refactor."""

from __future__ import annotations

import importlib
from typing import Any


def import_indexer_module() -> Any:
    """Import and return the Indexer module."""
    return importlib.import_module("Indexer")


def import_persist_module() -> Any:
    """Import and return the Persist module."""
    return importlib.import_module("Persist")


def initialize_libsql_adapter(*, database_url: str, auth_token: str | None, dim: int, engine: Any) -> Any:
    """Create a libsql persistence adapter instance using Persist helpers."""
    persist = import_persist_module()
    cfg = persist.LibsqlConfig.from_parts(database_url=database_url, auth_token=auth_token)
    return persist.create_persistence_adapter("libsql", cfg=cfg, dim=dim, engine=engine)


def initialize_postgres_adapter(*, url: str, dim: int, engine: Any) -> Any:
    """Create a postgres persistence adapter instance using Persist helpers."""
    persist = import_persist_module()
    cfg = persist.DBConfig(provider="postgres", url=url)
    return persist.create_persistence_adapter("postgres", cfg=cfg, dim=dim, engine=engine)
