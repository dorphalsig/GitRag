"""libSQL-backed persistence adapter implemented with SQLAlchemy."""
from __future__ import annotations

import logging
import math
import re
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

try:  # pragma: no cover - dependency guard for tooling envs
    from sqlalchemy import bindparam, create_engine, text
    from sqlalchemy.engine import Engine
except ModuleNotFoundError as exc:  # pragma: no cover
    bindparam = None  # type: ignore[assignment]
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[assignment]
    SQLALCHEMY_IMPORT_ERROR = exc
else:  # pragma: no cover - success path
    SQLALCHEMY_IMPORT_ERROR = None

from Chunker.Chunk import Chunk
from constants import (
    DEFAULT_FTS_TABLE_SUFFIX,
    DEFAULT_TABLE_NAME,
    EMBEDDING_DIMENSIONS,
    HYBRID_SEARCH_KEYWORD_WEIGHT,
    HYBRID_SEARCH_VECTOR_WEIGHT,
)
from .persistence_registry import get_persistence_adapter, register_persistence_adapter

logger = logging.getLogger(__name__)


class PersistenceAdapter(Protocol):
    def persist_batch(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_batch(self, paths: List[str], repo: str | None = None) -> None: ...

    def search(
        self,
        query_embedding: Any,
        query_text: str,
        limit: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]: ...

    def get_indexed_paths(self, repo: str | None = None) -> set[str]:
        """Return the set of file paths that have already been indexed."""
        ...


@dataclass(frozen=True)
class DBConfig:
    provider: str
    url: str
    auth_token: Optional[str] = None
    table_map: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LibsqlConfig(DBConfig):
    @classmethod
    def from_parts(
        cls,
        *,
        database_url: str,
        auth_token: Optional[str] = None,
        table: str = DEFAULT_TABLE_NAME,
        fts_table: Optional[str] = None,
    ) -> "LibsqlConfig":
        resolved_fts = fts_table or f"{table}{DEFAULT_FTS_TABLE_SUFFIX}"
        return cls(
            provider="libsql",
            url=database_url,
            auth_token=auth_token,
            table_map={
                "chunks": table,
                "chunks_fts": resolved_fts,
            },
        )

    @property
    def database_url(self) -> str:
        return self.url

    @property
    def table(self) -> str:
        return self.table_map.get("chunks", DEFAULT_TABLE_NAME)

    @property
    def fts_table(self) -> Optional[str]:
        return self.table_map.get("chunks_fts")

    @property
    def resolved_fts_table(self) -> str:
        return self.fts_table or f"{self.table}{DEFAULT_FTS_TABLE_SUFFIX}"


class PersistInLibsql(PersistenceAdapter):
    """Persistence adapter backed by libSQL using SQLAlchemy."""

    def __init__(
        self,
        *,
        cfg: DBConfig,
        dim: int = EMBEDDING_DIMENSIONS,
        engine: Optional[Engine] = None,
        engine_factory: Optional[Callable[[], Engine]] = None,
        **_: Any,
    ) -> None:
        if SQLALCHEMY_IMPORT_ERROR is not None:
            raise RuntimeError("SQLAlchemy is required for libsql persistence") from SQLALCHEMY_IMPORT_ERROR
        if cfg.provider != "libsql":
            raise TypeError("cfg.provider must be 'libsql' for PersistInLibsql")
        self._cfg = cfg
        self._dim = dim
        self._table = cfg.table_map.get("chunks", DEFAULT_TABLE_NAME)
        self._fts_table = cfg.table_map.get("chunks_fts", f"{self._table}{DEFAULT_FTS_TABLE_SUFFIX}")
        if engine is not None:
            self._engine = engine
            self._owns_engine = False
        else:
            factory = engine_factory or self._build_engine
            self._engine = factory()
            self._owns_engine = True
        self._bootstrap()

    def persist_batch(self, chunks: Sequence[Chunk]) -> None:
        valid = [chunk for chunk in chunks if chunk is not None]
        if not valid:
            return
        for chunk in valid:
            if chunk.embeddings is None:
                raise ValueError(f"Chunk {chunk.path} missing embeddings")
            if len(chunk.embeddings) != self._dim * 4:
                logger.warning("Embedding size mismatch for chunk %s", chunk.path)

        insert_stmt = text(self._upsert_sql)
        delete_fts_stmt = text(f"DELETE FROM {self._fts_table} WHERE id = :id")
        insert_fts_stmt = text(f"INSERT INTO {self._fts_table} (id, chunk) VALUES (:id, :chunk)")

        all_params = [self._chunk_params(c) for c in valid]
        fts_params = [{"id": p["id"], "chunk": p["chunk"]} for p in all_params]

        with self._engine.begin() as conn:
            conn.execute(insert_stmt, all_params)
            # Bulk FTS update: delete then re-insert to ensure synchronization.
            # SQLite FTS5 doesn't support 'IN' with expanding parameters in all versions
            # as efficiently, so we use a loop for delete if needed, but executemany is better.
            conn.execute(delete_fts_stmt, [{"id": p["id"]} for p in fts_params])
            conn.execute(insert_fts_stmt, fts_params)

    def delete_batch(self, paths: List[str], repo: str | None = None) -> None:
        if not paths:
            return
        where_clause = "path IN :paths"
        params = {"paths": tuple(paths)}
        if repo is not None:
            where_clause += " AND repo = :repo"
            params["repo"] = repo

        delete_stmt = text(
            f"DELETE FROM {self._table} WHERE {where_clause}"
        ).bindparams(bindparam("paths", expanding=True))
        delete_fts_stmt = text(
            f"DELETE FROM {self._fts_table} WHERE id NOT IN (SELECT id FROM {self._table})"
        )

        with self._engine.begin() as conn:
            conn.execute(delete_stmt, params)
            conn.execute(delete_fts_stmt)

    def get_indexed_paths(self, repo: str | None = None) -> set[str]:
        """Return the set of distinct file paths stored in the chunks table."""
        where = ""
        params = {}
        if repo is not None:
            where = " WHERE repo = :repo"
            params["repo"] = repo

        sql = text(f"SELECT DISTINCT path FROM {self._table}{where}")
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row[0] for row in rows if row[0] is not None}

    def close(self) -> None:
        if getattr(self, "_owns_engine", True):
            dispose = getattr(self._engine, "dispose", None)
            if callable(dispose):  # pragma: no branch - defensive
                dispose()

    def search(
        self,
        query_embedding: Any,
        query_text: str,
        limit: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]:
        normalized_limit = max(1, int(limit))
        query = (query_text or "").strip()
        keyword_rows = self._keyword_search(
            query=query,
            limit=normalized_limit * 5,
            repo=repo,
            branch=branch,
        )
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-9:
            query_vec = query_vec / query_norm
        vector_rows = self._vector_search(
            query_embedding=query_vec,
            limit=normalized_limit * 5,
            repo=repo,
            branch=branch,
        )
        if not keyword_rows and not vector_rows:
            return []

        merged: Dict[str, Dict[str, Any]] = {}
        for row in keyword_rows:
            merged[row["id"]] = row
        for row in vector_rows:
            existing = merged.get(row["id"])
            if existing is None:
                merged[row["id"]] = row
            else:
                if existing.get("embedding") is None and row.get("embedding") is not None:
                    existing["embedding"] = row["embedding"]
                existing["vector_score"] = row.get("vector_score", 0.0)

        candidates = list(merged.values())
        keyword_ranked = sorted(range(len(candidates)), key=lambda i: float(candidates[i].get("keyword_score") or 0.0), reverse=True)
        vector_ranked = sorted(range(len(candidates)), key=lambda i: float(candidates[i].get("vector_score") or 0.0), reverse=True)
        keyword_ranks = {idx: rank for rank, idx in enumerate(keyword_ranked)}
        vector_ranks = {idx: rank for rank, idx in enumerate(vector_ranked)}
        rrf_k = 60
        scored = []
        for i, row in enumerate(candidates):
            rrf_score = (1.0 / (rrf_k + keyword_ranks[i])) + (1.0 / (rrf_k + vector_ranks[i]))
            scored.append((rrf_score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_chunk(row) for _, row in scored[:normalized_limit]]

    @staticmethod
    def _vector_similarity(v1: Sequence[float], v2_bytes: bytes | None) -> float:
        if not v2_bytes or len(v1) == 0:
            return 0.0
        try:
            arr1 = np.asarray(v1, dtype=np.float32)
            arr2 = np.frombuffer(v2_bytes, dtype=np.float32)
            if arr1.shape != arr2.shape:
                return 0.0
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 < 1e-9 or norm2 < 1e-9:
                return 0.0
            return float(np.dot(arr1, arr2) / (norm1 * norm2))
        except Exception:
            return 0.0

    def _chunk_params(self, chunk: Chunk) -> Dict[str, Any]:
        start_row, start_col = chunk.start_rc
        end_row, end_col = chunk.end_rc
        return {
            "id": chunk.id(),
            "repo": chunk.repo,
            "branch": chunk.branch,
            "path": chunk.path,
            "language": chunk.language,
            "start_row": start_row,
            "start_col": start_col,
            "end_row": end_row,
            "end_col": end_col,
            "start_bytes": chunk.start_bytes,
            "end_bytes": chunk.end_bytes,
            "chunk": chunk.chunk,
            "status": "committed",
            "mutation_id": None,
            "search_vector": chunk.chunk,
            "embedding": chunk.embeddings,
        }

    @property
    def _upsert_sql(self) -> str:
        return (
            f"INSERT INTO {self._table} (id, repo, branch, path, language, start_row, start_col, end_row, end_col, "
            "start_bytes, end_bytes, chunk, status, mutation_id, search_vector, embedding) "
            "VALUES (:id, :repo, :branch, :path, :language, :start_row, :start_col, :end_row, :end_col, :start_bytes, :end_bytes, :chunk, :status, :mutation_id, :search_vector, :embedding) "
            "ON CONFLICT(id) DO UPDATE SET repo=excluded.repo, branch=excluded.branch, path=excluded.path, language=excluded.language, "
            "start_row=excluded.start_row, start_col=excluded.start_col, end_row=excluded.end_row, end_col=excluded.end_col, "
            "start_bytes=excluded.start_bytes, end_bytes=excluded.end_bytes, chunk=excluded.chunk, status=excluded.status, "
            "mutation_id=excluded.mutation_id, search_vector=excluded.search_vector, embedding=excluded.embedding"
        )


    def _bootstrap(self) -> None:
        ddl = [
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
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
              -- Kept for logical schema parity with postgres.
              search_vector TEXT,
              embedding BLOB NOT NULL
            )
            """,
            f"CREATE INDEX IF NOT EXISTS {self._table}_repo_idx ON {self._table}(repo)",
            f"CREATE INDEX IF NOT EXISTS {self._table}_repo_branch_idx ON {self._table}(repo, branch)",
            f"CREATE INDEX IF NOT EXISTS {self._table}_path_idx ON {self._table}(path)",
            f"CREATE INDEX IF NOT EXISTS {self._table}_repo_path_idx ON {self._table}(repo, path)",
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts_table} USING fts5(id, chunk)",
        ]
        with self._engine.begin() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))
            self._ensure_column(conn, "search_vector", "TEXT")

    def _ensure_column(self, conn, column_name: str, column_type: str) -> None:
        rows = conn.execute(text(f"PRAGMA table_info({self._table})")).mappings().all()
        existing = {row.get("name") for row in rows}
        if column_name in existing:
            return
        conn.execute(text(f"ALTER TABLE {self._table} ADD COLUMN {column_name} {column_type}"))

    def _build_engine(self) -> Engine:
        if create_engine is None:
            raise RuntimeError("SQLAlchemy is required for libsql persistence")
        url = self._cfg.url.rstrip("?")
        connect_args: Dict[str, Any] = {}
        if self._cfg.auth_token:
            connect_args["auth_token"] = self._cfg.auth_token
        return create_engine(
            f"sqlite+{url}?secure=true",
            future=True,
            connect_args=connect_args,
        )

    def _keyword_search(self, *, query: str, limit: int, repo: str | None = None, branch: str | None = None) -> List[
        Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}

        # Build filter clauses once, reuse for both FTS and fallback
        filter_clauses = []
        if repo is not None:
            filter_clauses.append("c.repo = :repo")
            params["repo"] = repo
        if branch is not None:
            filter_clauses.append("c.branch = :branch")
            params["branch"] = branch

        extra = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

        safe_query = self._safe_fts_query(query)

        fts_sql = text(
            f"SELECT c.id, c.repo, c.branch, c.path, c.language, c.start_row, c.start_col, "
            f"c.end_row, c.end_col, c.start_bytes, c.end_bytes, c.chunk, c.embedding, "
            f"-bm25({self._fts_table}) AS keyword_score, "
            f"0.0 AS vector_score "
            f"FROM {self._fts_table} AS f "
            f"JOIN {self._table} AS c ON c.id = f.id "
            f"WHERE f.chunk MATCH :query{extra} "
            f"ORDER BY bm25({self._fts_table}) ASC LIMIT :limit"
        )

        fallback_clauses = []
        if repo is not None:
            fallback_clauses.append("repo = :repo")
        if branch is not None:
            fallback_clauses.append("branch = :branch")
        fallback_extra = (" AND " + " AND ".join(fallback_clauses)) if fallback_clauses else ""

        fallback_sql = text(
            f"SELECT id, repo, branch, path, language, start_row, start_col, end_row, end_col, "
            f"start_bytes, end_bytes, chunk, embedding, "
            f"CASE WHEN lower(chunk) LIKE lower(:term) THEN 1 ELSE 0 END AS keyword_score, "
            f"0.0 AS vector_score "
            f"FROM {self._table} "
            f"WHERE lower(chunk) LIKE lower(:term){fallback_extra} "
            f"ORDER BY keyword_score DESC LIMIT :limit"
        )

        with self._engine.connect() as conn:
            if safe_query:
                try:
                    rows = conn.execute(fts_sql, {**params, "query": safe_query}).mappings().all()
                    if rows:
                        return [dict(row) for row in rows]
                except Exception:
                    logger.debug("FTS unavailable, falling back to LIKE", exc_info=True)

            term = f"%{query}%" if query else "%"
            rows = conn.execute(fallback_sql, {**params, "term": term}).mappings().all()
        return [dict(row) for row in rows]

    @staticmethod
    def _safe_fts_query(query: str) -> str:
        terms = [part.strip() for part in re.findall(r"[A-Za-z0-9_]+", query or "") if part.strip()]
        if not terms:
            return ""
        return " OR ".join(f'"{term}"' for term in terms)

    def _vector_search(
        self,
        *,
        query_embedding: np.ndarray,
        limit: int,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        filters = []
        if repo is not None:
            filters.append("repo = :repo")
            params["repo"] = repo
        if branch is not None:
            filters.append("branch = :branch")
            params["branch"] = branch
        where_clause = f" WHERE {' AND '.join(filters)}" if filters else ""
        sql = text(
            f"SELECT id, repo, branch, path, language, start_row, start_col, end_row, end_col, "
            f"start_bytes, end_bytes, chunk, embedding FROM {self._table}{where_clause}"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            row_dict["keyword_score"] = 0.0
            row_dict["vector_score"] = self._vector_similarity(query_embedding, row_dict.get("embedding"))
            scored.append(row_dict)
        scored.sort(key=lambda item: float(item.get("vector_score") or 0.0), reverse=True)
        return scored[:limit]


    @staticmethod
    def _row_to_chunk(row: Dict[str, Any]) -> Chunk:
        chunk_text = row.get("chunk") or ""
        return Chunk(
            chunk=chunk_text,
            repo=row.get("repo") or "",
            branch=row.get("branch"),
            path=row.get("path") or "",
            language=row.get("language") or "unknown",
            start_rc=(int(row.get("start_row") or 0), int(row.get("start_col") or 0)),
            end_rc=(int(row.get("end_row") or 0), int(row.get("end_col") or 0)),
            start_bytes=int(row.get("start_bytes") or 0),
            end_bytes=int(row.get("end_bytes") or len(chunk_text.encode("utf-8"))),
            embeddings=row.get("embedding"),
        )


def _libsql_factory(
    *,
    cfg: DBConfig,
    dim: int = EMBEDDING_DIMENSIONS,
    engine: Optional[Engine] = None,
    engine_factory: Optional[Callable[[], Engine]] = None,
    **kwargs: Any,
) -> PersistenceAdapter:
    return PersistInLibsql(
        cfg=cfg,
        dim=dim,
        engine=engine,
        engine_factory=engine_factory,
        **kwargs,
    )


register_persistence_adapter("libsql", _libsql_factory)

# Import side-effect: registers the postgres adapter.
from . import PersistPostgres  # noqa: F401


def create_persistence_adapter(
    adapter: str,
    *,
    cfg: DBConfig,
    dim: int = EMBEDDING_DIMENSIONS,
    **kwargs: Any,
) -> PersistenceAdapter:
    key = (adapter or "libsql").strip().lower()
    factory = get_persistence_adapter(key)
    if factory is None:
        raise ValueError(f"Unsupported persistence adapter '{adapter}'")
    return factory(cfg=cfg, dim=dim, **kwargs)


__all__ = [
    "DBConfig",
    "LibsqlConfig",
    "PersistInLibsql",
    "PersistenceAdapter",
    "create_persistence_adapter",
]
