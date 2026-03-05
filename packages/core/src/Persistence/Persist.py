"""libSQL-backed persistence adapter implemented with SQLAlchemy."""
from __future__ import annotations

import logging
import math
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
from persistence_registry import get_persistence_adapter, register_persistence_adapter

logger = logging.getLogger(__name__)


class PersistenceAdapter(Protocol):
    def persist_batch(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_batch(self, paths: List[str]) -> None: ...

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]: ...


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
        with self._engine.begin() as conn:
            for chunk in valid:
                params = self._chunk_params(chunk)
                conn.execute(insert_stmt, params)
                self._refresh_fts(conn, params["id"], chunk.chunk)

    def delete_batch(self, paths: List[str]) -> None:
        if not paths:
            return
        delete_stmt = text(
            f"DELETE FROM {self._table} WHERE path IN :paths"
        ).bindparams(bindparam("paths", expanding=True))
        delete_fts_stmt = text(
            f"DELETE FROM {self._fts_table} WHERE id NOT IN (SELECT id FROM {self._table})"
        )

        with self._engine.begin() as conn:
            conn.execute(delete_stmt, {"paths": tuple(paths)})
            conn.execute(delete_fts_stmt)

    def close(self) -> None:
        if getattr(self, "_owns_engine", True):
            dispose = getattr(self._engine, "dispose", None)
            if callable(dispose):  # pragma: no branch - defensive
                dispose()

    def search(
        self,
        query_embedding: List[float],
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
        if not keyword_rows:
            return []

        scored = []
        for row in keyword_rows:
            keyword_score = float(row.get("keyword_score") or 0.0)
            vector_score = self._vector_similarity(query_embedding, row.get("embedding"))
            hybrid_score = (vector_score * HYBRID_SEARCH_VECTOR_WEIGHT) + (keyword_score * HYBRID_SEARCH_KEYWORD_WEIGHT)
            scored.append((hybrid_score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_chunk(row) for _, row in scored[:normalized_limit]]

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
            "embedding": chunk.embeddings,
        }

    @property
    def _upsert_sql(self) -> str:
        return (
            f"INSERT INTO {self._table} (id, repo, branch, path, language, start_row, start_col, end_row, end_col, "
            "start_bytes, end_bytes, chunk, status, mutation_id, embedding) "
            "VALUES (:id, :repo, :branch, :path, :language, :start_row, :start_col, :end_row, :end_col, :start_bytes, :end_bytes, :chunk, :status, :mutation_id, :embedding) "
            "ON CONFLICT(id) DO UPDATE SET repo=excluded.repo, branch=excluded.branch, path=excluded.path, language=excluded.language, "
            "start_row=excluded.start_row, start_col=excluded.start_col, end_row=excluded.end_row, end_col=excluded.end_col, "
            "start_bytes=excluded.start_bytes, end_bytes=excluded.end_bytes, chunk=excluded.chunk, status=excluded.status, "
            "mutation_id=excluded.mutation_id, embedding=excluded.embedding"
        )

    def _refresh_fts(self, conn, chunk_id: str, text_value: str) -> None:
        delete_stmt = text(f"DELETE FROM {self._fts_table} WHERE id = :id")
        insert_stmt = text(
            f"INSERT INTO {self._fts_table} (id, chunk) VALUES (:id, :chunk)"
        )
        conn.execute(delete_stmt, {"id": chunk_id})
        conn.execute(insert_stmt, {"id": chunk_id, "chunk": text_value})

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

        fts_sql = text(
            f"SELECT c.id, c.repo, c.branch, c.path, c.language, c.start_row, c.start_col, "
            f"c.end_row, c.end_col, c.start_bytes, c.end_bytes, c.chunk, c.embedding, "
            f"-bm25({self._fts_table}) AS keyword_score "
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
            f"CASE WHEN lower(chunk) LIKE lower(:term) THEN 1 ELSE 0 END AS keyword_score "
            f"FROM {self._table} "
            f"WHERE lower(chunk) LIKE lower(:term){fallback_extra} "
            f"ORDER BY keyword_score DESC LIMIT :limit"
        )

        with self._engine.connect() as conn:
            if query:
                try:
                    rows = conn.execute(fts_sql, {**params, "query": query}).mappings().all()
                    if rows:
                        return [dict(row) for row in rows]
                except Exception:
                    logger.debug("FTS unavailable, falling back to LIKE", exc_info=True)

            term = f"%{query}%" if query else "%"
            rows = conn.execute(fallback_sql, {**params, "term": term}).mappings().all()
        return [dict(row) for row in rows]

    @staticmethod
    def _vector_similarity(query_embedding: List[float], raw_embedding: Any) -> float:
        if not query_embedding or raw_embedding is None:
            return 0.0
        vals = array("f")
        vals.frombytes(bytes(raw_embedding))
        if not vals:
            return 0.0
        q = query_embedding[: len(vals)]
        if not q:
            return 0.0
        dot = float(np.dot(q, vals))
        q_norm = math.sqrt(sum(a * a for a in q))
        v_norm = math.sqrt(sum(b * b for b in vals[: len(q)]))
        if q_norm == 0 or v_norm == 0:
            return 0.0
        return dot / (q_norm * v_norm)

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
import PersistPostgres  # noqa: F401


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
