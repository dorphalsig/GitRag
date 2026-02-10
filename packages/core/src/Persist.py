"""libSQL-backed persistence adapter implemented with SQLAlchemy."""
from __future__ import annotations

import logging
import math
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

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

from Chunk import Chunk
from persistence_registry import get_persistence_adapter, register_persistence_adapter

logger = logging.getLogger(__name__)


class PersistenceAdapter(Protocol):
    def persist_batch(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_batch(self, paths: List[str]) -> None: ...

    def search(self, query_embedding: List[float], query_text: str, limit: int = 10) -> List[Chunk]: ...


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
        table: str = "chunks",
        fts_table: Optional[str] = None,
    ) -> "LibsqlConfig":
        resolved_fts = fts_table or f"{table}_fts"
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
        return self.table_map.get("chunks", "chunks")

    @property
    def fts_table(self) -> Optional[str]:
        return self.table_map.get("chunks_fts")

    @property
    def resolved_fts_table(self) -> str:
        return self.fts_table or f"{self.table}_fts"


class PersistInLibsql(PersistenceAdapter):
    """Persistence adapter backed by libSQL using SQLAlchemy."""

    def __init__(
        self,
        *,
        cfg: DBConfig,
        dim: int,
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
        self._table = cfg.table_map.get("chunks", "chunks")
        self._fts_table = cfg.table_map.get("chunks_fts", f"{self._table}_fts")
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
        select_stmt = (
            text(f"SELECT id FROM {self._table} WHERE path IN :paths")
            .bindparams(bindparam("paths", expanding=True))
        )
        with self._engine.connect() as conn:
            rows = conn.execute(select_stmt, {"paths": tuple(paths)}).mappings().all()
        ids = [row.get("id") for row in rows if row.get("id")]
        if not ids:
            return

        delete_stmt = text(
            f"DELETE FROM {self._table} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        delete_fts_stmt = text(
            f"DELETE FROM {self._fts_table} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))

        with self._engine.begin() as conn:
            conn.execute(delete_fts_stmt, {"ids": tuple(ids)})
            conn.execute(delete_stmt, {"ids": tuple(ids)})

    def close(self) -> None:
        if getattr(self, "_owns_engine", True):
            dispose = getattr(self._engine, "dispose", None)
            if callable(dispose):  # pragma: no branch - defensive
                dispose()

    def search(self, query_embedding: List[float], query_text: str, limit: int = 10) -> List[Chunk]:
        normalized_limit = max(1, int(limit))
        query = (query_text or "").strip()
        keyword_rows = self._keyword_search(query=query, limit=normalized_limit * 5)
        if not keyword_rows:
            return []

        scored = []
        for row in keyword_rows:
            keyword_score = int(row.get("keyword_score") or 0)
            vector_score = self._vector_similarity(query_embedding, row.get("embedding"))
            hybrid_score = (vector_score * 0.7) + (keyword_score * 0.3)
            scored.append((hybrid_score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_chunk(row) for _, row in scored[:normalized_limit]]

    def _chunk_params(self, chunk: Chunk) -> Dict[str, Any]:
        start_row, start_col = chunk.start_rc
        end_row, end_col = chunk.end_rc
        return {
            "id": chunk.id(),
            "repo": chunk.repo,
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
            f"INSERT INTO {self._table} (id, repo, path, language, start_row, start_col, end_row, end_col, "
            "start_bytes, end_bytes, chunk, status, mutation_id, embedding) "
            "VALUES (:id, :repo, :path, :language, :start_row, :start_col, :end_row, :end_col, :start_bytes, :end_bytes, :chunk, :status, :mutation_id, :embedding) "
            "ON CONFLICT(id) DO UPDATE SET repo=excluded.repo, path=excluded.path, language=excluded.language, "
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

    def _keyword_search(self, *, query: str, limit: int) -> List[Dict[str, Any]]:
        select_sql = (
            f"SELECT c.id, c.repo, c.path, c.language, c.start_row, c.start_col, c.end_row, c.end_col, "
            f"c.start_bytes, c.end_bytes, c.chunk, c.embedding, bm25(f) AS keyword_score "
            f"FROM {self._fts_table} AS f "
            f"JOIN {self._table} AS c ON c.id = f.id "
            f"WHERE f.chunk MATCH :query "
            f"ORDER BY keyword_score LIMIT :limit"
        )
        fallback_sql = (
            f"SELECT id, repo, path, language, start_row, start_col, end_row, end_col, "
            f"start_bytes, end_bytes, chunk, embedding, "
            f"CASE WHEN lower(chunk) LIKE lower(:term) THEN 1 ELSE 0 END AS keyword_score "
            f"FROM {self._table} "
            f"WHERE lower(chunk) LIKE lower(:term) "
            f"ORDER BY keyword_score DESC LIMIT :limit"
        )

        with self._engine.connect() as conn:
            if query:
                try:
                    rows = conn.execute(text(select_sql), {"query": query, "limit": limit}).mappings().all()
                    if rows:
                        return [dict(row) for row in rows]
                except Exception:
                    logger.debug("FTS query unavailable, falling back to LIKE", exc_info=True)

            term = f"%{query}%" if query else "%"
            rows = conn.execute(text(fallback_sql), {"term": term, "limit": limit}).mappings().all()
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
        dot = sum(a * b for a, b in zip(q, vals))
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
    dim: int,
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
    dim: int,
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
