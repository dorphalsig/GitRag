"""PostgreSQL-backed persistence adapter with pgvector + native FTS."""
from __future__ import annotations

from array import array
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from Chunker.Chunk import Chunk
from .Persist import DBConfig, PersistenceAdapter
from constants import (
    DEFAULT_TABLE_NAME,
    EMBEDDING_DIMENSIONS,
    HYBRID_SEARCH_KEYWORD_WEIGHT,
    HYBRID_SEARCH_VECTOR_WEIGHT,
    POSTGRES_FTS_LANGUAGE,
)
from .persistence_registry import register_persistence_adapter


class PersistInPostgres(PersistenceAdapter):
    """Persistence adapter backed by PostgreSQL using pgvector and tsvector FTS."""

    def __init__(
        self,
        *,
        cfg: DBConfig,
        dim: int = EMBEDDING_DIMENSIONS,
        engine: Optional[Engine] = None,
        engine_factory: Optional[Callable[[], Engine]] = None,
        **_: Any,
    ) -> None:
        if cfg.provider != "postgres":
            raise TypeError("cfg.provider must be 'postgres' for PersistInPostgres")
        self._cfg = cfg
        self._dim = dim
        self._table = cfg.table_map.get("chunks", DEFAULT_TABLE_NAME)
        self._engine = engine or (engine_factory or self._build_engine)()
        self._bootstrap()

    def _build_engine(self) -> Engine:
        connect_args: Dict[str, Any] = {}
        if self._cfg.auth_token:
            connect_args["password"] = self._cfg.auth_token
        return create_engine(self._normalized_url(), future=True, connect_args=connect_args)

    def _normalized_url(self) -> str:
        url = self._cfg.url.strip()
        if url.startswith("postgresql+"):
            return url
        if url.startswith("postgres://"):
            return "postgresql+psycopg2://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+psycopg2://" + url[len("postgresql://") :]
        return url

    def _bootstrap(self) -> None:
        ddl = [
            "CREATE EXTENSION IF NOT EXISTS vector",
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
              embedding vector({self._dim}) NOT NULL,
              search_vector tsvector
            )
            """,
            f"CREATE INDEX IF NOT EXISTS {self._table}_search_vector_idx ON {self._table} USING GIN (search_vector)",
        ]
        with self._engine.begin() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))

    @property
    def _upsert_sql(self) -> str:
        return (
            f"INSERT INTO {self._table} (id, repo, branch, path, language, start_row, start_col, end_row, end_col, "
            "start_bytes, end_bytes, chunk, status, mutation_id, embedding, search_vector) "
            f"VALUES (:id, :repo, :branch, :path, :language, :start_row, :start_col, :end_row, :end_col, :start_bytes, :end_bytes, :chunk, :status, :mutation_id, :embedding, to_tsvector('{POSTGRES_FTS_LANGUAGE}', :chunk)) "
            "ON CONFLICT(id) DO UPDATE SET repo=excluded.repo, branch=excluded.branch, path=excluded.path, language=excluded.language, "
            "start_row=excluded.start_row, start_col=excluded.start_col, end_row=excluded.end_row, end_col=excluded.end_col, "
            "start_bytes=excluded.start_bytes, end_bytes=excluded.end_bytes, chunk=excluded.chunk, status=excluded.status, "
            "mutation_id=excluded.mutation_id, embedding=excluded.embedding, search_vector=excluded.search_vector"
        )

    def persist_batch(self, chunks: Sequence[Chunk]) -> None:
        valid = [chunk for chunk in chunks if chunk is not None]
        if not valid:
            return

        stmt = text(self._upsert_sql).bindparams(bindparam("embedding", type_=Vector(self._dim)))
        with self._engine.begin() as conn:
            for chunk in valid:
                if chunk.embeddings is None:
                    raise ValueError(f"Chunk {chunk.path} missing embeddings")
                start_row, start_col = chunk.start_rc
                end_row, end_col = chunk.end_rc
                conn.execute(
                    stmt,
                    {
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
                        "embedding": self._decode_embedding(chunk.embeddings),
                    },
                )

    def delete_batch(self, paths: List[str]) -> None:
        if not paths:
            return
        delete_stmt = text(
            f"DELETE FROM {self._table} WHERE path IN :paths"
        ).bindparams(bindparam("paths", expanding=True))
        with self._engine.begin() as conn:
            conn.execute(delete_stmt, {"paths": tuple(paths)})

    def search(
        self,
        query_embedding: Any,
        query_text: str,
        limit: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]:
        normalized_limit = max(1, int(limit))
        filters = [f":query_text = '' OR search_vector @@ websearch_to_tsquery('{POSTGRES_FTS_LANGUAGE}', :query_text)"]
        params: Dict[str, Any] = {
            "query_text": (query_text or "").strip(),
            "query_embedding": query_embedding,
            "limit": normalized_limit,
        }
        if repo is not None:
            filters.append("repo = :repo")
            params["repo"] = repo
        if branch is not None:
            filters.append("branch = :branch")
            params["branch"] = branch

        where_clause = " AND ".join(f"({flt})" for flt in filters)
        sql = text(
            f"""
            SELECT
                id,
                path,
                repo,
                branch,
                chunk,
                embedding,
                ts_rank_cd(search_vector, websearch_to_tsquery(:fts_lang, :query_text)) AS keyword_rank,
                1 - (embedding <=> :query_embedding) AS vector_rank,
                ((1 - (embedding <=> :query_embedding)) * :vector_weight +
                 ts_rank_cd(search_vector, websearch_to_tsquery(:fts_lang, :query_text)) * :keyword_weight) AS hybrid_rank
            FROM {self._table}
            WHERE {where_clause}
            ORDER BY hybrid_rank DESC
            LIMIT :limit
            """
        ).bindparams(bindparam("query_embedding", type_=Vector(self._dim)))
        params.update({
            "fts_lang": POSTGRES_FTS_LANGUAGE,
            "vector_weight": HYBRID_SEARCH_VECTOR_WEIGHT,
            "keyword_weight": HYBRID_SEARCH_KEYWORD_WEIGHT,
        })

        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._row_to_chunk(dict(row)) for row in rows]

    def get_indexed_paths(self) -> set[str]:
        sql = text(f"SELECT DISTINCT path FROM {self._table}")
        with self._engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0] for row in rows}

    def _decode_embedding(self, raw: bytes) -> np.ndarray:
        if len(raw) != self._dim * 4:
            raise ValueError(f"Embedding size mismatch: expected {self._dim * 4} bytes, got {len(raw)}")
        return np.frombuffer(raw, dtype=np.float32)

    def _row_to_chunk(self, row: Dict[str, Any]) -> Chunk:
        text_value = row.get("chunk") or ""
        embedding = row.get("embedding")
        embedding_bytes = None
        if embedding is not None:
            vals = array("f", embedding)
            embedding_bytes = vals.tobytes()
        return Chunk(
            chunk=text_value,
            repo=row.get("repo") or "",
            branch=row.get("branch"),
            path=row.get("path") or "",
            language=row.get("language") or "unknown",
            start_rc=(int(row.get("start_row") or 0), int(row.get("start_col") or 0)),
            end_rc=(int(row.get("end_row") or 0), int(row.get("end_col") or 0)),
            start_bytes=int(row.get("start_bytes") or 0),
            end_bytes=int(row.get("end_bytes") or len(text_value.encode("utf-8"))),
            embeddings=embedding_bytes,
        )


def _postgres_factory(
    *,
    cfg: DBConfig,
    dim: int,
    engine: Optional[Engine] = None,
    engine_factory: Optional[Callable[[], Engine]] = None,
    **kwargs: Any,
) -> PersistenceAdapter:
    return PersistInPostgres(
        cfg=cfg,
        dim=dim,
        engine=engine,
        engine_factory=engine_factory,
        **kwargs,
    )


register_persistence_adapter("postgres", _postgres_factory)

# Backward compatibility for legacy absolute import paths used by older tests.
sys.modules.setdefault("PersistPostgres", sys.modules[__name__])


__all__ = ["PersistInPostgres"]
