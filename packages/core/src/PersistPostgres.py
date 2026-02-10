"""PostgreSQL-backed persistence adapter with pgvector + native FTS."""
from __future__ import annotations

from array import array
from typing import Any, Callable, Dict, List, Optional, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from Chunk import Chunk
from Persist import DBConfig, PersistenceAdapter
from persistence_registry import register_persistence_adapter


class PersistInPostgres(PersistenceAdapter):
    """Persistence adapter backed by PostgreSQL using pgvector and tsvector FTS."""

    def __init__(
        self,
        *,
        cfg: DBConfig,
        dim: int,
        engine: Optional[Engine] = None,
        engine_factory: Optional[Callable[[], Engine]] = None,
        **_: Any,
    ) -> None:
        if cfg.provider != "postgres":
            raise TypeError("cfg.provider must be 'postgres' for PersistInPostgres")
        self._cfg = cfg
        self._dim = dim
        self._table = cfg.table_map.get("chunks", "chunks")
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
              path TEXT NOT NULL,
              repo TEXT NOT NULL,
              branch TEXT,
              chunk TEXT NOT NULL,
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
            f"INSERT INTO {self._table} (id, path, repo, branch, chunk, embedding, search_vector) "
            "VALUES (:id, :path, :repo, :branch, :chunk, :embedding, to_tsvector('english', :chunk)) "
            "ON CONFLICT(id) DO UPDATE SET path=excluded.path, repo=excluded.repo, branch=excluded.branch, chunk=excluded.chunk, "
            "embedding=excluded.embedding, search_vector=excluded.search_vector"
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
                conn.execute(
                    stmt,
                    {
                        "id": chunk.id(),
                        "path": chunk.path,
                        "repo": chunk.repo,
                        "branch": chunk.branch,
                        "chunk": chunk.chunk,
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
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        repo: str | None = None,
        branch: str | None = None,
    ) -> List[Chunk]:
        normalized_limit = max(1, int(limit))
        filters = [":query_text = '' OR search_vector @@ websearch_to_tsquery('english', :query_text)"]
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
                ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) AS keyword_rank,
                1 - (embedding <=> :query_embedding) AS vector_rank,
                ((1 - (embedding <=> :query_embedding)) * 0.7 +
                 ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) * 0.3) AS hybrid_rank
            FROM {self._table}
            WHERE {where_clause}
            ORDER BY hybrid_rank DESC
            LIMIT :limit
            """
        ).bindparams(bindparam("query_embedding", type_=Vector(self._dim)))

        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._row_to_chunk(dict(row)) for row in rows]

    def _decode_embedding(self, raw: bytes) -> List[float]:
        if len(raw) != self._dim * 4:
            raise ValueError(f"Embedding size mismatch: expected {self._dim * 4} bytes, got {len(raw)}")
        vals = array("f")
        vals.frombytes(raw)
        return list(vals)

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
            language="unknown",
            start_rc=(0, 0),
            end_rc=(0, 0),
            start_bytes=0,
            end_bytes=len(text_value.encode("utf-8")),
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


__all__ = ["PersistInPostgres"]
