import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, List, Any, Dict

import numpy as np
from cloudflare import Cloudflare

from Chunk import Chunk
from persistence_registry import register_persistence_adapter, get_persistence_adapter

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _resolve_attr(obj: Any, *names: str) -> Optional[str]:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj and obj[name]:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value:
                return value
    data = _to_dict(obj)
    if data:
        for name in names:
            value = data.get(name)
            if value:
                return value
    return None


def extract_mutation_id(obj: Any) -> Optional[str]:
    return _resolve_attr(obj, "mutation_id", "mutationId") or _resolve_attr(getattr(obj, "result", None), "mutation_id", "mutationId")


def extract_processed_mutation(obj: Any) -> Optional[str]:
    return _resolve_attr(obj, "processed_up_to_mutation", "processedUpToMutation") or _resolve_attr(getattr(obj, "result", None), "processed_up_to_mutation", "processedUpToMutation")


@dataclass(frozen=True)
class PersistConfig:
    account_id: str
    vectorize_index: str
    d1_database_id: str
    d1_table: str = "chunks"
    vector_metric: str = "cosine"
    batch_size: int = 1000
    poll_interval_s: float = 1.0
    poll_timeout_s: float = 300.0  # 5 minutes default


class PersistenceAdapter(Protocol):
    def persist_batch(self, chunks: Sequence[Chunk]) -> None: ...
    def delete_batch(self, paths: List[str]) -> None: ...


class PersistInVectorize(PersistenceAdapter):
    """
    Handles persistence of code chunks to Cloudflare Vectorize (embeddings) and D1 (chunk data).

    Consistency: writes D1 rows as 'pending' -> upserts vectors -> waits for Vectorize
    mutation to be processed -> marks D1 rows 'committed'. Readers can then filter on
    status='committed' to avoid mid-update states.

    Implementation details:
      - D1 mutations continue to use the official Cloudflare SDK.
      - Vectorize operations speak directly to the REST API using raw HTTP requests so
        NDJSON payloads are preserved exactly as required by the service.
    """

    def __init__(
        self,
        cfg: PersistConfig,
        dim: int,
        client: Cloudflare = None,
    ) -> None:
        """Create a persistence helper.

        Parameters
        ----------
        client : Cloudflare | None
            If provided, this Cloudflare SDK client will be used. If None, a new
            client will be constructed using environment configuration.
        cfg : PersistConfig
            Cloudflare resource identifiers and behavior knobs.
        dim : int
            Embedding vector dimension (used when creating the Vectorize index).
        """
        if client is None:
            # Cloudflare() will read CLOUDFLARE_API_TOKEN from the environment automatically.
            # This avoids hard-coding or leaking tokens here.
            self._client = Cloudflare()
        else:
            self._client = client
        self._cfg = cfg
        self._dim = dim
        self._setup_done = False

    def _ensure_setup(self) -> None:
        """Idempotent setup: create Vectorize index + metadata indexes + D1 table/indexes."""
        if self._setup_done:
            return
        account_id = self._cfg.account_id
        index_name = self._cfg.vectorize_index

        try:
            self._client.vectorize.indexes.get(index_name, account_id=account_id)
            logger.info("Vectorize index '%s' already exists", index_name)
        except Exception:
            logger.info(
                "Creating Vectorize index '%s' (dim=%d, metric=%s)",
                index_name,
                self._dim,
                self._cfg.vector_metric,
            )
            self._client.vectorize.indexes.create(
                account_id=account_id,
                name=index_name,
                config={"dimensions": self._dim, "metric": self._cfg.vector_metric},
            )

        present: set[str] = set()
        try:
            meta_resp = self._client.vectorize.indexes.metadata_index.list(
                index_name,
                account_id=account_id,
            )
            items = _to_dict(meta_resp) or {}
            indexes = items.get("indexes") or items.get("result", {}).get("indexes", [])
            for item in indexes or []:
                name = _resolve_attr(item, "property_name", "propertyName")
                if name:
                    present.add(name)
        except Exception:
            logger.warning(
                "Metadata indexes endpoint unavailable; skipping creation"
            )
            present = set()
            metadata_supported = False
        else:
            metadata_supported = True

        for prop in ("repo", "path", "language"):
            if not metadata_supported:
                break
            if prop in present:
                continue
            logger.info("Creating metadata index for '%s'", prop)
            try:
                self._client.vectorize.indexes.metadata_index.create(
                    index_name,
                    account_id=account_id,
                    index_type="string",
                    property_name=prop,
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message:
                    logger.info("Metadata index for '%s' already exists", prop)
                    continue
                logger.warning(
                    "Metadata index creation failed for '%s' (%s); skipping", prop, exc
                )
                metadata_supported = False
                break

        # 3) Ensure D1 table exists (SQLite-compatible)
        # status: 'pending' or 'committed'; mutation_id helps trace vectorize barrier
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._cfg.d1_table} (
          id TEXT PRIMARY KEY,
          repo TEXT NOT NULL,
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
          mutation_id TEXT
        );
        """
        self._client.d1.database.query(
            account_id=account_id,
            database_id=self._cfg.d1_database_id,
            sql=create_sql,
        )
        # Secondary indexes to support common filters
        for idx_sql in (
            f"CREATE INDEX IF NOT EXISTS idx_{self._cfg.d1_table}_repo_path ON {self._cfg.d1_table}(repo, path);",
            f"CREATE INDEX IF NOT EXISTS idx_{self._cfg.d1_table}_language ON {self._cfg.d1_table}(language);",
            f"CREATE INDEX IF NOT EXISTS idx_{self._cfg.d1_table}_status ON {self._cfg.d1_table}(status);",
        ):
            self._client.d1.database.query(
                account_id=account_id,
                database_id=self._cfg.d1_database_id,
                sql=idx_sql,
            )

        fts_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self._cfg.d1_table}_fts
        USING fts5(id, chunk);
        """
        self._client.d1.database.query(
            account_id=account_id,
            database_id=self._cfg.d1_database_id,
            sql=fts_sql,
        )

        self._setup_done = True
        logger.info("Persistence setup complete")

    def _insert_pending_rows(self, chunks: Sequence[Chunk]) -> None:
        """Insert or replace rows in D1 with status='pending'."""
        if not chunks:
            return
        sql = f"""
        INSERT INTO {self._cfg.d1_table}
        (id, repo, path, language, start_row, start_col, end_row, end_col, start_bytes, end_bytes, chunk, status, mutation_id)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'pending', NULL)
        ON CONFLICT(id) DO UPDATE SET
          repo=excluded.repo,
          path=excluded.path,
          language=excluded.language,
          start_row=excluded.start_row,
          start_col=excluded.start_col,
          end_row=excluded.end_row,
          end_col=excluded.end_col,
          start_bytes=excluded.start_bytes,
          end_bytes=excluded.end_bytes,
          chunk=excluded.chunk,
          status='pending',
          mutation_id=NULL
        ;
        """
        params = []
        for c in chunks:
            params.append([
                c.id(), c.repo, c.path, c.language,
                c.start_rc[0], c.start_rc[1],
                c.end_rc[0], c.end_rc[1],
                c.start_bytes, c.end_bytes,
                c.chunk,
            ])
        # D1 supports a single statement; execute per row for clarity & size safety.
        for row in params:
            self._client.d1.database.query(
                account_id=self._cfg.account_id,
                database_id=self._cfg.d1_database_id,
                sql=sql,
                params=row,
            )

    def _vector_records(self, chunks: Sequence[Chunk]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for c in chunks:
            if not isinstance(c.embeddings, (bytes, bytearray, memoryview)):
                raise ValueError("Chunk.embeddings must be bytes before persistence")
            values = np.frombuffer(c.embeddings, dtype=np.float32).tolist()
            metadata = {
                "repo": c.repo,
                "path": c.path,
                "language": c.language,
            }
            if c.metadata:
                metadata.update(c.metadata)
            records.append(
                {
                    "id": c.id(),
                    "values": values,
                    "metadata": metadata,
                }
            )
        return records

    def _upsert_vectors(self, records: List[Dict[str, Any]]) -> str:
        """Send vector records to Vectorize and return the mutation id."""
        upsert_result = self._client.vectorize.indexes.upsert(
            self._cfg.vectorize_index,
            account_id=self._cfg.account_id,
            body={"vectors": records},
            unparsable_behavior="error",
        )
        mutation_id = extract_mutation_id(upsert_result)
        if not mutation_id:
            raise RuntimeError("Vectorize upsert did not return a mutation_id")
        return mutation_id

    def _wait_processed(self, mutation_id: str) -> None:
        """
        Block until Vectorize index processes the given mutation id
        (consistent read barrier).
        """
        deadline = time.time() + self._cfg.poll_timeout_s
        while True:
            info = self._client.vectorize.indexes.info(
                self._cfg.vectorize_index,
                account_id=self._cfg.account_id,
            )
            processed = extract_processed_mutation(info)
            if processed is not None and processed >= mutation_id:
                return
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timeout waiting for Vectorize to process mutation {mutation_id}"
                )
            time.sleep(self._cfg.poll_interval_s)

    def _mark_committed(self, chunks: Sequence[Chunk], mutation_id: str) -> None:
        """Flip D1 rows to committed (single responsibility: finalize consistency)."""
        sql = f"UPDATE {self._cfg.d1_table} SET status='committed', mutation_id=?1 WHERE id=?2;"
        for c in chunks:
            self._client.d1.database.query(
                account_id=self._cfg.account_id,
                database_id=self._cfg.d1_database_id,
                sql=sql,
                params=[mutation_id, c.id()],
            )

    def _upsert_fts_rows(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        delete_sql = f"DELETE FROM {self._cfg.d1_table}_fts WHERE id=?1;"
        insert_sql = f"INSERT INTO {self._cfg.d1_table}_fts (id, chunk) VALUES (?1, ?2);"
        for c in chunks:
            chunk_id = c.id()
            self._client.d1.database.query(
                account_id=self._cfg.account_id,
                database_id=self._cfg.d1_database_id,
                sql=delete_sql,
                params=[chunk_id],
            )
            self._client.d1.database.query(
                account_id=self._cfg.account_id,
                database_id=self._cfg.d1_database_id,
                sql=insert_sql,
                params=[chunk_id, c.chunk],
            )

    def persist_batch(self, chunks: Sequence[Chunk]) -> None:
        """
        Persist a batch of chunks with a read-consistent workflow:

          1) D1 rows -> status=pending
          2) Vectorize upsert (NDJSON)
          3) Wait on processed_up_to_mutation >= mutation_id
          4) D1 rows -> status=committed

        Raises on any error (no silent guesswork).
        """
        if not chunks:
            return
        self._ensure_setup()
        self._insert_pending_rows(chunks)
        vector_records = self._vector_records(chunks)
        logger.info("Upserting %d vectors to '%s'...", len(chunks), self._cfg.vectorize_index)
        mutation_id = self._upsert_vectors(vector_records)
        logger.info("Upsert mutation_id=%s; waiting for processing...", mutation_id)
        self._wait_processed(mutation_id)
        self._mark_committed(chunks, mutation_id)
        self._upsert_fts_rows(chunks)
        logger.info("Batch committed (n=%d)", len(chunks))



    def delete_batch(self, paths: List[str]) -> None:
        """Delete all chunks for given file paths from Vectorize and D1; no-op if nothing matches.

        Steps
        -----
        1) Read IDs from D1 for the provided paths.
        2) If any, delete vectors by IDs from Vectorize.
        3) Delete rows from D1 by IDs.
        """
        if not paths:
            return
        placeholders = ",".join(["?"] * len(paths))
        select_query = f"SELECT id FROM {self._cfg.d1_table} WHERE path IN ({placeholders})"
        response = self._client.d1.database.query(
            account_id=self._cfg.account_id,
            database_id=self._cfg.d1_database_id,
            sql=select_query,
            params=paths,
        )
        rows = getattr(response, "result", None) or []
        ids = [row["id"] for row in rows if isinstance(row, dict) and row.get("id")]
        if not ids:
            return
        # Delete vectors in Vectorize
        try:
            self._client.vectorize.indexes.delete_by_ids(
                self._cfg.vectorize_index,
                account_id=self._cfg.account_id,
                ids=ids,
            )
        except Exception as e:
            logger.warning("Vectorize delete_by_ids failed (will still delete from D1): %s", e)
        self._delete_fts_rows(ids)
        # Delete rows from D1
        placeholders_ids = ",".join(["?"] * len(ids))
        delete_query = f"DELETE FROM {self._cfg.d1_table} WHERE id IN ({placeholders_ids})"
        self._client.d1.database.query(
            account_id=self._cfg.account_id,
            database_id=self._cfg.d1_database_id,
            sql=delete_query,
            params=ids,
        )

    def _delete_fts_rows(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        delete_sql = f"DELETE FROM {self._cfg.d1_table}_fts WHERE id IN ({placeholders});"
        self._client.d1.database.query(
            account_id=self._cfg.account_id,
            database_id=self._cfg.d1_database_id,
            sql=delete_sql,
            params=list(ids),
        )


def _cloudflare_factory(*, cfg: PersistConfig, dim: int, client: Cloudflare | None = None, **_: Any) -> PersistenceAdapter:
    return PersistInVectorize(
        cfg=cfg,
        dim=dim,
        client=client,
    )


for alias in ("cloudflare", "cf", "cloudflare_vectorize"):
    register_persistence_adapter(alias, _cloudflare_factory)


def create_persistence_adapter(
    adapter: str,
    *,
    cfg: PersistConfig,
    dim: int,
    client: Cloudflare | None = None,
    **kwargs: Any,
) -> PersistenceAdapter:
    key = (adapter or "cloudflare").strip().lower()
    factory = get_persistence_adapter(key)
    if factory is None:
        raise ValueError(f"Unsupported persistence adapter '{adapter}'")
    return factory(cfg=cfg, dim=dim, client=client, **kwargs)


__all__ = [
    "PersistConfig",
    "PersistInVectorize",
    "PersistenceAdapter",
    "create_persistence_adapter",
]
