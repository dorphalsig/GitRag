import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Sequence, List

import numpy as np
from cloudflare import Cloudflare

from Chunk import Chunk

logger = logging.getLogger(__name__)


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


class PersistInVectorize:
    """
    Handles persistence of code chunks to Cloudflare Vectorize (embeddings) and D1 (chunk data).

    Consistency: writes D1 rows as 'pending' -> upserts vectors -> waits for Vectorize
    mutation to be processed -> marks D1 rows 'committed'. Readers can then filter on
    status='committed' to avoid mid-update states.

    All Cloudflare SDK calls use documented Python endpoints:
      - D1: client.d1.database.query(account_id, database_id, sql, params)
      - Vectorize upsert: client.vectorize.indexes.upsert(index_name, account_id, body=ndjson_bytes, ...)
      - Vectorize info: client.vectorize.indexes.info(index_name, account_id)
      - Metadata indexes: create/list for repo, path, language
    """

    def __init__(self, cfg: PersistConfig, dim: int,  client: Cloudflare = None) -> None:
        """Create a persistence helper.

        Parameters
        ----------
        client : Cloudflare | None
            If provided, this Cloudflare SDK client will be used. If None, a new
            client will be constructed using the CF_API_TOKEN env var.
        cfg : PersistConfig
            Cloudflare resource identifiers and behavior knobs.
        dim : int
            Embedding vector dimension (used when creating the Vectorize index).
        """
        if client is None:
            # Cloudflare() will read CF_API_TOKEN from the environment automatically.
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

        # 1) Ensure Vectorize index exists
        try:
            self._client.vectorize.indexes.get(
                self._cfg.vectorize_index, account_id=account_id
            )
            logger.info("Vectorize index '%s' already exists", self._cfg.vectorize_index)
        except Exception:
            dim = self._dim
            logger.info("Creating Vectorize index '%s' (dim=%d, metric=%s)",
                        self._cfg.vectorize_index, dim, self._cfg.vector_metric)
            self._client.vectorize.indexes.create(
                account_id=account_id,
                name=self._cfg.vectorize_index,
                config={"dimensions": dim, "metric": self._cfg.vector_metric},
            )

        # 2) Ensure metadata indexes present (repo, path, language)
        present = set()
        try:
            lst = self._client.vectorize.indexes.metadata_index.list(
                self._cfg.vectorize_index, account_id=account_id
            )
            for item in getattr(lst, "indexes", []) or []:
                name = getattr(item, "property_name", None) or item.get("propertyName")  # be defensive
                if name:
                    present.add(name)
        except Exception as e:
            logger.warning("Could not list metadata indexes (will still attempt to create): %s", e)

        for prop in ("repo", "path", "language"):
            if prop in present:
                continue
            logger.info("Creating metadata index for '%s'", prop)
            self._client.vectorize.indexes.metadata_index.create(
                self._cfg.vectorize_index,
                account_id=account_id,
                index_type="string",
                property_name=prop,
            )

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

    def _ndjson_for(self, chunks: Sequence[Chunk]) -> bytes:
        """Build NDJSON bytes for Vectorize upsert: {id, values, metadata} per line."""
        buf = io.StringIO()
        for c in chunks:
            if not isinstance(c.embeddings, (bytes, bytearray, memoryview)):
                raise ValueError("Chunk.embeddings must be bytes before persistence")
            values = np.frombuffer(c.embeddings, dtype=np.float32).tolist()
            line = {
                "id": c.id(),
                "values": values,
                "metadata": {
                    "repo": c.repo,
                    "path": c.path,
                    "language": c.language,
                },
            }
            buf.write(json.dumps(line, separators=(",", ":")))
            buf.write("\n")
        return buf.getvalue().encode("utf-8")

    def _wait_processed(self, mutation_id: str) -> None:
        """
        Block until Vectorize index processes the given mutation id
        (consistent read barrier).
        """
        deadline = time.time() + self._cfg.poll_timeout_s
        while True:
            info = self._client.vectorize.indexes.info(
                self._cfg.vectorize_index, account_id=self._cfg.account_id
            )
            # Python SDK uses snake_case fields; be defensive with camelCase too.
            processed = getattr(info, "processed_up_to_mutation", None)
            if processed is None:
                processed = getattr(info, "processedUpToMutation", None)
            if isinstance(processed, str) and processed >= mutation_id:
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
        ndjson_bytes = self._ndjson_for(chunks)
        logger.info("Upserting %d vectors to '%s'...", len(chunks), self._cfg.vectorize_index)
        up = self._client.vectorize.indexes.upsert(
            self._cfg.vectorize_index,
            account_id=self._cfg.account_id,
            body=ndjson_bytes,
            unparsable_behavior="error",
        )
        mutation_id = getattr(up, "mutation_id", None) or getattr(up, "mutationId", None)
        if not mutation_id:
            raise RuntimeError("Vectorize upsert did not return a mutation_id")
        logger.info("Upsert mutation_id=%s; waiting for processing...", mutation_id)
        self._wait_processed(mutation_id)
        self._mark_committed(chunks, mutation_id)
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
            # Log but proceed to attempt D1 deletion to keep sources authoritative.
            logger.warning("Vectorize delete_by_ids failed (will still delete from D1): %s", e)
        # Delete rows from D1
        placeholders_ids = ",".join(["?"] * len(ids))
        delete_query = f"DELETE FROM {self._cfg.d1_table} WHERE id IN ({placeholders_ids})"
        self._client.d1.database.query(
            account_id=self._cfg.account_id,
            database_id=self._cfg.d1_database_id,
            sql=delete_query,
            params=ids,
        )
