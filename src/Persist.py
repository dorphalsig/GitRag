import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, List, Any, Dict

import numpy as np
from cloudflare import Cloudflare
import requests

from Chunk import Chunk
from persistence_registry import (
    register_persistence_adapter,
    get_persistence_adapter,
)

logger = logging.getLogger(__name__)


def _extract_result(data: Dict[str, Any], key: Optional[str] = None) -> Any:
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if key is None:
        if isinstance(result, dict):
            return result
        return data
    if isinstance(result, dict) and key in result:
        return result[key]
    return data.get(key)


def extract_mutation_id(data: Dict[str, Any]) -> Optional[str]:
    candidate = _extract_result(data)
    if isinstance(candidate, dict):
        value = candidate.get("mutation_id") or candidate.get("mutationId")
        if isinstance(value, str):
            return value
    if isinstance(data, dict):
        value = data.get("mutation_id") or data.get("mutationId")
        if isinstance(value, str):
            return value
    return None


def extract_processed_mutation(data: Dict[str, Any]) -> Optional[str]:
    candidate = _extract_result(data)
    if isinstance(candidate, dict):
        value = candidate.get("processed_up_to_mutation") or candidate.get("processedUpToMutation")
        if isinstance(value, str):
            return value
    if isinstance(data, dict):
        value = data.get("processed_up_to_mutation") or data.get("processedUpToMutation")
        if isinstance(value, str):
            return value
    return None


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
        http_client: Optional[requests.Session] = None,
        api_token: Optional[str] = None,
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
        self._http: Optional[requests.Session] = http_client
        if api_token is None:
            token = os.getenv("CLOUDFLARE_API_TOKEN")
        else:
            token = api_token
        self._api_token = (token or "").strip()
        self._base_url = "https://api.cloudflare.com/client/v4"

    def _ensure_setup(self) -> None:
        """Idempotent setup: create Vectorize index + metadata indexes + D1 table/indexes."""
        if self._setup_done:
            return
        account_id = self._cfg.account_id
        index_name = self._cfg.vectorize_index

        index_path = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}"
        indexes_path = f"/accounts/{account_id}/vectorize/v2/indexes"
        metadata_path = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}/metadata-indexes"

        try:
            self._vectorize_get(index_path)
            logger.info("Vectorize index '%s' already exists", index_name)
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            if status != 404:
                raise
            dim = self._dim
            logger.info(
                "Creating Vectorize index '%s' (dim=%d, metric=%s)",
                index_name,
                dim,
                self._cfg.vector_metric,
            )
            payload = {"name": index_name, "config": {"dimensions": dim, "metric": self._cfg.vector_metric}}
            self._vectorize_post(indexes_path, json_data=payload)

        present: set[str] = set()
        try:
            meta_resp = self._vectorize_get(metadata_path)
            items = _extract_result(meta_resp, "indexes")
            for item in items or []:
                name = None
                if isinstance(item, dict):
                    name = item.get("property_name") or item.get("propertyName")
                if name:
                    present.add(name)
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            if status != 404:
                raise

        metadata_supported = True
        for prop in ("repo", "path", "language"):
            if not metadata_supported:
                break
            if prop in present:
                continue
            logger.info("Creating metadata index for '%s'", prop)
            try:
                self._vectorize_post(
                    metadata_path,
                    json_data={"index_type": "string", "property_name": prop},
                )
            except requests.exceptions.HTTPError as exc:
                response = getattr(exc, "response", None)
                status = response.status_code if response is not None else None
                if status in {409, 400}:
                    logger.info("Metadata index for '%s' already exists", prop)
                    continue
                if status == 404:
                    logger.warning(
                        "Metadata indexes endpoint unavailable (status=404); skipping creation"
                    )
                    metadata_supported = False
                    break
                raise

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
            metadata = {
                "repo": c.repo,
                "path": c.path,
                "language": c.language,
            }
            if c.metadata:
                metadata.update(c.metadata)
            line = {
                "id": c.id(),
                "values": values,
                "metadata": metadata,
            }
            buf.write(json.dumps(line, separators=(",", ":")))
            buf.write("\n")
        return buf.getvalue().encode("utf-8")

    def _ensure_http_client(self) -> requests.Session:
        if not self._api_token:
            raise RuntimeError("CLOUDFLARE_API_TOKEN is required for Vectorize operations")
        if self._http is None:
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {self._api_token}"})
            self._http = session
        return self._http

    def _vectorize_request(
        self,
        method: str,
        path: str,
        *,
        json_data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        client = self._ensure_http_client()
        hdrs = dict(headers or {})
        if json_data is not None:
            hdrs.setdefault("Content-Type", "application/json")
        url = f"{self._base_url}{path}"
        resp = client.request(method, url, json=json_data, data=content, headers=hdrs, timeout=30)
        resp.raise_for_status()
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    def _vectorize_get(self, path: str) -> Dict[str, Any]:
        return self._vectorize_request("GET", path)

    def _vectorize_post(
        self,
        path: str,
        *,
        json_data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return self._vectorize_request("POST", path, json_data=json_data, content=content, headers=headers)

    def _upsert_vectors(self, payload: bytes) -> str:
        """Send NDJSON payload to Vectorize and return the mutation id."""
        account_id = self._cfg.account_id
        index_name = self._cfg.vectorize_index
        path = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}/upsert"
        data = self._vectorize_post(
            path,
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        mutation_id = extract_mutation_id(data)
        if not mutation_id:
            raise RuntimeError("Vectorize upsert did not return a mutation_id")
        return mutation_id

    def _wait_processed(self, mutation_id: str) -> None:
        """
        Block until Vectorize index processes the given mutation id
        (consistent read barrier).
        """
        deadline = time.time() + self._cfg.poll_timeout_s
        account_id = self._cfg.account_id
        index_name = self._cfg.vectorize_index
        path = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}/info"
        while True:
            info = self._vectorize_get(path)
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
        mutation_id = self._upsert_vectors(ndjson_bytes)
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
            account_id = self._cfg.account_id
            index_name = self._cfg.vectorize_index
            path = f"/accounts/{account_id}/vectorize/v2/indexes/{index_name}/delete_by_ids"
            self._vectorize_post(path, json_data={"ids": ids})
        except requests.exceptions.RequestException as e:
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


def _cloudflare_factory(*, cfg: PersistConfig, dim: int, client: Cloudflare | None = None, http_client: Optional[requests.Session] = None, api_token: Optional[str] = None, **_: Any) -> PersistenceAdapter:
    return PersistInVectorize(
        cfg=cfg,
        dim=dim,
        client=client,
        http_client=http_client,
        api_token=api_token,
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
