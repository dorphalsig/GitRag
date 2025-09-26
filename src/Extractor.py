from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Import chunker (support both package and module usage)
try:
    from .chunker import chunk_file, Chunk  # type: ignore
except Exception:  # pragma: no cover
    from src.chunker import chunk_file, Chunk  # type: ignore


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return environment variable or default.
    Keep this tiny helper to avoid sprinkling os.environ in many places.
    """
    return os.environ.get(name, default)




class CodeRankEmbedder:
    """Lazy loader for a local CodeRankEmbed model.

    The actual import happens on first use to minimize startup overhead for
    GitHub runners that only need deletion handling.
    """

    def __init__(self) -> None:
        self._model = None  # lazy

    def _ensure_model(self):
        """Load the embedding model only when needed."""
        if self._model is not None:
            return
        # Attempt to import a locally installed CodeRankEmbed model.
        # Adjust these imports to your specific local install if different.
        try:  # pragma: no cover - exercised only in runtime env with model installed
            import coderank_embed as cre  # hypothetical package name
            self._model = cre.load_default()  # type: ignore[attr-defined]
            return
        except Exception:
            pass
        try:  # Fallback to sentence-transformers if available
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            return
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "No embedding model found. Install CodeRankEmbed or sentence-transformers."
            ) from e

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: Iterable of text chunks to embed.
        Returns:
            List of embedding vectors (list of floats) aligned with input order.
        """
        self._ensure_model()
        # Simple adapter for the two supported backends
        if hasattr(self._model, "encode"):
            vecs = self._model.encode(list(texts))  # type: ignore[attr-defined]
        else:  # pragma: no cover - depends on a custom CodeRankEmbed API
            vecs = self._model.embed(list(texts))  # type: ignore[attr-defined]
        return [list(map(float, v)) for v in vecs]


class CloudflareVectorizeClient:
    """Client for Cloudflare Vectorize using the official Python SDK.

    Expects env vars: CF_ACCOUNT_ID, CF_API_TOKEN, CF_VECTORIZE_INDEX.
    All network calls go through the SDK (no raw urllib). Errors are wrapped
    into RuntimeError with concise messages.
    """

    def __init__(self,
                 account_id: Optional[str] = None,
                 api_token: Optional[str] = None,
                 index_name: Optional[str] = None) -> None:
        self.account_id = account_id or _env("CF_ACCOUNT_ID")
        self.api_token = api_token or _env("CF_API_TOKEN")
        self.index_name = index_name or _env("CF_VECTORIZE_INDEX")
        if not (self.account_id and self.api_token and self.index_name):  # pragma: no cover
            raise RuntimeError("Cloudflare Vectorize env vars missing: CF_ACCOUNT_ID, CF_API_TOKEN, CF_VECTORIZE_INDEX")

    def _sdk_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to a Cloudflare v4 path using the Python SDK.

        Tries the modern 'cloudflare' SDK first, then the legacy 'CloudFlare' SDK.
        Raises RuntimeError with a clear message on any error.
        """
        try:
            try:
                # New typed SDK
                from cloudflare import Cloudflare  # type: ignore
                cf = Cloudflare(api_token=self.api_token)
                client = getattr(cf, "_client", None) or getattr(cf, "client", None)
                if client and hasattr(client, "request"):
                    resp = client.request("POST", "/client/v4" + path, json=payload)
                    try:
                        return resp.json()
                    except Exception:
                        return {}
            except Exception:
                # Legacy SDK
                import CloudFlare  # type: ignore
                cf = CloudFlare.CloudFlare(token=self.api_token)
                if hasattr(cf, "api_call"):
                    return cf.api_call("POST", "/client/v4" + path, data=payload)
                if hasattr(cf, "raw"):
                    return cf.raw("POST", "/client/v4" + path, data=payload)
            raise RuntimeError("Cloudflare SDK found but request interface missing; please upgrade the 'cloudflare' package.")
        except Exception as e:
            raise RuntimeError(f"Cloudflare Vectorize API call failed ({path}): {e}") from e

    def upsert(self, items: List[Dict[str, Any]]) -> None:
        """Upsert vectors with metadata.

        Args:
            items: List of {"id": str, "values": List[float], "metadata": Dict[str,Any]}.
        """
        if not items:
            return
        payload = {"vectors": items}
        path = f"/accounts/{self.account_id}/vectorize/indexes/{self.index_name}/upsert"
        try:
            self._sdk_post(path, payload)
        except Exception as e:
            raise RuntimeError(f"Failed to upsert vectors to Cloudflare Vectorize: {e}") from e

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs. No-op if list is empty."""
        if not ids:
            return
        path = f"/accounts/{self.account_id}/vectorize/indexes/{self.index_name}/delete"
        try:
            self._sdk_post(path, {"ids": ids})
        except Exception as e:
            raise RuntimeError(f"Failed to delete vectors from Cloudflare Vectorize: {e}") from e

    def find_ids_by_filter(self, flt: Dict[str, Any]) -> List[str]:
        """Best-effort attempt to fetch IDs by metadata filter.

        If the SDK or endpoint doesn't support metadata-only search, returns [].
        Never raises; logs via return semantics to stay resilient.
        """
        try:
            path = f"/accounts/{self.account_id}/vectorize/indexes/{self.index_name}/metadata/search"
            data = self._sdk_post(path, {"filter": flt})
            vecs = (data.get("result") or {}).get("vectors") or []
            return [v.get("id") for v in vecs if isinstance(v, dict) and v.get("id")]
        except Exception:
            return []


class CloudflareD1Client:
    """Client for Cloudflare D1 using the Python SDK.

    Expects env vars: CF_ACCOUNT_ID, CF_API_TOKEN, CF_D1_DB. Uses parameterized SQL.
    All network calls go through the SDK; errors are wrapped with clear messages.
    """

    def __init__(self,
                 account_id: Optional[str] = None,
                 api_token: Optional[str] = None,
                 database_id: Optional[str] = None) -> None:
        self.account_id = account_id or _env("CF_ACCOUNT_ID")
        self.api_token = api_token or _env("CF_API_TOKEN")
        self.database_id = database_id or _env("CF_D1_DB")
        if not (self.account_id and self.api_token and self.database_id):  # pragma: no cover
            raise RuntimeError("Cloudflare D1 env vars missing: CF_ACCOUNT_ID, CF_API_TOKEN, CF_D1_DB")
        self._ensure_schema_done = False

    def _sdk_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to a Cloudflare v4 path using the Python SDK.

        Tries the modern 'cloudflare' SDK first, then the legacy 'CloudFlare' SDK.
        Returns parsed JSON-like dict when possible; raises RuntimeError otherwise.
        """
        try:
            try:
                from cloudflare import Cloudflare  # type: ignore
                cf = Cloudflare(api_token=self.api_token)
                client = getattr(cf, "_client", None) or getattr(cf, "client", None)
                if client and hasattr(client, "request"):
                    resp = client.request("POST", "/client/v4" + path, json=payload)
                    try:
                        return resp.json()
                    except Exception:
                        return {}
            except Exception:
                import CloudFlare  # type: ignore
                cf = CloudFlare.CloudFlare(token=self.api_token)
                if hasattr(cf, "api_call"):
                    return cf.api_call("POST", "/client/v4" + path, data=payload)
                if hasattr(cf, "raw"):
                    return cf.raw("POST", "/client/v4" + path, data=payload)
            raise RuntimeError("Cloudflare SDK found but request interface missing; please upgrade the 'cloudflare' package.")
        except Exception as e:
            raise RuntimeError(f"Cloudflare D1 API call failed ({path}): {e}") from e

    def _execute(self, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        path = f"/accounts/{self.account_id}/d1/database/{self.database_id}/query"
        body: Dict[str, Any] = {"sql": sql}
        if params:
            body["params"] = params
        data = self._sdk_post(path, body)
        # D1 returns envelope with 'result'; normalize
        return data.get("result", data)

    def ensure_schema(self) -> None:
        """Create the chunks table if it doesn't exist."""
        if self._ensure_schema_done:
            return
        sql = (
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id TEXT PRIMARY KEY,"
            "repo TEXT,"
            "path TEXT,"
            "language TEXT,"
            "start_rc_row INTEGER,"
            "start_rc_col INTEGER,"
            "end_rc_row INTEGER,"
            "end_rc_col INTEGER,"
            "start_bytes INTEGER,"
            "end_bytes INTEGER,"
            "signature TEXT,"
            "chunk TEXT"
            ")"
        )
        self._execute(sql)
        # Simple index for fast path deletes
        self._execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks(repo, path)")
        self._ensure_schema_done = True

    def upsert_chunks(self, rows: List[Tuple[str, Chunk]]) -> None:
        """Upsert chunk rows into D1.

        Each row is (id, chunk). We split composite fields and store full text.
        """
        if not rows:
            return
        self.ensure_schema()
        sql = (
            "INSERT OR REPLACE INTO chunks (id, repo, path, language, start_rc_row, start_rc_col, "
            "end_rc_row, end_rc_col, start_bytes, end_bytes, signature, chunk) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        for cid, ch in rows:
            sr, sc = ch.start_rc
            er, ec = ch.end_rc
            params = [
                cid, ch.repo, ch.path, ch.language, sr, sc, er, ec, ch.start_bytes, ch.end_bytes, ch.signature, ch.chunk
            ]
            self._execute(sql, params)

    def get_ids_by_path(self, repo: str, path: str) -> List[str]:
        """Return all chunk IDs for a repo+path from D1."""
        self.ensure_schema()
        sql = "SELECT id FROM chunks WHERE repo = ? AND path = ?"
        data = self._execute(sql, [repo, path])
        rows = data.get("results") or data.get("rows") or []
        # D1 returns different shapes; normalize
        ids: List[str] = []
        for row in rows:
            if isinstance(row, dict) and "id" in row:
                ids.append(row["id"])  # pragma: no cover
            elif isinstance(row, list) and row:
                ids.append(str(row[0]))
        return ids

    def delete_by_path(self, repo: str, path: str) -> None:
        """Delete all chunks for repo+path from D1."""
        self.ensure_schema()
        self._execute("DELETE FROM chunks WHERE repo = ? AND path = ?", [repo, path])


def _chunk_id(ch: Chunk) -> str:
    """Deterministic ID for a chunk using chunker.Chunk.id()."""
    try:
        return ch.id()
    except Exception:
        raw = f"{ch.repo}::{ch.path}::{ch.start_bytes}::{ch.end_bytes}"
        return hashlib.sha256(raw.encode()).hexdigest()


def _to_vectorize_item(ch: Chunk, vec: List[float], cid: str) -> Dict[str, Any]:
    """Map a chunk and its vector into a Vectorize payload item."""
    sr, sc = ch.start_rc
    er, ec = ch.end_rc
    meta = {
        "repo": ch.repo,
        "path": ch.path,
        "language": ch.language,
        "start_rc_row": sr,
        "start_rc_col": sc,
        "end_rc_row": er,
        "end_rc_col": ec,
        "start_bytes": ch.start_bytes,
        "end_bytes": ch.end_bytes,
        "signature": ch.signature,
    }
    return {"id": cid, "values": vec, "metadata": meta}


class ExtractorRunner:
    """Orchestrates chunking, embedding, and Cloudflare upserts/deletes."""

    def __init__(self, repo: str,
                 embedder: Optional[CodeRankEmbedder] = None,
                 vec_client: Optional[CloudflareVectorizeClient] = None,
                 d1_client: Optional[CloudflareD1Client] = None) -> None:
        self.repo = repo
        self.embedder = embedder or CodeRankEmbedder()
        self.vectorize = vec_client or CloudflareVectorizeClient()
        self.d1 = d1_client or CloudflareD1Client()

    def process_file(self, path: str) -> None:
        """Process a single file: chunk -> embed -> upsert to Vectorize and D1."""
        chunks = chunk_file(path, self.repo)
        if not chunks:
            return
        texts = [c.chunk for c in chunks]
        vecs = self.embedder.embed_texts(texts)
        rows = []
        items: List[Dict[str, Any]] = []
        for ch, v in zip(chunks, vecs):
            cid = _chunk_id(ch)
            items.append(_to_vectorize_item(ch, v, cid))
            rows.append((cid, ch))
        self.vectorize.upsert(items)
        self.d1.upsert_chunks(rows)

    def delete_file(self, path: str) -> None:
        """Delete stored vectors and chunks for a file from both backends."""
        flt = {"repo": self.repo, "path": path}
        ids = self.vectorize.find_ids_by_filter(flt)
        if not ids:
            ids = self.d1.get_ids_by_path(self.repo, path)
        self.vectorize.delete(ids)
        self.d1.delete_by_path(self.repo, path)

    def run(self, changes: Sequence[Dict[str, Any]]) -> None:
        """Run the extractor for a list of changes.

        Each change must have keys: {"file": str, "action": "process"|"delete"}.
        Errors for individual files are caught and reported; processing continues.
        """
        for change in changes:
            path = change.get("file") or change.get("path")
            action = (change.get("action") or "").lower()
            if not path or action not in {"process", "delete"}:
                continue
            try:
                if action == "process":
                    self.process_file(path)
                else:
                    self.delete_file(path)
            except Exception as e:
                print(f"ExtractorRunner error for {action} '{path}': {e}")


def run(changes: Union[str, Sequence[Dict[str, Any]]], repo: Optional[str] = None) -> None:
    """Convenience entry point for GitHub runners.

    Args:
        changes: Either a JSON string of change objects or a parsed list.
        repo: Repository identifier (e.g., "owner/repo"). If None, uses GITHUB_REPOSITORY.
    """
    if isinstance(changes, str):
        changes_list = json.loads(changes or "[]")
    else:
        changes_list = list(changes)
    repo_name = repo or _env("REPO") or _env("GITHUB_REPOSITORY") or ""
    ExtractorRunner(repo_name).run(changes_list)


if __name__ == "__main__":  # pragma: no cover
    import sys
    input_json = None
    if len(sys.argv) > 1:
        input_json = sys.argv[1]
    else:
        input_json = _env("CHANGES_JSON", "[]")
    repo_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(input_json, repo_arg)
