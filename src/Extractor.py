# sync_chunks.py
import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

from cloudflare import Cloudflare
from sentence_transformers import SentenceTransformer

# Your chunker must expose chunk_file(); Chunk dataclass shown in the prompt.
import chunker  # expects chunker.chunk_file(path) -> List[Chunk]

# ------------------------- Embeddings -----------------------------------------------------------

_model = None  # lazy global


def get_embedder() -> SentenceTransformer:
    """Lazy-load CodeRankEmbed (kept locally or via HF cache).

    Uses CODERANK_MODEL_DIR if set, otherwise loads "nomic-ai/CodeRankEmbed".
    Raises RuntimeError on failure (fail-fast per requirements).
    """
    global _model
    if _model is not None:
        return _model
    src = os.environ.get("CODERANK_MODEL_DIR", "nomic-ai/CodeRankEmbed")
    try:
        _model = SentenceTransformer(src, trust_remote_code=True, device="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load CodeRankEmbed from '{src}': {e}") from e
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encode a batch of code chunks with CodeRankEmbed."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list[str]")
    if not texts:
        return []
    model = get_embedder()
    vecs = model.encode(texts, batch_size=32)
    return vecs.tolist()


# ------------------------- Cloudflare clients & config -----------------------------------------

def cf_client() -> Cloudflare:
    """Construct Cloudflare SDK client from env token."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise SystemExit("CLOUDFLARE_API_TOKEN is not set")
    return Cloudflare(api_token=token)


def cf_ids() -> Tuple[str, str, str]:
    """Read required Cloudflare identifiers from env (validate presence)."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    index_name = os.environ.get("VECTORIZE_INDEX_NAME")
    d1_db_id = os.environ.get("D1_DATABASE_ID")
    missing = [k for k, v in {
        "CLOUDFLARE_ACCOUNT_ID": account_id,
        "VECTORIZE_INDEX_NAME": index_name,
        "D1_DATABASE_ID": d1_db_id,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    return account_id, index_name, d1_db_id


def ensure_tables(client: Cloudflare, account_id: str, d1_db_id: str) -> None:
    """Idempotently ensure the D1 table exists for chunks and a path index."""
    sql = """
          CREATE TABLE IF NOT EXISTS chunks
          (
              id
              TEXT
              PRIMARY
              KEY,
              repo
              TEXT
              NOT
              NULL,
              path
              TEXT
              NOT
              NULL,
              language
              TEXT
              NOT
              NULL,
              start_row
              INTEGER
              NOT
              NULL,
              start_col
              INTEGER
              NOT
              NULL,
              end_row
              INTEGER
              NOT
              NULL,
              end_col
              INTEGER
              NOT
              NULL,
              start_bytes
              INTEGER
              NOT
              NULL,
              end_bytes
              INTEGER
              NOT
              NULL,
              signature
              TEXT,
              chunk
              TEXT
              NOT
              NULL
          );
          CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path); \
          """
    client.d1.database.query(account_id=account_id, database_id=d1_db_id, sql=sql)


# ------------------------- Shaping data ---------------------------------------------------------


def make_vector_records(chunks: List[Any], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """Build Vectorize upsert records (exclude raw text; include rich metadata)."""
    records: List[Dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings):
        start_row, start_col = getattr(chunk, "start_rc", (0, 0))
        end_row, end_col = getattr(chunk, "end_rc", (0, 0))
        records.append({
            # Use authoritative stable ID from Chunk
            "id": chunk.id(),
            "values": embedding,
            "metadata": {
                "repo": getattr(chunk, "repo", ""),
                "path": getattr(chunk, "path", ""),
                "language": getattr(chunk, "language", ""),
                "start_row": start_row, "start_col": start_col,
                "end_row": end_row, "end_col": end_col,
                "start_bytes": getattr(chunk, "start_bytes", 0),
                "end_bytes": getattr(chunk, "end_bytes", 0),
                "signature": getattr(chunk, "signature", ""),
            },
        })


def ndjson_bytes(objs: Iterable[Dict[str, Any]]) -> bytes:
    """Serialize objects into NDJSON bytes suited for Vectorize insert/upsert."""
    lines: List[str] = []
    for o in objs:
        lines.append(json.dumps(o, separators=(",", ":"), ensure_ascii=False))
    lines.append("")  # newline at end
    return "\n".join(lines).encode("utf-8")


# ------------------------- D1 operations --------------------------------------------------------
def d1_upsert_chunk(client: Cloudflare, account_id: str, d1_database_id: str, chunk: Any) -> None:
    """Insert or replace one chunk row in Cloudflare D1.

    Uses the authoritative stable ID from `chunk.id()` and persists the raw text
    in the `chunk` column. Maps `start_rc`/`end_rc` tuples to the corresponding
    row/col columns. Raises any errors from the Cloudflare SDK unchanged.
    """
    start_row, start_col = chunk.start_rc
    end_row, end_col = chunk.end_rc

    sql = """
    INSERT OR REPLACE INTO chunks(
      id, repo, path, language, start_row, start_col, end_row, end_col,
      start_bytes, end_bytes, signature, chunk
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    params = [chunk.id(), chunk.repo, chunk.path, chunk.language, start_row, start_col, end_row, end_col,
              chunk.start_bytes, chunk.end_bytes, chunk.signature, chunk.chunk, ]
    client.d1.database.query(
        account_id=account_id,
        database_id=d1_database_id,
        sql=sql,
        params=params,
    )


def d1_ids_for_path(client: Cloudflare, account_id: str, d1_db_id: str, path: str) -> List[str]:
    """Return vector IDs for a given file path from D1.

    This avoids O(N) scans in Vectorize during deletes and relies on the
    authoritative chunks table (indexed by path).
    """
    sql = "SELECT id FROM chunks WHERE path = ?;"
    page = client.d1.database.query(
        account_id=account_id,
        database_id=d1_db_id,
        sql=sql,
        params=[path],
    )
    rows = (getattr(page, "result", None) or [{}])[0].get("results", [])
    return [r["id"] for r in rows]


def d1_delete_ids(client: Cloudflare, account_id: str, d1_db_id: str, ids: List[str], batch: int = 100) -> None:
    """Delete rows from D1 by ID in batches (safe for SQLite param limits)."""
    if not ids:
        return
    for i in range(0, len(ids), batch):
        subset = ids[i:i + batch]
        placeholders = ",".join("?" for _ in subset)
        sql = f"DELETE FROM chunks WHERE id IN ({placeholders});"
        client.d1.database.query(account_id=account_id, database_id=d1_db_id, sql=sql, params=subset)


# ------------------------- Vectorize operations -------------------------------------------------

def vz_upsert_records(client: Cloudflare, account_id: str, index_name: str, records: List[Dict[str, Any]]) -> None:
    """Upsert vectors (id + values + metadata) via official Python SDK (NDJSON body)."""
    if not records:
        return
    body = ndjson_bytes(records)
    client.vectorize.indexes.upsert(index_name=index_name, account_id=account_id, body=body)


def vz_delete_ids(client: Cloudflare, account_id: str, index_name: str, ids: List[str], batch: int = 1024) -> None:
    """Delete vectors by IDs from Vectorize (batches for large deletions)."""
    if not ids:
        return
    for i in range(0, len(ids), batch):
        client.vectorize.indexes.delete_by_ids(
            index_name=index_name, account_id=account_id, ids=ids[i:i + batch]
        )


def vz_ids_for_path(client: Cloudflare, account_id: str, index_name: str, path: str,
                    page_size: int = 1000, hydrate_batch: int = 500) -> List[str]:
    """Return **all** vector IDs whose metadata.path equals `path`.

    Strategy (robust to `top_k` limits): page `list_vectors()` to get IDs, then
    fetch metadata for those IDs via `get_by_ids()` in batches and filter client-side.
    """
    matched: List[str] = []
    cursor = None
    while True:
        page = client.vectorize.indexes.list_vectors(
            index_name=index_name, account_id=account_id, count=page_size, cursor=cursor
        )
        vec_items = getattr(page, "vectors", []) or []
        ids = [v.get("id") if isinstance(v, dict) else getattr(v, "id", None) for v in vec_items]
        ids = [i for i in ids if i]
        for i in range(0, len(ids), hydrate_batch):
            resp = client.vectorize.indexes.get_by_ids(
                index_name=index_name, account_id=account_id, ids=ids[i:i + hydrate_batch]
            )
            # SDK returns an object with .result (list) OR attributes; handle both.
            result = getattr(resp, "result", resp) or []
            for item in result:
                meta = (item.get("metadata") if isinstance(item, dict)
                        else getattr(item, "metadata", {}))
                if isinstance(meta, dict) and meta.get("path") == path:
                    matched.append(item["id"] if isinstance(item, dict) else getattr(item, "id"))
        cursor = getattr(page, "next_cursor", None) or getattr(page, "nextCursor", None)
        more = bool(cursor) or getattr(page, "is_truncated", False) or getattr(page, "isTruncated", False)
        if not more:
            break
    return matched


# ------------------------- File flows -----------------------------------------------------------

def process_file(path: str, client: Cloudflare, account_id: str, index_name: str, d1_db_id: str) -> None:
    """Chunk → embed → upsert to Vectorize (metadata) → upsert to D1 (with text)."""
    chunks = chunker.chunk_file(path)
    if not chunks:
        return
    vectors = embed_texts([getattr(c, "chunk", "") for c in chunks])
    records = make_vector_records(chunks, vectors)
    vz_upsert_records(client, account_id, index_name, records)
    for c, rec in zip(chunks, records):
        d1_upsert_chunk(client, account_id, d1_db_id, c, rec["id"])


def delete_file(path: str, client: Cloudflare, account_id: str, index_name: str, d1_db_id: str) -> None:
    """Find IDs in Vectorize where metadata.path == `path` → delete in Vectorize → delete same IDs in D1."""
    ids = vz_ids_for_path(client, account_id, index_name, path)
    if not ids:
        return
    vz_delete_ids(client, account_id, index_name, ids)
    d1_delete_ids(client, account_id, d1_db_id, ids)


# ------------------------- CLI ------------------------------------------------------------------

def parse_changes_arg(arg: str) -> List[Dict[str, str]]:
    """Support inline JSON or @file.json for the --changes argument."""
    if not arg:
        raise SystemExit("--changes is required")
    if arg.startswith("@"):
        with open(arg[1:], "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(arg)


def main() -> None:
    """Drive sync from a JSON list of {file, action}, validating env + schema."""
    parser = argparse.ArgumentParser(description="Sync code chunks to Cloudflare Vectorize (embeddings) + D1 (text).")
    parser.add_argument("--changes", required=True, help="JSON list of {file, action} or @/path/to/file.json")
    args = parser.parse_args()

    client = cf_client()
    account_id, index_name, d1_db_id = cf_ids()
    ensure_tables(client, account_id, d1_db_id)

    changes = parse_changes_arg(args.changes)
    for item in changes:
        path = item.get("file")
        action = (item.get("action") or "").lower()
        if not path or action not in {"process", "delete"}:
            print(f"Skipping invalid change item: {item}", file=sys.stderr)
            continue
        if action == "process":
            process_file(path, client, account_id, index_name, d1_db_id)
        else:
            delete_file(path, client, account_id, index_name, d1_db_id)


if __name__ == "__main__":
    main()
