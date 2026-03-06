# Database Provisioning

These snippets bootstrap GitRag persistence for both libSQL and PostgreSQL.

## Shared Logical Schema
Both providers use the same logical `chunks` fields:
- `id, repo, branch, path, language`
- `start_row, start_col, end_row, end_col`
- `start_bytes, end_bytes`
- `chunk, status, mutation_id`
- `embedding` (provider-specific type)
- `search_vector` (native `tsvector` in postgres; nullable text parity field in libSQL)

## Embedding Dimension
The embedding size is defined by `EMBEDDING_DIMENSIONS` in
`packages/core/src/constants.py` (currently `768`).
If this constant changes, update `provisioning/postgres/schema.sql`.

## libSQL
1. Install `sqlx`, `litecli`, or use `turso` CLI to connect.
2. Run [`libsql/schema.sql`](libsql/schema.sql).
3. Verify with `PRAGMA table_info(chunks);` and
   `SELECT name FROM sqlite_master WHERE name='chunks_fts';`.

## PostgreSQL
1. Connect to the target database (the database itself must already exist).
2. Run [`postgres/schema.sql`](postgres/schema.sql).
3. Verify with `\d chunks` and `\di chunks_*`.

## Syncing
If you point the indexer at a local SQLite file, pass the remote URL/token via
`sync_url` and `auth_token` (the SQLAlchemy adapter handles this automatically).

## MCP Server
`packages/mcp-server` does not define its own database tables.
It calls `Retriever`, which reads from the same `chunks` persistence schema above.
