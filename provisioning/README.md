# libSQL Provisioning

These snippets help bootstrap a libSQL (Turso) deployment for GitRag.

## Required Environment
- `TURSO_DATABASE_URL` — e.g. `libsql://<host>/<db>`
- `TURSO_AUTH_TOKEN` — JWT for the database.

## Applying Schema
1. Install `sqlx`, `litecli`, or use `turso` CLI to connect.
2. Run the statements in [`libsql/schema.sql`](libsql/schema.sql).
3. Optional: verify tables with `PRAGMA table_info(chunks);` and ensure the FTS
   table exists via `SELECT name FROM sqlite_master WHERE name='chunks_fts';`.

## Syncing
If you point the indexer at a local SQLite file, pass the remote URL/token via
`sync_url` and `auth_token` (the SQLAlchemy adapter handles this automatically).
