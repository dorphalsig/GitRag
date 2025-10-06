# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-10-06-libsql-persistence-migration/spec.md

## Technical Requirements

- Introduce a libSQL persistence adapter built on SQLAlchemy + `sqlalchemy-libsql`, using Turso URL + auth token environment variables to create an engine.
- Serialize embeddings as float32 arrays (768 dims) using numpy buffers, binding them to a BLOB column for both upsert and delete statements.
- Wrap write/delete flows in explicit transactions that update `chunks` and `chunks_fts` together while ensuring no partial writes escape.
- Ensure hybrid retrieval queries can apply `repo` / `path` filters efficiently by joining `chunks_fts` to `chunks` on `id`, leveraging dedicated B-tree indexes on `repo`, `path`, and `repo+path`.
- Provide schema bootstrap routines that apply DDL idempotently (tables, indexes, FTS) and detect drift via `pragma_table_info` / `pragma_index_list` checks.
- Extend integration tests to run against a reachable libSQL instance, verifying inserts, updates, deletes, and FTS query examples using fixture embeddings.
- Update CLI configuration to accept `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`, emitting structured logs for connection attempts and batch results.
- Refresh docs with step-by-step provisioning, connection troubleshooting, and SQL examples for hybrid retrieval queries.

## External Dependencies

- **SQLAlchemy (>=2.0)** - Core engine for connection pooling and transactional execution.
  - **Justification:** Provides battle-tested DBAPI integration and transaction management with minimal custom code.
- **sqlalchemy-libsql (>=0.2)** - SQLAlchemy dialect for remote libSQL deployments.
  - **Justification:** Adds the necessary driver hooks to connect to Turso/libSQL over HTTP(S).
