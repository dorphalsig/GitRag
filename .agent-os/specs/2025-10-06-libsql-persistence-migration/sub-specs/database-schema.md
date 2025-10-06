# Database Schema

This is the database schema implementation for the spec detailed in @.agent-os/specs/2025-10-06-libsql-persistence-migration/spec.md

## Changes

- Provide libSQL-native DDL and migration scripts for the indexer schema.
- Add `embedding F32_BLOB(768)` column to the `chunks` table and enforce NOT NULL constraint.
- Introduce ANN index creation using `libsql_vector_idx` (DiskANN + cosine) on `chunks.embedding`.
- Create `chunks_fts` virtual table (FTS5) mirroring chunk IDs and text, with triggers or transactional writes handled by the adapter.
- Deliver migration helpers to drop any deprecated artifacts from earlier deployments.

## Specifications

```sql
CREATE TABLE IF NOT EXISTS chunks (
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
  mutation_id TEXT,
  embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks (path);
CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks (repo, path);
CREATE INDEX IF NOT EXISTS chunks_status_idx ON chunks (status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  id UNINDEXED,
  chunk
);
```

- Migrations should remove any deprecated metadata tables from earlier deployments before applying new structures.
- Schema bootstrap script should encapsulate the above SQL in ordered steps with error handling and logging.

## Rationale

- Consolidating embeddings and metadata inside libSQL reduces operational overhead and eliminates the need for separate Vectorize APIs.
- DiskANN with cosine provides fast ANN queries tuned for normalized CodeRank embeddings.
- Maintaining `chunks_fts` enables BM25 keyword search that matches existing hybrid retrieval expectations; metadata filters remain in `chunks` with joins supported by the new B-tree indexes.
- NOT NULL and index constraints ensure adapter writes catch missing embeddings early and keep query performance predictable.
