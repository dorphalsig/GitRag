# Spec: libSQL Hybrid Retrieval Support

## Goal
Adopt libSQL as the persistence layer for chunk metadata, embeddings, and full-text postings so downstream retrieval (see `dorphalsig/GitRag-Retrieve`) can run hybrid (vector + lexical) queries without external managed-vector dependencies. The indexer must manage both vector indexes and FTS tables inside libSQL using SQLite-compatible syntax.

## Requirements
- Define schema DDL in `provisioning/libsql/` (mirrored for migrations) that creates:
  - `chunks` table with metadata columns plus `embedding F32_BLOB(768)` storing float32 vectors (dimension derived from `CodeRankCalculator().dimensions`).
  - `libsql_vector_idx` DiskANN cosine index on `chunks.embedding` for ANN search.
  - `chunks_fts` FTS5 virtual table mirroring chunk IDs + text for BM25 scoring.
- Ensure CLI provisioning step applies the libSQL DDL (table + ANN index + FTS).
- Write path: when persisting chunks, insert or replace corresponding rows in both `chunks` and `chunks_fts`, binding vectors as float32 blobs.
- Delete path: remove rows from `chunks` and `chunks_fts` when files are removed or chunks are retracted.
- Keep vector + FTS updates inside the existing pending→committed workflow so downstream readers never observe partial writes.
- Provide regression tests (unit/integration) that assert:
  - Vector blobs are stored/read back correctly (shape + dtype) from libSQL.
  - DiskANN index returns expected nearest neighbors for fixture embeddings.
  - FTS rows are created and deleted alongside their chunk counterparts.
  - Hybrid retrieval query examples (FTS match filtered by ANN results) surface known fixture chunks.
- Document libSQL connection requirements (URL/auth), vector binding format, and ANN/FTS query examples in `docs`.

## Non-Goals
- Implementing the retrieval/query service (handled separately in `GitRag-Retrieve`).
- Ranking logic changes beyond wiring ANN + FTS results; scoring remains existing blend logic.

## Risks
- Incorrect blob serialization may corrupt vectors; enforce float32 encoding helpers with tests.
- ANN index rebuild costs could spike on bulk updates; ensure CLI batches updates and avoids frequent full reindexing.

## Open Questions
- Do we keep chunk embeddings co-located with metadata, or split into a dedicated `embeddings` table for growth? (Default: single `chunks` table with vector column.)
- Should ANN queries run via SQL helper views or through libSQL SDK primitives? (Initial plan: issue raw SQL to keep portability.)
