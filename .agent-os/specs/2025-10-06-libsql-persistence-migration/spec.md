# Spec Requirements Document

> Spec: libsql-persistence-migration
> Created: 2025-10-06

## Overview

Ensure GitRag's persistence layer centers on libSQL so the indexer stores metadata, text, and embeddings inside one SQLite-compatible deployment while preserving hybrid retrieval capabilities.

## User Stories

### Unified libSQL deployment

As an AI platform engineer, I want the indexer to bootstrap libSQL tables, vector indexes, and FTS structures automatically, so that I can provision GitRag without relying on external managed vector services.

The CLI should detect schema drift, apply required migrations, verify ANN and FTS indexes exist, and emit actionable errors when libSQL connectivity fails.

### Adapter-level observability

As a developer productivity lead, I want persistence metrics and logging to indicate which libSQL operations run (upserts, deletes, ANN queries), so that I can audit indexing runs and resolve ingestion failures quickly.

CLI runs should log libSQL connection targets, mutation batches, and row counts while exposing retry-safe errors for operational tooling.

### Hybrid retrieval parity

As a retrieval service maintainer, I want the new libSQL schema to support both vector ANN and FTS queries with consistent chunk identifiers, so that downstream services can continue to blend lexical and semantic scores without reindexing data.

The indexer must guarantee that every chunk write updates both the vector index and the FTS table inside a single transaction scope.

## Spec Scope

1. **LibSQL persistence adapter** - Implement a production-ready adapter that handles connections, transactions, and serialization of 768-dimension embeddings as float32 blobs.
2. **Schema provisioning and migrations** - Deliver libSQL DDL covering `chunks`, `chunks_fts`, DiskANN index creation, and supporting metadata indexes (`repo`, `path`, and `repo+path`) with automated bootstrapping from the CLI.
3. **Configuration & documentation refresh** - Provide configuration, environment variables, and docs aligned with libSQL usage, including deployment and troubleshooting guidance.

## Out of Scope

- Changing the embedding model or chunking pipeline.
- Building retrieval service query logic beyond documenting SQL patterns.

## Expected Deliverable

1. CLI runs persist chunk metadata, text, and embeddings into libSQL with passing integration tests for insert/update/delete flows.
2. Provisioning utilities create libSQL schema (tables, FTS, DiskANN index) and are validated by automated tests or smoke scripts.
3. Documentation and configuration defaults reflect libSQL usage, including environment variables, connection strings, and operational runbooks.
