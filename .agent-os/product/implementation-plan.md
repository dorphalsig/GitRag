# Implementation Plan

## GitHub Action Wrapper

- Package the CLI as an action with composite `uses` syntax so any repository can reuse it.
- Inputs: repository identifier, optional changes payload, credentials (`CF_API_TOKEN`, `CF_ACCOUNT_ID`, `CF_VECTORIZE_INDEX`, `CF_D1_DATABASE_ID`).
- Steps: checkout, set up Python 3.11 with cached dependencies, install `requirements.txt`, run `python src/Indexer.py --repo ${{ inputs.repo }}`.
- Outputs and summaries: expose JSON summary emitted by the CLI, surface metrics via job outputs.
- Publishing: add action metadata (`action.yml`) and README, register in GitHub Marketplace once smoke tested.

## Initial Indexing Flag

- CLI option `--full` to emit `{path, action="process"}` for all tracked, non-ignored, text files.
- Use shared `BinaryDetector` plus `.gitignore` aware `git ls-files` to build candidate set.
- Guard rails: bail with clear error when repository exceeds configurable file count threshold, provide dry-run summary.
- Persistence: reuse existing `_process_files` pipeline with generated change list.

## Storage Adapter Pattern

- Define protocol `VectorPersistence` with `persist_batch` and `delete_batch` signatures mirroring Cloudflare implementation.
- Move Cloudflare logic into `PersistInCloudflare` adapter implementing the protocol.
- Provide factory that reads env (or CLI flag) to choose adapter, defaulting to Cloudflare.
- Document adapter expectations (chunk embedding bytes, mutation barriers) so contributors can implement alternatives (e.g., Pinecone + Postgres).

## D1 FTS Hybrid Retrieval Layer

- Add an FTS5 virtual table (e.g., `chunks_fts`) keyed on chunk `id` to store normalized text for BM25 scoring.
- Update persistence flows to insert/update/delete FTS rows in lockstep with the primary `chunks` table.
- Ensure migration scripts (or documented DDL in `provisioning/d1/`) include the virtual table and triggers if needed.
- Document the hybrid retrieval contract so `GitRag-Retrieve` can combine Vectorize scores with FTS BM25 results.
- Provide smoke tests (flagged optional) that index a small fixture, query the FTS table, and confirm keyword hits align with expectations.

## Requirement Traceability & Compliance Queries

- Extend the chunker to tag requirement documents and extract heading/topic metadata so requirement specs (e.g., pricing rules) remain searchable alongside code.
- Persist requirement-topic annotations in D1 metadata fields to enable topic-scoped retrieval without rescanning raw documents.
- Add CLI/query ergonomics to surface requirement snippets next to matching implementation chunks, letting reviewers confirm code changes satisfy the documented rules.
- Provide guidance/tests that validate requirement lookup flows (topic filter, code-to-requirement cross reference) to prevent regressions once shipped.
