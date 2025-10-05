# Spec: D1 FTS Hybrid Retrieval Support

## Goal
Introduce a first-class FTS5 table in Cloudflare D1 so downstream retrieval (see `dorphalsig/GitRag-Retrieve`) can blend keyword/BM25 scores with Vectorize embeddings. The indexer must write, update, and delete FTS rows alongside the existing `chunks` table.

## Requirements
- Create a virtual table `chunks_fts` (FTS5) mirroring chunk IDs and text content.
- Ensure DDL lives in `provisioning/d1/` so CI/CD can provision the structure.
- Write path: when persisting chunks, insert or replace the row in both `chunks` and `chunks_fts`.
- Delete path: remove rows from both tables for any deleted file IDs.
- Keep FTS updates within the pending→committed workflow so readers never see half-populated data.
- Provide regression tests (unit/integration) that assert:
  - FTS rows are created with the expected text.
  - Deletes remove both the canonical and FTS entries.
  - Queries against `chunks_fts` return results for known keywords in the fixture data.
- Document how retrieval services should issue hybrid queries (BM25 + Vectorize) in the docs/README.

## Non-Goals
- Implementing the retrieval/query service (handled separately in `GitRag-Retrieve`).
- Ranking logic changes; only data persistence is in scope.

## Risks
- Double-writing failures could leave FTS and canonical tables inconsistent. Mitigate by keeping operations within the existing pending/committed transaction flow.
- Large text payloads might inflate D1 size; consider storing normalized text (lowercase, trimmed) instead of raw content if quotas become an issue.

## Open Questions
- Do we strip Markdown/HTML tags before inserting into FTS? (Default: store raw chunk text to preserve fidelity; revisit if scoring suffers.)
- Should we create triggers in D1, or keep logic in the Python client? (Initial implementation will keep logic in the client for portability.)
