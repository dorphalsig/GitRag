# GitRag Mission & Architectural Principles

GitRag exists to make source repositories queryable for retrieval-augmented
generation systems without compromising developer workflows. The following
principles guide implementation and review:

1. **Deterministic Indexing.** Chunk boundaries, signatures, and metadata must be
   reproducible for identical inputs. This enables confidence when diffing
   persisted indexes and simplifies incremental updates.
2. **Source of Truth in Git.** The indexer reads from the working tree and
   persists to external stores, but the canonical project state remains the Git
   repository. The indexer must never mutate repository contents.
3. **Minimal Operational Load.** Cloudflare Vectorize + D1 are the default
   persistence targets because they provide built-in free-tier quotas. Adapter
   hooks exist so other stores can be added without widening the runtime
   dependency surface.
4. **Sensitive Data Hygiene.** Chunking keeps provenance (path, repo) and
   discourages embedding secrets by honoring .gitignore, binary detection, and
   redaction helpers. Additional redaction passes should be treated as part of
   the ingestion pipeline rather than the core chunker.
5. **Incremental First.** The CLI always prioritizes delta indexing (last commit
   range) and treats `--full` as an explicit opt-in. Persist layer hooks
   guarantee idempotent writes and safe retries.
6. **Extensibility Through Small Interfaces.** Key seams—embedding calculators,
   persistence adapters, language grammars—are isolated behind narrow protocols
   so that new implementations can be dropped in without rewriting orchestration.

If a proposed change conflicts with these principles, capture the trade-offs in
code review so future maintainers can revisit the decision with full context.
