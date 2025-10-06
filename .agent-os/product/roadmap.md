# Product Roadmap

## Phase 0: Already Completed

- [x] Tree-sitter powered chunker covering code and structured config formats with fallback strategies
- [x] Commit-diff aware CLI that batches process/delete actions per repository change set
- [x] libSQL persistence workflow with transactional writes and schema management
- [x] Pluggable embedding calculator built on CodeRankEmbed via SentenceTransformers
- [x] Fixture-backed pytest suite validating chunking helpers and grammar configuration

## Phase 1: Current Development

**Goal:** Validate end-to-end indexing and surface configuration guidance
**Success Criteria:** Automated tests for persistence path, documented env requirements, sample run recorded

### Features

- [ ] Smoke-test embeddings and persistence against libSQL sandboxes `[M]`
- [ ] Harden binary detection by consolidating git heuristics and fallbacks `[S]`
- [ ] Document configuration knobs, env variables, and usage walkthrough in README `[S]`
- [ ] Add requirement-topic tagging and search so reviewers can surface pricing/compliance specs next to code changes `[M]`
- [ ] Ensure libSQL FTS tables are populated for hybrid keyword retrieval `[M]`
- [ ] Add regression tests that query the FTS table for known keywords `[S]`

### Dependencies

- Turso/libSQL staging credentials
- GitHub repository with representative language mix for validation
- Curated requirement documents (e.g., pricing policies) for indexing validation
- Updated libSQL schema (FTS table + provisioning scripts)

## Phase 2: Actions & Extensibility

**Goal:** Ship reusable GitHub Action wrapper and abstractions for storage backends
**Success Criteria:** Action published, storage interface extracted, sample adapters verified

### Features

- [x] Provide composite GitHub Action wrapper that runs the CLI across repositories `[M]`
- [x] Add initial indexing flag to process all non-ignored files with binary-safe filtering `[S]`
- [x] Extract persistence adapters to allow alternative vector/database endpoints `[M]`
- [ ] Publish Action metadata to GitHub Marketplace `[S]`
- [ ] Ship CLI subcommand that links implementation chunks to the retrieved requirement snippets for change validation `[M]`

### Dependencies

- GitHub Marketplace submission process
- Candidate alternative vector/database SDKs for adapter examples
- Requirement-topic metadata emitted by chunker and stored in libSQL

## Phase 3: Scale & Governance

**Goal:** Improve reliability, observability, and openness for broader adoption
**Success Criteria:** Robust logging/metrics, OSS license applied, contributor docs available

### Features

- [ ] Add structured logging and metrics hooks for CLI runs `[S]`
- [ ] Publish detailed contributor guide and testing strategy `[S]`
- [ ] Apply a permissive OSS license with attribution clause `[XS]`

### Dependencies

- Decision on telemetry stack (e.g., stdout JSON, optional OpenTelemetry)
- Legal review of permissive licensing options (MIT/BSD/Apache)
