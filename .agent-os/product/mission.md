# Product Mission

## Pitch

GitRag is a repository indexing pipeline that helps platform and applied AI teams keep GitHub codebases searchable for retrieval-augmented generation workflows by providing AST-aware chunking, incremental sync, and libSQL-native persistence.

## Users

### Primary Customers

- AI Platform Teams: Maintain retrieval infrastructure and need deterministic ingestion of repository knowledge for LLM-powered assistants.
- Developer Experience Engineers: Curate internal knowledge bases and require lightweight tooling that runs inside standard GitHub tiers.

### User Personas

**AI Platform Engineer** (28-45 years old)
- **Role:** Owns model integration and tooling for ML product squads
- **Context:** Coordinates RAG ingestion jobs across multiple GitHub organizations with strict security controls
- **Pain Points:** Hard-to-maintain custom chunkers, brittle pipelines that break on new languages
- **Goals:** Automate ingestion, guarantee consistent chunking, stay within GitHub Action resource budgets

**Developer Productivity Lead** (30-50 years old)
- **Role:** Delivers internal tooling and documentation platforms for engineering teams
- **Context:** Manages documentation search across many repositories without dedicated infra resources
- **Pain Points:** Manual full-repo re-indexing, lack of audit trails, slow iteration on language coverage
- **Goals:** Enable incremental updates, plug into existing CI, keep costs and operational load low

## The Problem

### Stale and fragmented code knowledge

Engineering teams need their latest code and configuration reflected in RAG indexes, yet manual ingestion and coarse chunking leave assistants unaware of recent changes. The resulting gaps cost developers time searching, and manual re-indexing exceeds GitHub free-tier budgets.

## Differentiators

- Tree-sitter powered AST chunking across code and structured config files with size-aware fallbacks
- Incremental git-diff driven ingestion that respects rename and binary heuristics
- libSQL vector persistence that provisions DiskANN cosine indexes and manages vector blobs without external services
- Pluggable embedding calculator interface that already supports CodeRankEmbed with minimal runtime footprint
- Planned FTS-backed keyword postings in libSQL to power hybrid (vector + lexical) retrieval alongside the GitRag-Retrieve service

## Key Features

- Deterministic chunker that covers multiple languages and structured formats with test coverage
- CLI orchestrator that resolves commit deltas, filters binaries, and batches persistence operations
- Extensible embedding calculator powered by SentenceTransformers with configurable model source
- Persistence layer that provisions libSQL schema (vector blobs + DiskANN index) while handling upserts and mutation tracking
- Fixture-backed helper library to generate realistic test data for chunking behaviors

## Use Cases

- Verify that a code change satisfies stated requirements by retrieving the relevant specification fragments alongside the implementation for review.
- Locate requirement language tied to a specific domain topic (e.g., pricing calculations) so developers can cross-reference business rules while coding or reviewing PRs.
