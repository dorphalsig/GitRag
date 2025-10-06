# GitRag — A RAG Indexer for Git Repositories

## Index
- [Vision](#vision)
- [Roadmap vs Implementation](#roadmap-vs-implementation)
- [Feature-by-Feature Summary](#feature-by-feature-summary)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [License](#license)

## Vision
GitRag keeps Git and GitHub repositories searchable for retrieval-augmented generation by combining Tree-sitter powered chunking with lightweight embedding and persistence steps that fit within the GitHub Actions free tier. The pipeline focuses on incremental updates, storing metadata, raw text, and embeddings inside libSQL so hybrid vector/keyword search remains fast and fully portable.

Looking for the retrieval/query side? Check out the companion service in `dorphalsig/GitRag-Retrieve` which consumes these indexes for hybrid (vector + keyword) search.

## Roadmap vs Implementation
### Already Delivered
- Deterministic AST + document chunker covering code and structured formats with size-aware fallbacks.
- Git-aware CLI (`src/Indexer.py`) that inspects the last commit, filters binaries, and batches process/delete operations (includes `--full` initial indexing support).
- libSQL persistence workflow with schema bootstrapping, transactional writes, and verified end-to-end smoke tests.
- SentenceTransformers CodeRankEmbed calculator wired through chunking → embedding → persistence.
- GitHub Action wrapper (`action.yml`) to run the indexer in CI.
- Pluggable persistence adapter registry with libSQL provided out of the box.
- Test coverage for chunking helpers/docs, language fixtures, CLI processing, binary detection, and persistence stubs.
- Contributor guide and mission principles published (`CONTRIBUTING.md`, `docs/mission.md`).

### Future Ideas
- Ship additional persistence adapter examples (e.g., Pinecone/Postgres) using the registry.
- Publish libSQL provisioning helpers and migration tooling.

## Feature-by-Feature Summary
### AST Chunking *(complete, tested)*
- `src/chunker.py` uses Tree-sitter grammars from `src/grammar_queries.json` to emit logical units (methods, config nodes) with byte-perfect spans.
- Fallback `_newline_aligned_ranges` ensures coverage when grammars fail, respecting `SOFT_MAX_BYTES`, `HARD_CAP_BYTES`, and overlap constants.

### Embedding *(complete, verified against libSQL runs)*
- `Calculators.CodeRankCalculator` loads CodeRankEmbed via SentenceTransformers, normalizes vectors, and exposes dimensionality.
- Embedding source is overrideable through `CODERANK_MODEL_DIR`.

### Persistence *(complete, verified against libSQL runs)*
- `src/Persist.py` stores chunk metadata, embeddings (BLOB), and FTS content inside libSQL via SQLAlchemy.
- Batches execute within a single transaction per chunk batch; FTS mirrors are refreshed transactionally.
- `delete_batch` resolves chunk IDs by path and removes rows from both primary and FTS tables.

### Git-aware CLI *(complete, tested)*
- `src/Indexer.py` determines the last commit range, classifies change actions, consolidates binary detection with a shared `BinaryDetector`, and forwards work to the persistence layer.
- `_process_files` loads chunk data, runs embeddings, and persists in a single batch per invocation.

### Extensibility *(adapter-ready)*
- Embedding calculators already adhere to `EmbeddingCalculator` protocol.
- Persistence adapters remain pluggable through `persistence_registry`, with libSQL provided by default.
- GitHub Action packaging exposes the CLI as a reusable composite action.

### Community & Documentation *(complete)*
- See [`docs/mission.md`](docs/mission.md) for architectural principles and goals.
- Contribution process, testing expectations, and conduct guidelines live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Configuration
### Environment Variables (Required)
- `TURSO_DATABASE_URL` — base libSQL endpoint (e.g. `libsql://<host>/<db>`).
- `TURSO_AUTH_TOKEN` — Turso/libSQL auth token.

### Environment Variables (Optional)
- `CODERANK_MODEL_DIR` — Alternate Hugging Face repo or path for the embedding model.

### CLI Inputs & Flags
- `python src/Indexer.py <repo>` — positional argument representing the repository identifier stored alongside chunks.
- `--full` — index every tracked and unignored text file using the shared binary detector (initial sync).

### Heuristics & Knobs
- Chunk sizing constants: `SOFT_MAX_BYTES=16_384`, `HARD_CAP_BYTES=24_576`, `NEWLINE_WINDOW=2_048`, `FALLBACK_OVERLAP_RATIO≈0.10` (see `src/chunker.py`).
- Binary detection reads git attributes when available and samples up to 8 KiB of file bytes to guard against mislabeled binaries (`src/text_detection.py`).

## Usage
### Quickstart
1. Install dependencies: `pip install -r requirements.txt`.
2. Export `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`.
3. Run `python src/Indexer.py <namespace/repo>` from a Git repository you wish to index. The script inspects the last commit to generate process/delete actions and prints a JSON summary.

### Full Indexing
- Run `python src/Indexer.py <namespace/repo> --full` to build the index from scratch. The command enumerates all tracked and unignored files (`git ls-files` + unignored additions), filters binaries via `BinaryDetector`, and persists chunks in a single batch.

### Data Flow
1. Collect git changes (rename-aware) and filter binary files via `BinaryDetector`.
2. Chunk files with Tree-sitter and fallbacks; compute embeddings through `CodeRankCalculator`.
3. Persist chunk metadata, embeddings, and FTS rows to libSQL inside a single transaction.

### Supported File Families
- Code: Kotlin, Java, Dart, Pascal, and others specified in `grammar_queries.json` (`code_extensions`).
- Structured content: Markdown, JSON/JSONL, YAML, XML/Plist, TOML, HTML, CSS, etc. via `noncode_ts_grammar`.
- Unknown extensions fall back to newline-aligned byte windowing.

### GitHub Action Wrapper
Use the bundled composite action to run GitRag in any repository:

```yaml
jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          repo: your-org/your-repo
          full_index: true
        env:
          TURSO_DATABASE_URL: ${{ secrets.TURSO_DATABASE_URL }}
          TURSO_AUTH_TOKEN: ${{ secrets.TURSO_AUTH_TOKEN }}
```

Inputs:
- `repo` (required) — repository identifier stored with each chunk.
- `full_index` (optional) — `true` to append `--full` for initial indexing.
- `turso_database_url` / `turso_auth_token` — supplied via workflow `with:` values or repo secrets and forwarded as env vars.

### Provisioning Helpers
- Apply the statements in `provisioning/libsql/schema.sql` when creating a new database.

## Development
### Tooling
- Python ≥3.9, Tree-sitter runtimes, SentenceTransformers.
- Install dev dependencies via `pip install -r requirements.txt`.

### Tests
- Run unit tests with `pytest` (covers chunking helpers, fixtures, binary detection, CLI batching, and persistence stubs). In environments without optional dependencies (Tree-sitter, numpy, sentence-transformers, libSQL drivers), fall back to `python3 -m unittest` which skips dependent suites.
- Fixture helpers in `tests/fixtures.py` generate deterministic inputs; tests auto-create missing sample files.

### Document Chunking & Metadata
- Non-code formats (Markdown, JSON/JSONL/NDJSON, YAML/TOML, CSV/TSV, plaintext) use a document-focused chunker with `DOC_GRAMMAR_VERSION=doc-chunker-v1`.
- Each document chunk now carries metadata persisted to libSQL: `chunk_kind` (`doc`, `req`, `fence`, `table`, `json`, `yaml`, etc.), `heading_breadcrumb`, slug `anchors`, extracted `requirement_sentences`, inline/code-fence `code_refs`, per-table `table_delimiter`, detected `eol`, and fixed `overlap_bytes` (currently 256).
- Markdown segmentation tracks headings/code fences to build breadcrumbs and include fence languages in `code_refs`.
- JSON/TOML/YAML chunking associates top-level keys with breadcrumbs and surfaces requirement sentences from structured text.
- Plaintext falls back to UTF-8 windows; undecodable files degrade to byte-range chunking with the same metadata schema.

### Key Modules
- `src/Indexer.py` — CLI orchestrator and git integration.
- `src/chunker.py` — Chunk slicing logic and Tree-sitter integration.
- `src/Persist.py` — libSQL persistence adapter implemented with SQLAlchemy.
- `src/text_detection.py` — Shared binary detector used by CLI and upcoming initial indexing flows.
- `tests/` — Chunker, CLI, and persistence coverage.

## License
GitRag is released under the MIT License with attribution. See [`LICENSE`](LICENSE).
