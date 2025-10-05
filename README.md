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
GitRag keeps Git and GitHub repositories searchable for retrieval-augmented generation by combining Tree-sitter powered chunking with lightweight embedding and persistence steps that fit within the GitHub Actions free tier. The pipeline focuses on incremental updates, storing metadata and vectors in Cloudflare while keeping raw text in D1 for keyword search.

Looking for the retrieval/query side? Check out the companion service in `dorphalsig/GitRag-Retrieve` which consumes these indexes for hybrid (vector + keyword) search.

## Roadmap vs Implementation
### Already Delivered
- Deterministic AST + document chunker covering code and structured formats with size-aware fallbacks.
- Git-aware CLI (`src/Indexer.py`) that inspects the last commit, filters binaries, and batches process/delete operations (includes `--full` initial indexing support).
- Cloudflare Vectorize + D1 persistence workflow with mutation tracking, schema bootstrapping, and verified end-to-end smoke tests.
- SentenceTransformers CodeRankEmbed calculator wired through chunking → embedding → persistence.
- GitHub Action wrapper (`action.yml`) to run the indexer in CI.
- Pluggable persistence adapter registry with a Cloudflare implementation shipped out of the box.
- Test coverage for chunking helpers/docs, language fixtures, CLI processing, binary detection, and persistence stubs.
- Contributor guide and mission principles published (`CONTRIBUTING.md`, `docs/mission.md`).

### Future Ideas
- Ship additional persistence adapter examples (e.g., Pinecone/Postgres) using the registry.
- Auto-provision helper scripts for Cloudflare resources.

## Feature-by-Feature Summary
### AST Chunking *(complete, tested)*
- `src/chunker.py` uses Tree-sitter grammars from `src/grammar_queries.json` to emit logical units (methods, config nodes) with byte-perfect spans.
- Fallback `_newline_aligned_ranges` ensures coverage when grammars fail, respecting `SOFT_MAX_BYTES`, `HARD_CAP_BYTES`, and overlap constants.

### Embedding *(complete, verified against Cloudflare runs)*
- `Calculators.CodeRankCalculator` loads CodeRankEmbed via SentenceTransformers, normalizes vectors, and exposes dimensionality.
- Embedding source is overrideable through `CODERANK_MODEL_DIR`.

### Persistence *(complete, verified against Cloudflare runs)*
- `src/Persist.py` creates the Cloudflare Vectorize index (with metadata indexes) and D1 table (`chunks` with pending/committed status`).
- Batches are written as NDJSON vectors, waited on via `processed_up_to_mutation`, and then marked committed inside D1.
- `delete_batch` reads stored IDs and removes vectors + rows.

### Git-aware CLI *(complete, tested)*
- `src/Indexer.py` determines the last commit range, classifies change actions, consolidates binary detection with a shared `BinaryDetector`, and forwards work to the persistence layer.
- `_process_files` loads chunk data, runs embeddings, and persists in a single batch per invocation.

### Extensibility *(adapter-ready)*
- Embedding calculators already adhere to `EmbeddingCalculator` protocol.
- Persistence adapters are pluggable via `persistence_registry.register_persistence_adapter(...)`; a Cloudflare implementation ships by default.
- GitHub Action packaging exposes the CLI as a reusable composite action.

### Community & Documentation *(complete)*
- See [`docs/mission.md`](docs/mission.md) for architectural principles and goals.
- Contribution process, testing expectations, and conduct guidelines live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Configuration
### Environment Variables (Required)
- `CLOUDFLARE_API_TOKEN` — Cloudflare API token used by the SDK client.
- `CLOUDFLARE_ACCOUNT_ID` — Account identifier for both Vectorize and D1 resources.
- `CLOUDFLARE_VECTORIZE_INDEX` — Target Vectorize index name.
- `CLOUDFLARE_D1_DATABASE_ID` — D1 database identifier that stores chunk metadata and text.

### Environment Variables (Optional)
- `CODERANK_MODEL_DIR` — Alternate Hugging Face repo or path for the embedding model.
- `GITRAG_PERSIST_ADAPTER` — Persistence adapter key (defaults to `cloudflare`).

### CLI Inputs & Flags
- `python src/Indexer.py <repo>` — positional argument representing the repository identifier stored in Vectorize metadata.
- `--full` — index every tracked and unignored text file using the shared binary detector (initial sync).
- `--adapter` — override the persistence adapter key (defaults to env or `cloudflare`).

### Heuristics & Knobs
- Chunk sizing constants: `SOFT_MAX_BYTES=16_384`, `HARD_CAP_BYTES=24_576`, `NEWLINE_WINDOW=2_048`, `FALLBACK_OVERLAP_RATIO≈0.10` (see `src/chunker.py`).
- Binary detection reads git attributes when available and samples up to 8 KiB of file bytes to guard against mislabeled binaries (`src/text_detection.py`).

## Usage
### Quickstart
1. Install dependencies: `pip install -r requirements.txt`.
2. Export required Cloudflare environment variables.
3. Run `python src/Indexer.py <namespace/repo>` from a Git repository you wish to index. The script inspects the last commit to generate process/delete actions and prints a JSON summary.

### Full Indexing
- Run `python src/Indexer.py <namespace/repo> --full` to build the index from scratch. The command enumerates all tracked and unignored files (`git ls-files` + unignored additions), filters binaries via `BinaryDetector`, and persists chunks in a single batch.

### Data Flow
1. Collect git changes (rename-aware) and filter binary files via `BinaryDetector`.
2. Chunk files with Tree-sitter and fallbacks; compute embeddings through `CodeRankCalculator`.
3. Persist chunk metadata/content to Cloudflare D1 and vectors to Cloudflare Vectorize with mutation tracking.

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
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          CLOUDFLARE_VECTORIZE_INDEX: your-vector-index
          CLOUDFLARE_D1_DATABASE_ID: your-d1-db
```

Inputs:
- `repo` (required) — repository identifier stored with each chunk.
- `full_index` (optional) — `true` to append `--full` for initial indexing.
- `persistence_adapter` (optional) — override adapter key; defaults to `cloudflare`.

### Provisioning Helpers
- Copy-paste D1 schema and Vectorize payloads from the `provisioning/` directory when setting up infrastructure.

## Development
### Tooling
- Python ≥3.9, Tree-sitter runtimes, SentenceTransformers.
- Install dev dependencies via `pip install -r requirements.txt`.

### Tests
- Run unit tests with `pytest` (covers chunking helpers, fixtures, binary detection, CLI batching, and persistence stubs). In environments without optional dependencies (Tree-sitter, numpy, sentence-transformers, Cloudflare SDK), fall back to `python3 -m unittest` which skips dependent suites.
- Fixture helpers in `tests/fixtures.py` generate deterministic inputs; tests auto-create missing sample files.

### Document Chunking & Metadata
- Non-code formats (Markdown, JSON/JSONL/NDJSON, YAML/TOML, CSV/TSV, plaintext) use a document-focused chunker with `DOC_GRAMMAR_VERSION=doc-chunker-v1`.
- Each document chunk now carries metadata persisted to Vectorize: `chunk_kind` (`doc`, `req`, `fence`, `table`, `json`, `yaml`, etc.), `heading_breadcrumb`, slug `anchors`, extracted `requirement_sentences`, inline/code-fence `code_refs`, per-table `table_delimiter`, detected `eol`, and fixed `overlap_bytes` (currently 256).
- Markdown segmentation tracks headings/code fences to build breadcrumbs and include fence languages in `code_refs`.
- JSON/TOML/YAML chunking associates top-level keys with breadcrumbs and surfaces requirement sentences from structured text.
- Plaintext falls back to UTF-8 windows; undecodable files degrade to byte-range chunking with the same metadata schema.

### Key Modules
- `src/Indexer.py` — CLI orchestrator and git integration.
- `src/chunker.py` — Chunk slicing logic and Tree-sitter integration.
- `src/Persist.py` — Cloudflare persistence adapter.
- `src/text_detection.py` — Shared binary detector used by CLI and upcoming initial indexing flows.
- `tests/` — Chunker, CLI, and persistence coverage.

## License
GitRag is released under the MIT License with attribution. See [`LICENSE`](LICENSE).
