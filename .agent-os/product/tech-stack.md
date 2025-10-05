# Tech Stack

## Languages & Runtimes

- Python 3.9+ for the CLI, chunker, and persistence orchestration

## Core Libraries

- tree-sitter==0.23.2 for parsing source files into AST nodes
- tree-sitter-language-pack==0.9.1 supplying prebuilt grammars
- sentence-transformers==5.1.1 to load the CodeRankEmbed encoder
- numpy==2.x for vector manipulations and serialization
- cloudflare==4.3.1 for D1 and Vectorize API access

## Data & Storage

- Cloudflare Vectorize for embedding storage with cosine distance metric
- Cloudflare D1 (SQLite) for chunk metadata, text content, mutation status, and planned FTS5 postings to support keyword retrieval

## Infrastructure & Integrations

- Git and GitHub repositories as the ingestion source of truth
- Composite GitHub Action wrapper (root `action.yml`) to execute the CLI within CI
- Cloudflare API token + account identifiers supplied via environment variables

## Testing & Tooling

- pytest==8.4.2 for unit tests covering chunking helpers and fixtures
- Deterministic fixture generation utilities under `tests/fixtures.py`

## Extensibility Points

- `Calculators.EmbeddingCalculator` protocol to swap embedding providers
- `Persist.create_persistence_adapter` factory with `PersistenceAdapter` protocol to support alternative vector/database endpoints

## Configuration

- Required environment variables: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_VECTORIZE_INDEX`, `CLOUDFLARE_D1_DATABASE_ID`
- Optional overrides: `CODERANK_MODEL_DIR` for embedding model source, `GITRAG_PERSIST_ADAPTER` / `--adapter` for persistence selection, `--full` for initial indexing

## Deployment Targets

- GitHub Action runners on the free tier (Linux) with available memory for SentenceTransformers inference
- Local developer machines for ad-hoc indexing and debugging
