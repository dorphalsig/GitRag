# Tech Stack

## Languages & Runtimes

- Python 3.9+ for the CLI, chunker, and persistence orchestration

## Core Libraries

- tree-sitter==0.23.2 for parsing source files into AST nodes
- tree-sitter-language-pack==0.9.1 supplying prebuilt grammars
- sentence-transformers==5.1.1 to load the CodeRankEmbed encoder
- numpy==2.x for vector manipulations and serialization
- SQLAlchemy==2.0.x for ORM-free database interactions
- sqlalchemy-libsql==0.2.x for remote libSQL connectivity

## Data & Storage

- Turso/libSQL for chunk metadata, text content, embeddings, and FTS via SQLite-compatible storage

## Infrastructure & Integrations

- Git and GitHub repositories as the ingestion source of truth
- Composite GitHub Action wrapper (root `action.yml`) to execute the CLI within CI
- libSQL connection string supplied via environment variables

## Testing & Tooling

- pytest==8.4.2 for unit tests covering chunking helpers and fixtures
- Deterministic fixture generation utilities under `tests/fixtures.py`

## Extensibility Points

- `Calculators.EmbeddingCalculator` protocol to swap embedding providers
- `Persist.create_persistence_adapter` factory with `PersistenceAdapter` protocol to support alternative vector/database endpoints

## Configuration

- Required environment variables: `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`
- Optional overrides: `CODERANK_MODEL_DIR` for embedding model source, `--full` for initial indexing

## Deployment Targets

- GitHub Action runners on the free tier (Linux) with available memory for SentenceTransformers inference
- Local developer machines for ad-hoc indexing and debugging
