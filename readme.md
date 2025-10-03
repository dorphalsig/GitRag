# GitRag — Cloudflare D1 FTS5 & Vectorize Strategy (Feeding Only)

Scope
- Hybrid RAG feeding pipeline: maintain a keyword/BM25 search corpus in Cloudflare D1 (SQLite/FTS5) and a semantic vector index in Cloudflare Vectorize.
- Feeding only: this repository contains the extractor / chunker / embedding feed logic (no query/orchestration).
- Index is always built against the latest commit (HEAD) only.

Highlights
- Tree-sitter powered structural chunking for code and many config/formats.
- Semantic vectors stored in Cloudflare Vectorize (embeddings + metadata; no raw text).
- Metadata + content postings stored in Cloudflare D1 (FTS5).
- Contract: feed changes as an array of { file, action } objects: action ∈ { process, delete }.

Quick links
- CLI driver: src/Extractor.py (main sync: chunk → embed → upsert Vectorize + D1)
- Chunker: src/chunker.py (Tree-sitter + fallback packing)
- Grammar config: src/grammar_queries.json
- Tests: tests/

Supported languages / file families
- Code (Tree-sitter driven): Kotlin, Java, Dart, TypeScript, JavaScript, Python, Kotlin, etc. (see src/grammar_queries.json → code_extensions).
- Non-code structured formats (Tree-sitter grammars): Markdown, JSON/JSONL, YAML, XML/Plist, TOML, Gradle/KTS (when parsed), HTML, CSS, and many others listed in noncode_ts_grammar in src/grammar_queries.json.
- Fallback: plain text fallback chunking for unknown extensions.

Chunking strategy (what gets indexed)
- Primary approach: use Tree-sitter grammar queries (in src/grammar_queries.json) to extract semantic nodes and emit "chunks" that map to logical program/configuration units.
- Node taxonomy (examples):
  - Code: method nodes (primary unit), class/type nodes (context unit), fallback "chunk" for code outside parsed nodes or oversized regions.
  - Markdown: md_section, md_codeblock
  - JSON: json_object, json_array, json_prop
  - YAML: yaml_mapping, yaml_sequence, yaml_doc
  - XML/Plist: xml_element
  - TOML: toml_table, toml_kv
  - Gradle/KTS: treated as code if parseable, otherwise fallback.
- Fallback packing: when Tree-sitter isn't available for an extension, use a byte/line-based packing fallback that aims for consistent coverage.
- Size guidance & constants (from implementation):
  - SOFT_MAX_BYTES = 16_384 (packing target)
  - HARD_CAP_BYTES = 24_576 (absolute per-chunk)
  - NEWLINE_WINDOW = 2_048 (cut "nudge" window)
  - FALLBACK_OVERLAP_RATIO ≈ 0.10 (for fallback overlaps)
  - Logical guidance: preferred node sizes ~800–1,000 tokens; tiny nodes are merged, oversized nodes are split with minimal overlap.

Chunk identity
- Each emitted chunk is represented by the Chunk dataclass (src/chunker.py) with fields for repo, path, language, start/end row/col, start/end byte offsets, content, and optional signature.
- Stable chunk ID (implementation): chunk.id() computes SHA-256 of the string "{repo}::{path}::{start_bytes}::{end_bytes}" (see src/chunker.py).

Embedding model
- Model used: "CodeRankEmbed" via the SentenceTransformers API.
- Implementation notes:
  - Loader uses SentenceTransformer with a source that defaults to "nomic-ai/CodeRankEmbed".
  - You can override the local/HF source via the CODERANK_MODEL_DIR environment variable. The loader is lazy and raises on failure (fail-fast).
  - Embedding function: embed_texts(texts: List[str]) → List[List[float]] (batching used, batch_size=32). See src/Extractor.py.

Data flows & operations
- Input contract (from GH Workflow or --changes CLI): an array of objects: { "file": "<path>", "action": "process" | "delete" }.
  - process: parse file, emit nodes → upsert to Vectorize (vectors + metadata) and upsert to D1 (metadata + text/postings as appropriate).
  - delete: delete all nodes previously emitted for that path from both Vectorize and D1.
  - Renames are handled upstream as delete (old path) + process (new path).
- Latest-only policy: the index reflects HEAD only. IDs do not embed commit SHA; sha_head is stored in metadata for traceability.

Cloudflare D1 schema (SQLite)
- The extractor idempotently creates a metadata table for chunks. The SQL used in src/Extractor.py (ensure_tables) establishes the following table:
```SQL
CREATE TABLE IF NOT EXISTS chunks
(
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    start_row INTEGER NOT NULL,
    start_col INTEGER NOT NULL,
    end_row INTEGER NOT NULL,
    end_col INTEGER NOT NULL,
    start_bytes INTEGER NOT NULL,
    end_bytes INTEGER NOT NULL,
    signature TEXT,
    chunk TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
```
- Design notes:
  - The database holds chunk metadata and the textual chunk content for FTS posting.
  - The project design describes a contentless FTS5 postings approach (metadata table + a postings/FTS virtual table). In practice, the canonical table above is created by ensure_tables; the repository's higher-level design mentions an accompanying FTS5 postings setup (docs / docs_fts or similar) for efficient BM25/keyword search queries and to keep control over text storage.

Cloudflare Vectorize (embedding index) — metadata schema
- Vectorize stores embeddings (vectors) plus a minimal consistent set of metadata per chunk. Typical metadata fields emitted with each vector (see project design readme and src/Extractor.py usage) include:
  - id (stable chunk id)
  - repo
  - branch
  - sha_head
  - path
  - language
  - node_type
  - start_line, end_line
  - start_bytes, end_bytes
  - symbol (logical symbol name when applicable)
  - signature
  - logical_path
- Important: Vectorize stores embeddings and metadata only — raw chunk text is *not* stored in Vectorize (text is stored/posted into D1 for keyword search traceability).

Usage

1) As a CLI (local / CI)
- The extractor driver is src/Extractor.py. Example invocation:
  - python src/Extractor.py --changes @/path/to/changes.json
- The --changes argument accepts:
  - an inline JSON string like '[{"file":"src/foo.py","action":"process"}]'
  - or a file reference prefixed with @ (e.g. @/tmp/deltas.json).
- Behavior:
  - The script validates env/config, ensures D1 tables exist, and then iterates changes calling process_file(path, ...) or delete_file(path, ...).

2) Inputs from GitHub Actions (typical integration)
- The intended upstream usage is a GitHub Actions workflow that emits a per-PR delta array of {file, action} items.
- The pipeline is "multi-repo aware" in the sense that it can index chunks coming from multiple repositories, but the index policy is "HEAD-only": for any given repo the index reflects only HEAD (latest commit) — historic commits are not separately indexed. Each chunk stores sha_head in metadata so the origin commit can be traced.

Environment variables
- Required (must be set for the extractor to operate):
  - Cloudflare API auth (a Cloudflare API token or equivalent) — the Cloudflare SDK client is constructed from an environment token (the extractor validates presence). This token is required to call Vectorize and D1 APIs.
  - Cloudflare account identifier, Vectorize index name, and D1 database identifier — the extractor reads Cloudflare identifiers from the environment (cf_ids() validates presence of these identifiers at startup).
- Optional / override:
  - CODERANK_MODEL_DIR — when set, this path or HF identifier is passed to SentenceTransformer as the embedding source. Defaults to "nomic-ai/CodeRankEmbed".
- Notes:
  - The exact environment variable names for account/id/index/db are validated and read by cf_ids() in src/Extractor.py; ensure your CI secrets expose these values to the runtime environment.
  - The Cloudflare SDK/client will also need standard auth envs depending on your Cloudflare SDK setup.

Where this runs
- Intended runtime environments:
  - GitHub Actions (recommended): run in CI as a job that receives the per-PR deltas.
  - Locally: as a CLI for ad-hoc feeding (developer runs with proper env vars and a --changes file).
  - Any environment with network access to Cloudflare Vectorize and D1 where the required env variables/credentials are available.
- Note: monitor Vectorize and D1 usage against Cloudflare Free-tier limits when running at scale.

Development & testing
- Grammar configuration: src/grammar_queries.json centralizes language metadata and Tree-sitter queries. Add or update language support here.
- Chunker API: src/chunker.py exposes chunk_file(path, repo) → List[Chunk]; the tests exercise grammar config parity and chunking invariants in tests/.
- Tests include fixtures generation and chunker helpers (tests/fixtures.py, tests/test_chunker_helpers.py).

Design reminders and invariants
- Index is HEAD-only; stable IDs are used so reprocessing head produces the same IDs for stable file regions.
- Vectorize contains vectors + metadata only; text for keywords/FTS is represented via D1.
- Contract is per-file: process → upsert; delete → delete everywhere.

Example change list (CLI input)
```json
[
  { "file": "src/app/service.py", "action": "process" },
  { "file": "README.md", "action": "process" },
  { "file": "old/path/removed.rs", "action": "delete" }
]
```
Where to look in the repo
- src/chunker.py — chunking implementation, constants, Chunk dataclass
- src/Extractor.py — CLI driver, embedder loader, D1 ensure_tables SQL, process/delete flows
- src/grammar_queries.json — central grammar queries + extension maps
- tests/ — unit tests and fixtures showing expected behavior

License
- See repository root for licensing information (if absent, add appropriate LICENSE file).

If you want, I can:
- Produce a ready-to-commit README.md file with this content formatted for immediate use (and include a brief example workflow snippet).
```
