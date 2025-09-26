# Cloudflare D1 FTS5 & Vectorize Strategy — Hybrid RAG (Feeding Only)

**Scope**: Define how to index and maintain both a **keyword/BM25** search corpus in **Cloudflare D1 (SQLite/FTS5)** and a **semantic vector index** in **Cloudflare Vectorize**. This document covers **feeding only** (no query/orchestration). Indexing is always against the **latest commit (HEAD)**.

---

## 1) Inputs & Contract (from GH Workflow)

* Workflow emits **deltas** per PR as an array of objects:

  * `{ file: <path>, action: <process|delete> }`
* **Semantics**:

  * `process` → (re)parse file, emit nodes and index them
  * `delete` → remove all nodes previously emitted for that path
  * **Renames** are handled upstream as `delete` (old path) + `process` (new path)

---

## 2) Node Taxonomy (what gets indexed)

**Code files** (Kotlin/Java/Dart/TS/JS/Python, etc.):

* `method` nodes (primary unit)
* `class`/`type` nodes (context unit)
* `chunk` (fallback for code outside AST nodes or oversized regions)

**Config/Docs** (Markdown/JSON/YAML/XML/TOML/Gradle):

* Markdown: `md_section`, `md_codeblock`
* JSON: `json_object`, `json_array`, `json_prop`
* YAML: `yaml_mapping`, `yaml_sequence`, `yaml_doc`
* XML: `xml_element`
* TOML: `toml_table`, `toml_kv`
* Gradle/KTS: treat as code if language parsed; otherwise `chunk`

---

## 3) Stable IDs (shared across Vectorize & D1)

* ID = `sha1(repo|path|logical-identity|start_line:end_line)`

  * Code: `logical-identity = fully_qualified_symbol`
  * Config/docs: `logical-identity = logical_path`
  * Fallback chunks: omit logical identity

---

## 4) Metadata (minimal, consistent)

For every node:

* `id`
* `repo`, `branch`, `sha_head`
* `path`
* `language`
* `node_type`
* `start_line`, `end_line`
* `symbol`
* `signature`
* `logical_path`

---

## 5) Vectorize Index (semantic vectors)

* **What is stored**: embedding + metadata (no raw text)
* **Index name**: configurable per repo (default `<owner>_<repo>_code`)
* **Process**:

  * On `process`, parse file → emit nodes → embed (via Workers AI) → upsert embedding + metadata keyed by `id`
  * On `delete`, remove all node IDs for the file path from Vectorize
* **Batching**: group embeddings (64–256 nodes) for efficiency
* **Backoff**: handle 429/5xx with exponential retry (max 5 attempts, 30s cap)
* **Metadata discipline**: include `path`, `start_line`, `end_line`, `node_type`, `language`, `symbol`/`logical_path`, `sha_head`; truncate large arrays

---

## 6) D1 Schema (contentless FTS5)

### 6.1 Structured metadata table

```
docs(
  id TEXT PRIMARY KEY,
  repo TEXT NOT NULL,
  branch TEXT NOT NULL,
  sha_head TEXT NOT NULL,
  path TEXT NOT NULL,
  language TEXT,
  node_type TEXT NOT NULL,
  symbol TEXT,
  signature TEXT,
  logical_path TEXT,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL
)
```

### 6.2 FTS index table (contentless)

```
docs_fts USING fts5(
  id,
  content='',
  tokenize='unicode61'
)
```

**Recommended indexes on `docs`**

* `INDEX docs_path ON docs(path)`
* `INDEX docs_repo_path ON docs(repo, path)`
* `INDEX docs_sha ON docs(sha_head)`

---

## 7) Feeding Operations

### 7.1 PROCESS (upsert one file)

1. Parse file with **tree-sitter-language-pack** (preferred) to emit nodes per taxonomy.
2. For each node:

   * Upsert metadata into `docs`
   * Insert node text into `docs_fts` (postings only)
   * Upsert embedding + metadata into Vectorize

### 7.2 DELETE (remove one file)

* Lookup all IDs for `path`
* Delete postings from `docs_fts`
* Delete rows from `docs`
* Delete embeddings from Vectorize

---

## 8) Text to Index

* Use **tree-sitter-language-pack** grammars to segment code/configs/docs by node taxonomy
* Node text = exact slice (method/class body, JSON object, XML element, etc.)
* Preserve identifiers, paths, and comments
* Node size guidance: \~800–1,000 tokens; merge tiny nodes; split oversized nodes (≤3k chars) with minimal overlap

---

## 9) Latest-Only Policy

* Index reflects **HEAD only**
* IDs do not include commit SHA (stable across updates)
* `sha_head` stored in metadata for traceability

---

## 10) Validation & Monitoring

* Verify that `docs` row count matches `docs_fts` postings count after batch
* Periodically test FTS queries for rare identifiers and confirm metadata rows exist
* Monitor Vectorize + D1 usage against Free-tier limits

---

## 11) Summary

* **Vectorize**: embeddings + metadata, no raw text
* **D1**: metadata (`docs`) + postings (`docs_fts`)
* Contract: `{file, action}` with `process` → upsert everywhere, `delete` → remove everywhere
* Tree-sitter provides consistent node boundaries across code and config formats
* Index = HEAD only, deletions handled cleanly

---

## Appendix — Grammar Configuration JSON

`src/grammar_queries.json` centralizes the language metadata that powers the chunker:

```json
{
  "code_extensions": { "kt": "kotlin", "java": "java", "pas": "pascal", ... },
  "noncode_ts_grammar": { "md": "markdown", "yaml": "yaml", ... },
  "grammar_queries": {
    "Package": { "java": ["(package_declaration) @package"], ... },
    "Type": { "kotlin": ["(class_declaration) @type", ...], ... },
    ...
  }
}
```

- `code_extensions` maps file extensions (without leading `.`) to Tree-sitter language identifiers used for code chunking.
- `noncode_ts_grammar` maps non-code extensions to the Tree-sitter grammar name used for structural chunking.
- `grammar_queries` is a nested object keyed by semantic category (`Package`, `Type`, `Method`, etc.). Each category maps language identifiers to an array of S-expression query strings whose `@capture` matches the category name (lowercased). Helpers such as `_build_containers` and `_get_package_name` derive their behavior from this data, so new languages or grammar tweaks should extend this JSON file rather than hardcoding values in Python.
