# GitRag

GitRag is a Git-aware code indexing and retrieval toolkit.

It focuses on two jobs:
- indexing repository content into a vector-capable database
- retrieving relevant code snippets, with optional repo/branch filtering and MCP exposure

The current codebase includes:
- a Git-based indexer that can run in full or delta mode
- Tree-sitter-based chunking for code and structured text handling for common data files
- libSQL and PostgreSQL persistence backends
- a retriever with optional reranking
- an MCP server that exposes a `search_code` tool

## What GitRag does well

- **Indexes Git changes** instead of blindly reprocessing everything on every run
- **Understands deletions** and removes deleted paths from the index
- **Supports full re-indexing** with resume-friendly behavior when combined with remote persistence
- **Filters retrieval by repo and branch** when you want narrower search scope
- **Skips ignored paths** via `GITRAG_IGNORE`
- **Can serve search over MCP** with authentication enabled by default

## Configuration

GitRag selects its persistence backend with `DB_PROVIDER`.

### libSQL

Use `DB_PROVIDER=libsql` and provide:
- `DATABASE_URL` (the code also accepts `TURSO_DATABASE_URL`)
- `DB_AUTH_TOKEN` (the code also accepts `TURSO_AUTH_TOKEN`)

### PostgreSQL

Use `DB_PROVIDER=postgres` and provide:
- `DATABASE_URL`
- `DB_AUTH_TOKEN`

### Excluding files

Use `GITRAG_IGNORE` with a comma- or semicolon-separated list of glob patterns.

Example:

```bash
export GITRAG_IGNORE="dist/**,build/**,*.min.js"
```

## Indexing behavior

The indexer accepts:
- a required `repo` identifier
- `--full` for a full scan
- `--branch` for branch-aware indexing
- `--from-sha` and `--to-sha` for delta indexing across a commit range

For long runs, GitRag also supports a soft timeout via the `SOFT_TIMEOUT` environment variable. When the timeout is exceeded, the indexer exits with code `75`, which is useful for retry-based workflows.

## GitHub Action

The repository includes a composite GitHub Action in `action.yml` for running the indexer.

Basic example:

```yaml
name: GitRag Indexing

on:
  push:
    branches: ["master"]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dorphalsig/gitrag@master
        with:
          repo: ${{ github.repository }}
          db_provider: libsql
          database_url: ${{ secrets.TURSO_DATABASE_URL }}
          db_auth_token: ${{ secrets.TURSO_AUTH_TOKEN }}
          branch: ${{ github.ref_name }}
```

Optional inputs supported by the action:
- `full_index`
- `soft_timeout`

If `full_index` is not enabled, the action passes the Git commit range to the indexer so it can process changes incrementally.

## MCP server

GitRag ships an MCP server package that exposes a `search_code` tool backed by the retriever.

The current server implementation creates a FastMCP server and returns:
- structured result objects
- a markdown rendering of the matched snippets
- XML-safe formatted snippets for downstream consumers

### MCP auth setup

Authentication is enabled by default.

To run the authenticated MCP server, set:
- `SCALEKIT_ENVIRONMENT_URL`
- `SCALEKIT_CLIENT_ID`
- `SCALEKIT_RESOURCE_ID`

Optional:
- `MCP_BASE_URL` — overrides the callback/base URL used by the Scalekit provider. If unset, GitRag falls back to `http://127.0.0.1:8000/mcp`.

Auth behavior is controlled like this:
- `GITRAG_MCP_DISABLE_AUTH=1` disables auth
- otherwise, `GITRAG_MCP_REQUIRE_AUTH=true|false` can explicitly enable or disable auth
- if neither variable is set, auth stays enabled

For local development, disabling auth can be useful while wiring up clients or tests. It should not be treated as the default production setup.

### Minimal server wiring

```python
from gitrag_mcp_server.server import create_mcp_server

mcp = create_mcp_server(retriever=my_retriever)
mcp.run(transport="sse", port=8000, host="0.0.0.0")
```

## Retrieval

The retriever returns the top matching chunks for a query. It can:
- search with vector + text persistence backends
- limit results with `top_k`
- scope results by `repo` and `branch`
- optionally rerank candidates with `Qwen/Qwen3-Reranker-0.6B`

## Supported content

Current tests and code cover:
- source files chunked through Tree-sitter
- markdown
- JSON / JSONL
- YAML
- XML
- TOML

## License

MIT.
