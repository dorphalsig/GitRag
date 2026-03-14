# GitRag

**GitRag** is a lightweight, Git-native RAG (Retrieval-Augmented Generation) indexer and retriever. It is architected to operate efficiently within the resource constraints of free-tier environments like GitHub Actions and Hugging Face Spaces.

## 💡 Why GitRag?

Unlike general-purpose RAG frameworks (like **QwenRag**) or high-throughput Rust-based ETL pipelines (like **CocoIndex**), GitRag is built for the Git lifecycle:

* **Git-Native Delta Logic**: Uses `git diff` and `ls-files` to process only what changed. It handles renames and deletions natively, preventing index bloat.
* **Repo & Branch Awareness**: Supports optional filtering by `repo` and `branch` during indexing and retrieval to narrow the search universe and improve precision.
* **Edge-First Hybrid Search**: Defaults to **libSQL (Turso)** for portable, low-latency hybrid search (Vector + BM25) on the edge.
* **Structured AST Chunking**: Uses Tree-sitter to break code into logical units (functions, classes) rather than arbitrary line counts.

## 🛠 Configuration

GitRag selects its persistence layer based on the `DB_PROVIDER` environment variable.

### 1. libSQL / Turso (Default)
Recommended for edge deployments and serverless environments.
* `DB_PROVIDER`: `libsql`
* `TURSO_DATABASE_URL`: Your database endpoint (e.g., `libsql://db-name.turso.io`).
* `TURSO_AUTH_TOKEN`: Your access token.

### 2. PostgreSQL / pgvector
Recommended for existing database clusters.
* `DB_PROVIDER`: `postgres`
* `DATABASE_URL`: Your connection string (e.g., `postgresql://user:pass@host:5432/db`).

## 📦 Deployment

### GitHub Action (The Indexer)
Automate your index updates on every push. Create `.github/workflows/index.yml`:

```yaml
name: GitRag Indexing

on:
  push:
    branches: [ "master" ]
  workflow_dispatch:
    inputs:
      full_scan:
        description: 'Perform a full repository scan?'
        type: boolean
        default: false

jobs:
  index:
    uses: dorphalsig/gitrag/.github/workflows/GitRag.yml@master
    with:
      repo: ${{ github.repository }}
      branch: ${{ github.ref_name }}
      db_provider: "libsql" or "postgres"
      full_index: ${{ inputs.full_scan || false }}
    secrets:
      DATABASE_URL: libsql://<db_name>.aws-eu-west-1.turso.io OR postgres://user@host/db
      DB_AUTH_TOKEN: ${{ secrets.dark }}
```

### Handling Timeouts for Large Repositories

For large repositories (500+ files), indexing may exceed GitHub Actions' 6-hour timeout. GitRag supports **soft timeouts** with automatic resume via database checkpointing.

**How Resume Works:**
1. On startup with `full_index: true`, GitRag queries the database for already-indexed paths
2. Files already in the database are skipped during processing
3. Since the database is remote, checkpointing works across different workflow runs and runners
4. When a soft timeout is exceeded, the indexer exits with code 75, allowing the workflow to retry

**Recommended Retry Pattern:**

```yaml
name: GitRag Indexing with Retry

on:
  push:
    branches: [ "master" ]
  workflow_dispatch:
    inputs:
      full_scan:
        description: 'Perform a full repository scan?'
        type: boolean
        default: false

jobs:
  index:
    uses: dorphalsig/gitrag/.github/workflows/GitRag.yml@master
    with:
      repo: ${{ github.repository }}
      db_provider: "libsql"
      full_index: ${{ inputs.full_scan || false }}
      soft_timeout: 14400  # 4 hours - adjust based on repo size
    secrets:
      DATABASE_URL: ${{ secrets.TURSO_DATABASE_URL }}
      DB_AUTH_TOKEN: ${{ secrets.TURSO_AUTH_TOKEN }}

  # Retry on timeout - dispatches new workflow with fresh 6-hour window
  retry-on-timeout:
    needs: index
    if: failure()
    runs-on: ubuntu-latest
    steps:
      - name: Trigger retry with full_index
        run: |
          gh workflow run "${{ github.workflow }}" \
            --ref "${{ github.ref }}" \
            -f full_scan=true \
            -f db_provider="libsql"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Important:** Always use `full_index: true` on retry to trigger the resume mechanism. Incremental mode (`--from-sha`/`--to-sha`) does not use checkpointing.

### Hugging Face Spaces (The Retriever)
Deploy as a **Docker** Space to host a search API or MCP server. For this use the [Dockerfile](Dockerfile).

**Requirements for HF Spaces Free:**
* **SDK**: Docker.
* **Hardware**: Minimum **"Standard CPU" (2 vCPU, 16GB RAM)**.
* **Secrets**: Add your `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` in the Space settings.

## 🖥 Hardware Requirements

| Component | Minimum | Recommended | Why? |
| :--- | :--- | :--- | :--- |
| **Indexer** | 2 vCPU, 4GB RAM | 2 vCPU, 7GB RAM | Standard GitHub Runners handle embedding tasks easily. |
| **Retriever** | 2 vCPU, 16GB RAM | 4 vCPU, 16GB RAM | The **Qwen3-Reranker-0.6B** model requires significant RAM to load and stay resident. |

*Note: On free CPU tiers, expect a ~10 second delay for the first query as the model loads into memory.* ☕

## 🏗 Supported Environments
* **Code**: Kotlin, Java, Dart, Python, and more via Tree-sitter.
* **Data**: Markdown (with breadcrumbs), JSON/JSONL, YAML, XML, TOML.
* **Interfaces**: Standard Python API and Model Context Protocol (MCP) server.

## 📜 License
Released under the MIT License.
