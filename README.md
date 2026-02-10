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
    branches: [main]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2 # Required to calculate the git diff

      - name: Run GitRag Indexer
        uses: dorphalsig/gitrag@master
        with:
          repo: ${{ github.repository }}
          branch: ${{ github.ref_name }} # Optional: track branch for filtering
          turso_database_url: ${{ secrets.TURSO_DATABASE_URL }}
          turso_auth_token: ${{ secrets.TURSO_AUTH_TOKEN }}
          full_index: false # Use 'true' for the initial repo sync
```

### Hugging Face Spaces (The Retriever)
Deploy as a **Docker** Space to host a search API or MCP server.

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

## ⚠️ Compatibility Warning
**Embeddings are model-specific.** If you change your embedding model (e.g., from Qwen 2.5 to 3, or from 0.6B to 8B), you **must** perform a full re-index (`--full`). Mixing vectors from different versions or sizes will result in zero retrieval accuracy.

## 🏗 Supported Environments
* **Code**: Kotlin, Java, Dart, Python, and more via Tree-sitter.
* **Data**: Markdown (with breadcrumbs), JSON/JSONL, YAML, XML, TOML.
* **Interfaces**: Standard Python API and Model Context Protocol (MCP) server.

## 📜 License
Released under the MIT License.
