FROM python:3.11-slim

# 1. Install System Dependencies
# Git is required to clone the repository at runtime
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2. Clone the Repository
# We clone directly into /app to match the expected structure
WORKDIR /app
RUN git clone https://github.com/dorphalsig/gitrag.git .

# 3. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Setup Environment
# Ensure Python can find the source packages
ENV PYTHONPATH="/app/packages/core/src:/app/packages/mcp-server/src" \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/tmp/.cache/huggingface"

# 5. Create the Runtime Entrypoint ("Glue Script")
# This script includes a strict pre-flight check for DB credentials
RUN echo 'import os, sys\n\
\n\
def check_config():\n\
    db_url = os.environ.get("TURSO_DATABASE_URL")\n\
    db_token = os.environ.get("TURSO_AUTH_TOKEN")\n\
    \n\
    if not db_url or not db_token:\n\
        # Use stderr so it shows up as an error in logs\n\
        sys.stderr.write("\\n")\n\
        sys.stderr.write("!" * 60 + "\\n")\n\
        sys.stderr.write("!! FATAL ERROR: MISSING DATABASE CONFIGURATION             !!\\n")\n\
        sys.stderr.write("!!                                                         !!\\n")\n\
        sys.stderr.write("!! You must set the following Secrets in your Space:       !!\\n")\n\
        sys.stderr.write("!!   - TURSO_DATABASE_URL                                  !!\\n")\n\
        sys.stderr.write("!!   - TURSO_AUTH_TOKEN                                    !!\\n")\n\
        sys.stderr.write("!!                                                         !!\\n")\n\
        sys.stderr.write("!! Go to: Settings > Variables and secrets                 !!\\n")\n\
        sys.stderr.write("!" * 60 + "\\n")\n\
        sys.stderr.write("\\n")\n\
        sys.exit(1)\n\
    return db_url, db_token\n\
\n\
if __name__ == "__main__":\n\
    # 1. Pre-flight Check\n\
    db_url, db_token = check_config()\n\
\n\
    # 2. Imports (Delayed until after config check to speed up failure)\n\
    try:\n\
        from Persist import PersistInLibsql\n\
        from Retriever import Retriever, Qwen3Reranker\n\
        from Calculators.CodeRankCalculator import CodeRankCalculator\n\
        from gitrag_mcp_server.server import create_mcp_server\n\
    except ImportError as e:\n\
        sys.stderr.write(f"FATAL: Failed to import GitRag modules: {e}\\n")\n\
        sys.exit(1)\n\
\n\
    print("--- GitRag Standalone Retriever Starting ---")\n\
    \n\
    # 3. Initialize Models & DB\n\
    print("Loading Embedding Model (Qwen3-0.6B)...")\n\
    calculator = CodeRankCalculator()\n\
    \n\
    print("Connecting to LibSQL...")\n\
    persistence = PersistInLibsql(db_url, db_token)\n\
    \n\
    print("Loading Reranker (Qwen3-0.6B)...")\n\
    reranker = Qwen3Reranker()\n\
    \n\
    retriever = Retriever(persistence, calculator, reranker=reranker)\n\
    \n\
    # 4. Start Server\n\
    print("Starting MCP Server on port 7860...")\n\
    mcp = create_mcp_server(retriever=retriever)\n\
    mcp.run(transport="sse", port=7860, host="0.0.0.0")\n\
' > /app/run_space.py

# 6. Permissions & User
# Hugging Face Spaces runs as user 1000 and needs write access to /tmp
RUN mkdir -p /tmp/.cache/huggingface && chmod -R 777 /tmp
USER 1000
EXPOSE 7860

# 7. Launch
CMD ["python", "/app/run_space.py"]