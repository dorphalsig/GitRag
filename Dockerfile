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
    HF_HOME="/tmp/.cache/huggingface" \
    RETRIEVAL_OPTIMIZER="onnx"

# 5. Permissions & User
# Hugging Face Spaces runs as user 1000 and needs write access to /tmp
RUN mkdir -p /tmp/.cache/huggingface && chmod -R 777 /tmp
USER 1000
EXPOSE 7860

# 6. Launch
CMD ["python", "packages/core/src/Retriever.py", "--server", "--port", "7860"]
