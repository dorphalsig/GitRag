"""Centralized project constants."""
import os

# Embedding constants
EMBEDDING_MODEL_ID = os.getenv("model", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIMENSIONS = int(os.getenv("dimensions", 1_024))
EMBEDDING_BATCH_SIZE = int(os.getenv("batch_size", 64))
DYNAMIC_SEQ_LENGTH = os.getenv("DYNAMIC_SEQ_LENGTH", "true").lower() == "true"
SOFT_TIMEOUT_SECONDS = int(os.getenv("SOFT_TIMEOUT", 0))
EXIT_CODE_TIMEOUT = 75

# Retriever constants
RETRIEVAL_QUERY_PREFIX = os.getenv("retrieval_prefix", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:")
DEFAULT_RERANKER_MODEL = os.getenv("reranker","jinaai/jina-reranker-v3")
DEFAULT_RERANK_TASK_INSTRUCTION = os.getenv("rerank_instruction","")
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_TOP_K = int(os.getenv("topk", 10))
DEFAULT_INITIAL_RETRIEVAL_LIMIT = DEFAULT_TOP_K*5

# Persistence constants
DEFAULT_TABLE_NAME = "chunks"
DEFAULT_FTS_TABLE_SUFFIX = "_fts"
HYBRID_SEARCH_VECTOR_WEIGHT = 0.7
HYBRID_SEARCH_KEYWORD_WEIGHT = 0.3
POSTGRES_FTS_LANGUAGE = "english"
DEFAULT_DB_PROVIDER = "libsql"

# Chunker constants
SOFT_MAX_BYTES = 1024
HARD_CAP_BYTES = 2048
NEWLINE_WINDOW = 2_048
FALLBACK_OVERLAP_RATIO = 0.10

# Document Chunker constants
DOC_OVERLAP_BYTES = 256
DOC_MIN_CHUNK_BYTES = 512
DOC_GRAMMAR_VERSION = "doc-chunker-v1"

DOC_MARKDOWN_EXTS = {"md", "markdown", "rst"}
DOC_JSON_EXTS = {"json", "jsonl", "ndjson"}
DOC_YAML_EXTS = {"yaml", "yml"}
DOC_TOML_EXTS = {"toml"}
DOC_CSV_EXTS = {"csv"}
DOC_TSV_EXTS = {"tsv"}
