"""Centralized project constants."""

# Retriever constants
RETRIEVAL_QUERY_PREFIX = 'QUERY_INSTRUCTION = "Given a code search query, retrieve relevant code snippets and documentation"\nQuery:'
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANK_TASK_INSTRUCTION = "Given a user query and a code/document chunk, output a relevance score for retrieval reranking."
DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_INITIAL_RETRIEVAL_LIMIT = 50
DEFAULT_TOP_K = 10

# Persistence constants
DEFAULT_TABLE_NAME = "chunks"
DEFAULT_FTS_TABLE_SUFFIX = "_fts"
HYBRID_SEARCH_VECTOR_WEIGHT = 0.7
HYBRID_SEARCH_KEYWORD_WEIGHT = 0.3
POSTGRES_FTS_LANGUAGE = "english"
DEFAULT_DB_PROVIDER = "libsql"

# Chunker constants
SOFT_MAX_BYTES = 16_384  # packing target
HARD_CAP_BYTES = 24_576  # absolute per-chunk limit
NEWLINE_WINDOW = 2_048  # cut nudge window
FALLBACK_OVERLAP_RATIO = 0.10

# Document Chunker constants
DOC_SOFT_MAX_BYTES = 8_192
DOC_HARD_CAP_BYTES = 16_384
DOC_OVERLAP_BYTES = 256
DOC_MIN_CHUNK_BYTES = 2_048
DOC_GRAMMAR_VERSION = "doc-chunker-v1"

DOC_MARKDOWN_EXTS = {"md", "markdown", "rst"}
DOC_JSON_EXTS = {"json", "jsonl", "ndjson"}
DOC_YAML_EXTS = {"yaml", "yml"}
DOC_TOML_EXTS = {"toml"}
DOC_CSV_EXTS = {"csv"}
DOC_TSV_EXTS = {"tsv"}

# Indexer constants
INDEXER_FILE_BATCH_SIZE = 64

# Embedding constants
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 32
