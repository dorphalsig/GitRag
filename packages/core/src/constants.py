"""Centralized project constants."""
import math
import os

# Embedding constants
EMBEDDING_MODEL_ID = os.getenv("model", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIMENSIONS = int(os.getenv("dimensions", 1_024))
EMBEDDING_BATCH_SIZE = int(os.getenv("batch_size", 64))
MAX_SEQ_LENGTH =  int(os.getenv("max_seq_length", 1_024))

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
SOFT_MAX_BYTES = MAX_SEQ_LENGTH
HARD_CAP_BYTES = MAX_SEQ_LENGTH * 2  # this is the max size chunk for code. 2 bytes / token
NEWLINE_WINDOW = 2_048  # cut nudge window
FALLBACK_OVERLAP_RATIO = 0.10

# Document Chunker constants
DOC_SOFT_MAX_BYTES = MAX_SEQ_LENGTH * 2
DOC_HARD_CAP_BYTES = MAX_SEQ_LENGTH * 4  # max size of text chunk: 4bytes / token
DOC_OVERLAP_BYTES = 256
DOC_MIN_CHUNK_BYTES = 2_048
DOC_GRAMMAR_VERSION = "doc-chunker-v1"

DOC_MARKDOWN_EXTS = {"md", "markdown", "rst"}
DOC_JSON_EXTS = {"json", "jsonl", "ndjson"}
DOC_YAML_EXTS = {"yaml", "yml"}
DOC_TOML_EXTS = {"toml"}
DOC_CSV_EXTS = {"csv"}
DOC_TSV_EXTS = {"tsv"}
