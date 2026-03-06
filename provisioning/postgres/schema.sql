-- GitRag PostgreSQL schema
-- Keep embedding dimension in sync with EMBEDDING_DIMENSIONS in
-- packages/core/src/constants.py (currently 768).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  repo TEXT NOT NULL,
  branch TEXT,
  path TEXT NOT NULL,
  language TEXT NOT NULL,
  start_row INTEGER NOT NULL,
  start_col INTEGER NOT NULL,
  end_row INTEGER NOT NULL,
  end_col INTEGER NOT NULL,
  start_bytes INTEGER NOT NULL,
  end_bytes INTEGER NOT NULL,
  chunk TEXT NOT NULL,
  status TEXT NOT NULL,
  mutation_id TEXT,
  embedding vector(768) NOT NULL,
  search_vector tsvector
);

CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);
CREATE INDEX IF NOT EXISTS chunks_repo_branch_idx ON chunks (repo, branch);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks (path);
CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks (repo, path);
CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING GIN (search_vector);
