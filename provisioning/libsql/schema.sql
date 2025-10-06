-- GitRag libSQL schema
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  repo TEXT NOT NULL,
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
  embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);
CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks (path);
CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks (repo, path);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  id UNINDEXED,
  chunk
);
