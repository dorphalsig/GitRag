-- GitRag chunk storage schema for Cloudflare D1
-- Apply once during environment provisioning.

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
    mutation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks (repo, path);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks (language);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks (status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id,
    chunk
);
