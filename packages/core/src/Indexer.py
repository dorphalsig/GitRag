#!/usr/bin/env python3
"""
Indexer.py — Simplified and robust orchestrator.

Behavior:
- Uses one git-diff-based selection path for both delta and full runs.
- Lazy iteration for file selection and chunking.
- Multi-batch file buffering: persists only complete files.
- Timeout gate before each embedding batch.
- Returns exit code 75 on timeout.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pathspec
from more_itertools import batched

import text_detection
from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker import chunker
from Chunker.Chunk import Chunk
from Persistence.Persist import DBConfig, LibsqlConfig, create_persistence_adapter
from constants import (
    DEFAULT_DB_PROVIDER, DEFAULT_TABLE_NAME, EMBEDDING_BATCH_SIZE, EXIT_CODE_TIMEOUT, SOFT_TIMEOUT_SECONDS, )

LOG4J_FORMAT = "%(asctime)s %(levelname)-5s %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG4J_FORMAT,
    datefmt=DATE_FORMAT
)
logger = logging.getLogger("indexer")


@dataclass
class IndexingResult:
    processed_chunks: int = 0
    processed_files: int = 0
    deleted_files: int = 0
    failed_paths: List[str] = field(default_factory=list)
    timed_out: bool = False
    skipped_binary: List[str] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)


class Indexer:
    calculator = EmbeddingCalculator()

    def __init__(self, repo: str, branch: str, from_sha: str, to_sha: str, is_full: bool = False, ):
        self.repo = repo
        self.branch = branch
        self.is_full = is_full
        cfg = _resolve_db_cfg()
        self.db = create_persistence_adapter(cfg.provider, cfg=cfg)
        self.sha0, self.sha1 = _resolve_range(from_sha, to_sha, is_full)
        self.start_time = time.time()

    @staticmethod
    def _get_ignore_spec() -> pathspec.PathSpec | None:
        val = (os.environ.get("GITRAG_IGNORE") or "").strip()
        if not val:
            return None
        patterns = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
        if not patterns:
            return None
        return pathspec.PathSpec.from_lines("gitignore", patterns)

    def _check_timeout(self):
        if time.time() - self.start_time > SOFT_TIMEOUT_SECONDS:
            logger.error("Soft timeout reached. Stopping indexing.")
            exit(EXIT_CODE_TIMEOUT)

    def _iter_git_changes(self) -> tuple[set[str], set[str]]:
        """returns a set of processed files and a set of deleted files.
        honors     """
        raw = _run_git(["diff", "--name-status", "--no-renames", "-z", f"{self.sha0}..{self.sha1}"])
        tokens = raw.strip("\x00").split("\x00")
        process = set()
        delete = set()
        already_indexed = self.db.get_indexed_paths(repo=self.repo) if self.is_full else set()
        binary_detector = text_detection.BinaryDetector()
        ignore_spec = self._get_ignore_spec()
        for action, file in batched(tokens, 2):
            if ignore_spec is None or not ignore_spec.match_file(file):
                if action == "D":
                    delete.add(file)
                else:
                    if not binary_detector.is_binary(file):
                        process.add(file)
        process -= already_indexed
        return process, delete

    def index(self):
        to_process, to_delete = self._iter_git_changes()
        if to_delete:
            self.db.delete_batch(to_delete, repo=self.repo)

        chunk_stream: Iterable[Chunk] = itertools.chain.from_iterable(
            (chunker.chunk_file(path, self.repo, self.branch) for path in to_process))

        current_path = None
        accumulated_chunks = []

        for batch in batched(chunk_stream, EMBEDDING_BATCH_SIZE):
            # Extract text and compute embeddings for the max-capacity batch
            self._check_timeout()
            text_chunks = [obj.chunk for obj in batch]
            try:
                embeddings = self.calculator.calculate_batch(text_chunks)
            except Exception as e:
                logger.error("Failed to compute embeddings for batch: %r", e)
                continue

            # 2. Re-align the computed embeddings with their origin objects
            for chunk_obj, embedding in zip(batch, embeddings):
                chunk_obj.embeddings = embedding

                # 3. Path boundary detected
                if current_path and chunk_obj.path != current_path:
                    # Persist all data accumulated for the previous path
                    self.db.persist_batch(accumulated_chunks)

                    # Update the state machine for the new path
                    accumulated_chunks = []

                # Accumulate the embedding for the current path
                current_path = chunk_obj.path
                accumulated_chunks.append(chunk_obj)

            # 4. Stream exhausted. Persist the final path's data.
        if current_path is not None and accumulated_chunks:
            self.db.persist_batch(accumulated_chunks)


# for chunk_obj in all_chunk_objs:
#    self._process_chunk(batch, chunk_obj, current_path, flush_indexes, persistance_queue)


def _run_git(args: List[str]) -> str:
    """Run a git command and return stdout as text."""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotePath=false", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out.stdout
    except FileNotFoundError as e:
        raise RuntimeError("git not found on PATH") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}") from e


def _resolve_range(from_sha: str = "HEAD^", to_sha: str = "HEAD", from_empty: bool = False, ) -> \
        Tuple[str, str]:
    """Return (from_ref, to_ref). Prefer args, fallback to last commit."""
    try:
        _run_git(["rev-parse", "HEAD^"])
        from_sha = _run_git(["rev-parse", "HEAD"]).strip()
    except Exception:
        from_empty = True

    from_sha = from_sha if not from_empty else _run_git(["hash-object", "-t", "tree", "/dev/null"]).strip()
    return from_sha, to_sha


def _resolve_db_cfg() -> DBConfig:
    provider = (os.environ.get("DB_PROVIDER") or DEFAULT_DB_PROVIDER).lower()
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("TURSO_DATABASE_URL")

    if not database_url:
        raise RuntimeError(f"DATABASE_URL is required (provider='{provider}')")

    auth_token = os.environ.get("DB_AUTH_TOKEN") or os.environ.get("TURSO_AUTH_TOKEN")
    if provider == "libsql":
        table = os.environ.get("LIBSQL_TABLE") or DEFAULT_TABLE_NAME
        fts_table = os.environ.get("LIBSQL_FTS_TABLE")
        return LibsqlConfig.from_parts(
            database_url=database_url,
            auth_token=auth_token,
            table=table,
            fts_table=fts_table,
        )

    return DBConfig(provider=provider, url=database_url, auth_token=auth_token, table_map={})


def main() -> None:
    parser = argparse.ArgumentParser(description="Process repository changes for indexing.")
    parser.add_argument("repo", help="Repository identifier (e.g., namespace/repo)")
    parser.add_argument("--full", action="store_true", help="Index all files.")
    parser.add_argument("--branch", default=None, help="Optional branch name.")
    parser.add_argument("--from-sha", default=None, help="Start SHA")
    parser.add_argument("--to-sha", default=None, help="End SHA")
    args = parser.parse_args()

    logger.info("Starting indexer: repo=%s branch=%s mode=%s",
                args.repo, args.branch or "none", "full" if args.full else "delta")
    indexer = Indexer(args.repo, args.branch, args.from_sha, args.to_sha, args.full)
    indexer.index()

if __name__ == "__main__":
    main()