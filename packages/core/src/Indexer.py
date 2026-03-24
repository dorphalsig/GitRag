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
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Set, Tuple, Any

import pathspec

from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker import chunker
from Chunker.Chunk import Chunk
from Persistence.Persist import DBConfig, LibsqlConfig, create_persistence_adapter
from constants import (
    DEFAULT_DB_PROVIDER,
    DEFAULT_TABLE_NAME,
    EMBEDDING_BATCH_SIZE,
    SOFT_TIMEOUT_SECONDS,
    EXIT_CODE_TIMEOUT,
)
from text_detection import BinaryDetector

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


def _resolve_range(from_sha: str | None = None, to_sha: str | None = None) -> Tuple[str, str]:
    """Return (from_ref, to_ref). Prefer args, fallback to last commit."""
    if from_sha and to_sha:
        return from_sha, to_sha

    # Fallback to checking just the last commit
    try:
        _run_git(["rev-parse", "HEAD^"])
        return "HEAD^", "HEAD"
    except Exception:
        empty_tree = _run_git(["hash-object", "-t", "tree", "/dev/null"]).strip()
        return empty_tree, "HEAD"


def _iter_git_changes(rng: Tuple[str, str]) -> Iterator[Tuple[str, str, str]]:
    """Lazily parse git diff for changes. Yields (status, old_path, new_path)."""
    frm, to = rng
    raw = _run_git(["diff", "--name-status", "--find-renames", "-z", f"{frm}..{to}"])
    tokens = raw.split("\x00")

    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue

        if status.startswith(("R", "C")):
            if i + 2 < len(tokens):
                yield status, tokens[i + 1], tokens[i + 2]
                i += 3
            else:
                i += 1
        else:
            if i + 1 < len(tokens):
                yield status, tokens[i + 1], ""
                i += 2
            else:
                i += 1


def _get_ignore_spec() -> pathspec.PathSpec | None:
    val = (os.environ.get("GITRAG_IGNORE") or "").strip()
    if not val:
        return None
    patterns = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def _iter_selected_paths(
    rng: Tuple[str, str],
    detector: BinaryDetector,
    is_full: bool,
    already_indexed: Set[str],
) -> Iterator[Dict[str, Any]]:
    """Lazily yield file selection actions (process/delete) with filtering applied."""
    spec = _get_ignore_spec()

    for status, p1, p2 in _iter_git_changes(rng):
        # Handle rename/copy: p1 is old, p2 is new
        if status.startswith(("R", "C")):
            # Delete old path
            if spec is None or not spec.match_file(p1):
                yield {"action": "delete", "path": p1, "reason": "rename/copy"}

            # Process new path
            path = p2
            reason = "rename/copy"
        else:
            path = p1
            if status == "D":
                if spec is None or not spec.match_file(path):
                    yield {"action": "delete", "path": path, "reason": "status=D"}
                continue
            reason = f"status={status}"

        # Filtering for process actions
        if spec and spec.match_file(path):
            continue

        if detector.is_binary(path):
            yield {"action": "skip", "path": path, "reason": "binary"}
            continue

        if is_full and path in already_indexed:
            yield {"action": "skip", "path": path, "reason": "already-indexed"}
            continue

        yield {"action": "process", "path": path, "reason": reason}


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


def _run_indexing(
    repo: str,
    rng: Tuple[str, str],
    is_full: bool,
    branch: str | None = None,
    start_time: float | None = None,
) -> IndexingResult:
    """Orchestrate the indexing pipeline."""
    calc = EmbeddingCalculator()
    cfg = _resolve_db_cfg()
    persist = create_persistence_adapter(cfg.provider, cfg=cfg, dim=calc.dimensions)
    detector = BinaryDetector()

    already_indexed = persist.get_indexed_paths(repo=repo) if is_full else set()
    res = IndexingResult()

    # Selection and processing pipeline
    embedding_queue: List[Chunk] = []
    # file_buffers maps path -> {chunk_count, embedded_count, chunking_done, chunks}
    file_buffers: Dict[str, Dict[str, Any]] = {}

    def _flush_embedding_batch(batch: List[Chunk]) -> int:
        if not batch:
            return 0
        logger.info("Embedding batch: %d chunks", len(batch))
        texts = [c.chunk for c in batch]
        try:
            embeddings = calc.calculate_batch(texts)
            for c, emb in zip(batch, embeddings):
                object.__setattr__(c, "embeddings", emb)

            # Identify which files are now complete
            completed_files: List[str] = []
            for c in batch:
                buf = file_buffers.get(c.path)
                if not buf:
                    continue
                buf["embedded_count"] += 1
                if buf["chunking_done"] and buf["embedded_count"] == buf["chunk_count"]:
                    completed_files.append(c.path)

            # Persist completed files
            for path in completed_files:
                buf = file_buffers.get(path)
                if not buf:
                    continue
                persist.persist_batch(buf["chunks"])
                res.processed_files += 1
                del file_buffers[path]

            return len(batch)
        except Exception:
            logger.exception("Batch failed")
            for c in batch:
                res.failed_paths.append(c.path)
                if c.path in file_buffers:
                    del file_buffers[c.path]
            return 0

    # 1. Selection & Deletion pass
    selection_iter = _iter_selected_paths(rng, detector, is_full, already_indexed)
    to_process: List[str] = []

    for item in selection_iter:
        path = item["path"]
        action = item["action"]
        res.actions.append(item)

        if action == "delete":
            try:
                persist.delete_batch([path], repo=repo)
                res.deleted_files += 1
            except Exception:
                logger.exception("Failed to delete %s", path)
                res.failed_paths.append(path)
        elif action == "process":
            to_process.append(path)
        elif action == "skip" and item["reason"] == "binary":
            res.skipped_binary.append(path)

    # 2. Lazy chunking and embedding loop
    for path in to_process:
        # Timeout gate before potentially launching a batch or starting a new file
        if start_time and SOFT_TIMEOUT_SECONDS > 0:
            if time.time() - start_time >= SOFT_TIMEOUT_SECONDS:
                res.timed_out = True
                break

        try:
            file_buffers[path] = {
                "chunk_count": 0,
                "embedded_count": 0,
                "chunking_done": False,
                "chunks": []
            }

            for c in chunker.chunk_file(path, repo, branch=branch):
                file_buffers[path]["chunk_count"] += 1
                file_buffers[path]["chunks"].append(c)
                embedding_queue.append(c)
                if len(embedding_queue) >= EMBEDDING_BATCH_SIZE:
                    # Timeout gate immediately before launch
                    if start_time and SOFT_TIMEOUT_SECONDS > 0:
                        if time.time() - start_time >= SOFT_TIMEOUT_SECONDS:
                            res.timed_out = True
                            return res # Stop immediately, do not launch this batch

                    res.processed_chunks += _flush_embedding_batch(embedding_queue)
                    embedding_queue = []

            buf = file_buffers[path]
            buf["chunking_done"] = True
            if buf["chunk_count"] == 0:
                # Still count as processed if it has no chunks (e.g. empty file)
                res.processed_files += 1
                del file_buffers[path]
            elif buf["embedded_count"] == buf["chunk_count"]:
                persist.persist_batch(buf["chunks"])
                res.processed_files += 1
                del file_buffers[path]

        except Exception:
            logger.exception("Failed to chunk %s", path)
            res.failed_paths.append(path)
            # Drop partially collected chunks for this file; never persist partial files.
            if path in file_buffers:
                del file_buffers[path]
            embedding_queue = [c for c in embedding_queue if c.path != path]

    # Final flush
    if not res.timed_out and embedding_queue:
        res.processed_chunks += _flush_embedding_batch(embedding_queue)

    return res


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

    start_time = time.time()

    try:
        if args.full:
            # Full run starts from empty tree to HEAD
            empty_tree = _run_git(["hash-object", "-t", "tree", "/dev/null"]).strip()
            rng = (empty_tree, "HEAD")
        else:
            rng = _resolve_range(args.from_sha, args.to_sha)

        res = _run_indexing(args.repo, rng, args.full, branch=args.branch, start_time=start_time)

        summary = {
            "repo": args.repo,
            "range": {"from": rng[0], "to": rng[1]},
            "mode": "full" if args.full else "delta",
            "processed_files": res.processed_files,
            "deleted_files": res.deleted_files,
            "processed_chunks": res.processed_chunks,
            "skipped_binary": res.skipped_binary,
            "actions": res.actions,
            "failed_paths": sorted(list(set(res.failed_paths))),
            "timeout": res.timed_out,
        }
        print(json.dumps(summary, ensure_ascii=False))

        if res.timed_out:
            sys.exit(EXIT_CODE_TIMEOUT)
        if res.failed_paths:
            sys.exit(1)

    except Exception:
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
