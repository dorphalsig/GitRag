import argparse
import itertools
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import more_itertools
import pathspec
from Calculators.EmbeddingCalculator import EmbeddingCalculator
from Chunker import chunker
from Chunker.Chunk import Chunk
from Persistence.Persist import (DBConfig, LibsqlConfig, PersistenceAdapter,
                                 create_persistence_adapter)
from constants import (EMBEDDING_BATCH_SIZE, EXIT_CODE_TIMEOUT,
                       SOFT_TIMEOUT_SECONDS)
import text_detection

LOG4J_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    error_reports: List[Dict[str, object]] = field(default_factory=list)


class Indexer:
    calculator = EmbeddingCalculator()

    def __init__(
        self,
        repo: str,
        branch: str,
        from_sha: str,
        to_sha: str,
        is_full: bool = False,
    ):
        self.repo = repo
        self.branch = branch
        self.is_full = is_full
        cfg = _resolve_db_cfg()
        self.db = create_persistence_adapter(cfg.provider, cfg=cfg)
        self.sha0, self.sha1 = self._resolve_range(from_sha, to_sha, is_full)
        self.start_time = time.time()

    def _check_timeout(self):
        if SOFT_TIMEOUT_SECONDS > 0 and (
            time.time() - self.start_time > SOFT_TIMEOUT_SECONDS
        ):
            logger.error("Indexing exceeded soft timeout of %ds", SOFT_TIMEOUT_SECONDS)
            sys.exit(EXIT_CODE_TIMEOUT)

    @staticmethod
    def _get_ignore_spec() -> pathspec.PathSpec:
        ignore_env = os.environ.get("GITRAG_IGNORE", "")
        # Standardize separators
        raw = [p.strip() for p in ignore_env.replace(";", ",").split(",") if p.strip()]
        return pathspec.PathSpec.from_lines("gitignore", raw)

    def _iter_git_changes(self) -> Tuple[List[str], List[str]]:
        to_process = []
        to_delete = []

        cmd = ["diff", "--name-status", "--no-renames", "-z", f"{self.sha0}..{self.sha1}"]
        stdout = self._run_git(cmd)

        ignore_spec = self._get_ignore_spec()
        detector = text_detection.BinaryDetector()

        tokens = stdout.split("\0")
        i = 0
        while i < len(tokens) - 1:
            status = tokens[i]
            if not status:
                i += 1
                continue
            path = tokens[i + 1]
            i += 2

            if ignore_spec.match_file(path):
                continue

            if status == "D":
                to_delete.append(path)
            elif status in ["A", "M"]:
                if detector.is_binary(path):
                    logger.info("Skipping binary file: %s", path)
                    continue
                to_process.append(path)

        if self.is_full:
            indexed = self.db.get_indexed_paths(repo=self.repo)
            # Subtract already-indexed paths from the set of additions/modifications
            to_process = list(set(to_process) - indexed)

        return to_process, to_delete

    def index(self):
        result = IndexingResult()
        to_process, to_delete = self._iter_git_changes()
        if to_delete:
            self.db.delete_batch(to_delete, repo=self.repo)
            result.deleted_files += len(to_delete)

        chunk_stream: Iterable[Chunk] = itertools.chain.from_iterable(
            (chunker.chunk_file(path, self.repo, self.branch) for path in to_process)
        )

        current_path = None
        accumulated_chunks = []

        for batch in more_itertools.batched(chunk_stream, EMBEDDING_BATCH_SIZE):
            self._check_timeout()
            text_chunks = [obj.chunk for obj in batch]
            try:
                embeddings = self.calculator.calculate_batch(text_chunks)
            except Exception as e:
                logger.error("Failed to compute embeddings for batch: %r", e)
                for chunk_obj in batch:
                    result.error_reports.append(
                        {
                            "message": str(e),
                            "path": chunk_obj.path,
                            "start_rc": chunk_obj.start_rc,
                            "end_rc": chunk_obj.end_rc,
                            "signature": chunk_obj.signature,
                        }
                    )
                continue

            for chunk_obj, embedding in zip(batch, embeddings):
                chunk_obj.embeddings = embedding
                if current_path and chunk_obj.path != current_path:
                    self.db.persist_batch(accumulated_chunks)
                    result.processed_files += 1
                    result.processed_chunks += len(accumulated_chunks)
                    accumulated_chunks = []
                current_path = chunk_obj.path
                accumulated_chunks.append(chunk_obj)

        if current_path is not None and accumulated_chunks:
            self.db.persist_batch(accumulated_chunks)
            result.processed_files += 1
            result.processed_chunks += len(accumulated_chunks)

        for entry in result.error_reports:
            logger.error(
                "Indexing error: %s | %s | %s -> %s | %s",
                entry["message"],
                entry["path"],
                entry["start_rc"],
                entry["end_rc"],
                entry["signature"] or "",
            )

        return result


    def _run_git(self, args: List[str]) -> str:
        cmd = ["git", "-c", "core.quotePath=false", *args]
        logger.debug("Running: %s", " ".join(map(shlex.quote, cmd)))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Git failed: {res.stderr}")
        return res.stdout


    def _resolve_range(
        self, from_sha: Optional[str], to_sha: Optional[str], is_full: bool
    ) -> Tuple[str, str]:
        if to_sha is None:
            to_sha = self._run_git(["rev-parse", "HEAD"]).strip()

        if is_full:
            # Diff from the empty tree to include everything in the current tree
            from_sha = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        elif from_sha is None:
            try:
                from_sha = self._run_git(["rev-parse", "HEAD^"]).strip()
            except RuntimeError:
                # Single-commit repo; fall back to empty tree
                from_sha = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

        return from_sha, to_sha


def _resolve_db_cfg():
    url = os.environ.get("DATABASE_URL") or os.environ.get("TURSO_DATABASE_URL")
    if not url:
        return DBConfig(provider="postgres", url="postgresql://postgres:postgres@localhost:5432/gitrag")

    if url.startswith("libsql://") or "turso.io" in url:
        return LibsqlConfig.from_parts(database_url=url)
    return DBConfig(provider="postgres", url=url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--from-sha")
    parser.add_argument("--to-sha")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    indexer = Indexer(args.repo, args.branch, args.from_sha, args.to_sha, args.full)
    indexer.index()


if __name__ == "__main__":
    main()
