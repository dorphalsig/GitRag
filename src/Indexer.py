#!/usr/bin/env python3
"""
Indexer.py — lean orchestrator

- Input: single positional argument `repo`
- Detects last-commit changes (handles initial commit)
- Renames => DELETE (old path) + PROCESS (new path)
- Filters binaries via `git --numstat` / gitattributes
- PROCESS: chunk -> embed -> persist
- DELETE: persist.delete_batch(paths)
- No Cloudflare client details leak from here; Persist handles it
- Prints a concise JSON summary to stdout
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from typing import Dict, Iterable, List, Set, Tuple

from Persist import (
    PersistConfig,
    create_persistence_adapter,
    PersistenceAdapter,
)
from Calculators.CodeRankCalculator import CodeRankCalculator
import chunker
from text_detection import BinaryDetector

logger = logging.getLogger("feed")


def _run_git(args: List[str]) -> str:
    """Run a git command and return stdout as text.

    Raises:
        RuntimeError: when `git` is missing or the command fails.

    Notes:
        - We keep this wrapper minimal and deterministic.
        - stderr from the failing command is surfaced in the message.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out.stdout
    except FileNotFoundError as e:  # pragma: no cover
        raise RuntimeError("git not found on PATH") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}") from e


def _resolve_range() -> Tuple[str, str]:
    """Return (from_ref, to_ref) for the *last* commit, handling the initial commit case."""
    try:
        _run_git(["rev-parse", "HEAD^"])
        return "HEAD^", "HEAD"
    except Exception:
        empty_tree = _run_git(["hash-object", "-t", "tree", "/dev/null"]).strip()
        return empty_tree, "HEAD"


def _collect_changes(rng: Tuple[str, str]) -> Tuple[Set[str], Set[str], List[Dict[str, str]]]:
    """Summarize file changes in a commit range.

    Inputs:
        rng: (from_ref, to_ref) commit-ish pair to compare (e.g., ("HEAD^", "HEAD")).

    Behavior:
        - Runs: git diff --name-status --find-renames -z <from>..<to>
        - With -z, each record is NUL-terminated; within a record, status and first path are tab-separated.
        - Renames/Copies (R*/C*): record encodes "STATUS<TAB>old", and the next NUL token is "new".
          We treat them as DELETE(old) + PROCESS(new).
        - 'D' → DELETE; everything else → PROCESS.

    Outputs:
        (to_process, to_delete, actions): sets of paths and a flat audit log.

    Exceptions:
        Propagates RuntimeError from _run_git(...) if git is missing or the command fails.
    """
    frm, to = rng
    raw = _run_git(["diff", "--name-status", "--find-renames", "-z", f"{frm}..{to}"])
    tokens = [t for t in raw.split("\x00") if t]

    to_process: Set[str] = set()
    to_delete: Set[str] = set()
    actions: List[Dict[str, str]] = []

    i = 0
    while i < len(tokens):
        rec = tokens[i]
        tab = rec.find("\t")
        if tab < 0:
            logger.debug("Skipping malformed name-status record (no tab): %r", rec)
            i += 1
            continue

        status = rec[:tab]
        path1 = rec[tab + 1 :]

        if status.startswith(("R", "C")):
            if i + 1 < len(tokens):
                old_path, new_path = path1, tokens[i + 1]
                to_delete.add(old_path)
                to_process.add(new_path)
                actions.append({"action": "delete", "path": old_path, "reason": "rename/copy"})
                actions.append({"action": "process", "path": new_path, "reason": "rename/copy"})
                i += 2
            else:
                logger.debug("Rename/copy record missing new path: %r", rec)
                i += 1
            continue

        if status == "D":
            to_delete.add(path1)
            actions.append({"action": "delete", "path": path1, "reason": "status=D"})
        else:
            to_process.add(path1)
            actions.append({"action": "process", "path": path1, "reason": f"status={status}"})
        i += 1

    return to_process, to_delete, actions



def _filter_text_files(paths: Set[str], detector: BinaryDetector | None = None) -> Set[str]:
    """Filter candidate paths down to text files using a shared binary detector."""
    if not paths:
        return set()

    detector = detector or BinaryDetector(_run_git)
    text: Set[str] = set()
    for p in sorted(paths):
        try:
            if not detector.is_binary(p):
                text.add(p)
        except Exception as exc:  # pragma: no cover - defensive log only
            logger.debug("Binary detection failed for %s: %s", p, exc)
    return text





def _collect_full_repo(detector: BinaryDetector) -> Tuple[Set[str], List[str], List[Dict[str, str]]]:
    """Enumerate all repo files (tracked + unignored) and classify text vs binary."""
    tracked = _list_paths(["ls-files", "--cached"])
    others = _list_paths(["ls-files", "--others", "--exclude-standard"])
    candidates = tracked | others
    text = _filter_text_files(candidates, detector=detector)
    skipped = sorted(candidates - text)
    actions = [{"action": "process", "path": p, "reason": "full-index"} for p in sorted(text)]
    return text, skipped, actions


def _list_paths(args: List[str]) -> Set[str]:
    """Return git command output as a set of file paths, ignoring blanks."""
    raw = _run_git(args)
    return {line for line in (s.strip() for s in raw.splitlines()) if line}


def _env_value(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _resolve_cf_ids() -> PersistConfig:
    account_id = _env_value("CLOUDFLARE_ACCOUNT_ID")
    vectorize_index = _env_value("CLOUDFLARE_VECTORIZE_INDEX")
    d1_database_id = _env_value("CLOUDFLARE_D1_DATABASE_ID")
    if not (account_id and vectorize_index and d1_database_id):
        raise RuntimeError(
            "Missing Cloudflare env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID"
        )
    return PersistConfig(
        account_id=account_id,
        vectorize_index=vectorize_index,
        d1_database_id=d1_database_id,
    )


def _load_components(repo: str, adapter_name: str | None = None) -> Tuple[CodeRankCalculator, PersistenceAdapter]:
    """Initialize the embedding calculator and persistence layer.

    Raises:
        RuntimeError: when any required env var is missing.
    """
    cfg = _resolve_cf_ids()

    calc = CodeRankCalculator()
    adapter_key = (adapter_name or os.environ.get("GITRAG_PERSIST_ADAPTER", "cloudflare")).strip() or "cloudflare"
    persist = create_persistence_adapter(adapter_key, cfg=cfg, dim=calc.dimensions)
    logger.info("Initialized components for repo=%s (dim=%d, adapter=%s)", repo, calc.dimensions, adapter_key)
    return calc, persist


def _process_files(paths: Iterable[str], repo: str, calc: CodeRankCalculator, persist: PersistenceAdapter) -> int:
    """Chunk, embed, and persist all given text files.

    Returns:
        Total number of chunks persisted. Per-file failures are logged and skipped.
    """
    total = 0
    batch: List = []
    for p in paths:
        logger.info("Processing file: %s", p)
        try:
            chunks = chunker.chunk_file(p, repo)
            logger.debug("File %s produced %d chunks", p, len(chunks))
            for c in chunks:
                logger.debug(
                    "Calculating embeddings for %s:%d-%d", p, c.start_bytes, c.end_bytes
                )
                c.calculate_embeddings(calc)
            batch.extend(chunks)
        except Exception as e:
            logger.error("Failed processing %s: %s", p, e)
    if batch:
        try:
            logger.info("Persisting batch of %d chunks", len(batch))
            persist.persist_batch(batch)
            total = len(batch)
        except Exception as e:
            logger.error("Persist failed for %d chunks: %s", len(batch), e)
    return total


def main() -> None:
    """CLI entrypoint.

    - Parses single positional `repo`
    - Determines last-commit range
    - Computes to-process / to-delete (with rename handling)
    - Filters binaries out
    - Deletes removed paths; processes text paths
    - Prints JSON summary
    """
    parser = argparse.ArgumentParser(description="Process repository changes for indexing.")
    parser.add_argument("repo", help="Repository identifier (e.g., namespace/repo)")
    parser.add_argument("--full", action="store_true", help="Index all tracked and unignored files (initial sync).")
    parser.add_argument("--adapter", help="Persistence adapter key (default: cloudflare)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    detector = BinaryDetector(_run_git)

    if args.full:
        logger.info("Running in full indexing mode")
        text_to_proc, skipped_binary, actions = _collect_full_repo(detector)
        rng = ("FULL_REBUILD", "HEAD")
        to_del: Set[str] = set()
    else:
        rng = _resolve_range()
        to_proc, to_del, actions = _collect_changes(rng)
        logger.info("Running in delta mode: %d process candidates, %d deletions", len(to_proc), len(to_del))
        text_to_proc = _filter_text_files(to_proc, detector=detector)
        skipped_binary = sorted(list(to_proc - text_to_proc))

    calc, persist = _load_components(args.repo, adapter_name=args.adapter)

    deleted_count = 0
    if to_del:
        try:
            persist.delete_batch(sorted(to_del))
            deleted_count = len(to_del)
        except Exception as e:
            logger.error("Deletion pass failed: %s", e)

    processed_chunks = _process_files(sorted(text_to_proc), args.repo, calc, persist)

    summary = {
        "repo": args.repo,
        "range": {"from": rng[0], "to": rng[1]},
        "mode": "full" if args.full else "delta",
        "processed_files": len(text_to_proc),
        "deleted_files": deleted_count,
        "processed_chunks": processed_chunks,
        "skipped_binary": skipped_binary,
        "actions": actions,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
