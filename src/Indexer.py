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

from Persist import PersistConfig, PersistInVectorize
from Calculators.CodeRankCalculator import CodeRankCalculator
import Chunker

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



def _filter_text_files(paths: Set[str], rng: Tuple[str, str]) -> Set[str]:
    """Filter a candidate path set down to text files only.

    Inputs:
        paths: candidate paths to consider for processing.
        rng: (from_ref, to_ref) used to query file stats for the same change range.

    Behavior:
        - Runs: git diff --numstat --find-renames -z <from>..<to>
        - For each NUL-terminated record:
            * Normal shape: "added<TAB>deleted<TAB>path"
            * Rename shape: "added<TAB>deleted<TAB>old" (this token) then "new" (next token)
          Binaries show '-' counts; only digit counts are treated as text.
        - Classifies exactly the path present in `paths`:
            * normal → the path field
            * rename → the *new* path
        - Any remaining candidates fall back to: git check-attr binary -- <path>
          (skip if 'binary: set').

    Outputs:
        A set with only text files (safe to chunk/encode).

    Exceptions:
        Propagates RuntimeError from _run_git(...) if git fails; attribute checks are best-effort.
    """
    if not paths:
        return set()

    frm, to = rng
    raw = _run_git(["diff", "--numstat", "--find-renames", "-z", f"{frm}..{to}"])
    tokens = [t for t in raw.split("\x00") if t]

    text: Set[str] = set()
    i = 0
    while i < len(tokens):
        parts = tokens[i].split("\t")
        if len(parts) < 3:
            logger.debug("Skipping malformed numstat record: %r", tokens[i])
            i += 1
            continue

        added, deleted, path_or_old = parts[0], parts[1], parts[2]
        counts_are_digits = added.isdigit() and deleted.isdigit()

        # Normal case: a,d,path in this token
        if path_or_old in paths:
            if counts_are_digits:
                text.add(path_or_old)
            i += 1
            continue

        # Rename case: a,d,old in this token; next token is new Path
        if i + 1 < len(tokens):
            new_path = tokens[i + 1]
            if new_path in paths and counts_are_digits:
                text.add(new_path)
            i += 2
        else:
            i += 1

    # Fallback attribute check for any remaining candidates
    for p in sorted(paths - text):
        try:
            out = _run_git(["check-attr", "binary", "--", p]).strip()
            if not out.endswith(": set"):
                text.add(p)
        except Exception:
            logger.debug("check-attr failed for %s; leaving excluded as conservative default", p)

    return text





def _load_components(repo: str) -> Tuple[CodeRankCalculator, PersistInVectorize]:
    """Initialize the embedding calculator and persistence layer.

    Env:
        CF_ACCOUNT_ID, CF_VECTORIZE_INDEX, CF_D1_DATABASE_ID

    Raises:
        RuntimeError: when any required env var is missing.
    """
    cfg = PersistConfig(
        account_id=(os.environ.get("CF_ACCOUNT_ID") or "").strip(),
        vectorize_index=(os.environ.get("CF_VECTORIZE_INDEX") or "").strip(),
        d1_database_id=(os.environ.get("CF_D1_DATABASE_ID") or "").strip(),
    )
    if not (cfg.account_id and cfg.vectorize_index and cfg.d1_database_id):
        raise RuntimeError("Missing Cloudflare env: CF_ACCOUNT_ID, CF_VECTORIZE_INDEX, CF_D1_DATABASE_ID")

    calc = CodeRankCalculator()
    persist = PersistInVectorize(client=None, cfg=cfg, dim=calc.dimensions)
    logger.info("Initialized components for repo=%s (dim=%d)", repo, calc.dimensions)
    return calc, persist


def _process_files(paths: Iterable[str], repo: str, calc: CodeRankCalculator, persist: PersistInVectorize) -> int:
    """Chunk, embed, and persist all given text files.

    Returns:
        Total number of chunks persisted. Per-file failures are logged and skipped.
    """
    total = 0
    batch: List = []
    for p in paths:
        try:
            chunks = Chunker.chunk_file(p, repo)
            for c in chunks:
                c.calculate_embeddings(calc)
            batch.extend(chunks)
        except Exception as e:
            logger.error("Failed processing %s: %s", p, e)
    if batch:
        try:
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
    parser = argparse.ArgumentParser(description="Process last-commit changes for a repository.")
    parser.add_argument("repo", help="Repository identifier (e.g., namespace/repo)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rng = _resolve_range()
    to_proc, to_del, actions = _collect_changes(rng)
    text_to_proc = _filter_text_files(to_proc, rng)

    calc, persist = _load_components(args.repo)

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
        "processed_files": len(text_to_proc),
        "deleted_files": deleted_count,
        "processed_chunks": processed_chunks,
        "skipped_binary": sorted(list(to_proc - text_to_proc)),
        "actions": actions,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
