from __future__ import annotations

import sys
from array import array
from unittest import mock

import pytest

from Chunk import Chunk
import Indexer


class _Calc:
    def calculate(self, _: str) -> bytes:
        return array("f", [1.0, 0.0]).tobytes()


class _Persist:
    def __init__(self) -> None:
        self.batches: list[list[Chunk]] = []

    def persist_batch(self, chunks):
        self.batches.append(list(chunks))


def test_process_files_forwards_branch_to_chunker_and_persisted_chunks(monkeypatch) -> None:
    seen: list[tuple[str, str, str | None]] = []

    def fake_chunk_file(path: str, repo: str, branch: str | None = None):
        seen.append((path, repo, branch))
        return [
            Chunk(
                chunk="print('x')",
                repo=repo,
                branch=branch,
                path=path,
                language="python",
                start_rc=(0, 0),
                end_rc=(0, 10),
                start_bytes=0,
                end_bytes=10,
            )
        ]

    monkeypatch.setattr(Indexer.chunker, "chunk_file", fake_chunk_file)

    persist = _Persist()
    total = Indexer._process_files(["src/a.py"], "org/repo", _Calc(), persist, branch="feature-x")

    assert total == 1
    assert seen == [("src/a.py", "org/repo", "feature-x")]
    assert len(persist.batches) == 1
    assert persist.batches[0][0].branch == "feature-x"


def test_main_threads_branch_argument_end_to_end(monkeypatch) -> None:
    persist = _Persist()

    monkeypatch.setattr(Indexer, "_resolve_range", lambda: ("HEAD^", "HEAD"))
    monkeypatch.setattr(
        Indexer,
        "_collect_changes",
        lambda rng: ({"src/a.py"}, set(), [{"action": "process", "path": "src/a.py", "reason": "status=M"}]),
    )
    monkeypatch.setattr(Indexer, "_filter_text_files", lambda paths, detector=None: set(paths))
    monkeypatch.setattr(Indexer, "_load_components", lambda repo: (_Calc(), persist))

    seen_branch: list[str | None] = []

    def fake_chunk_file(path: str, repo: str, branch: str | None = None):
        seen_branch.append(branch)
        return [
            Chunk(
                chunk="hello",
                repo=repo,
                branch=branch,
                path=path,
                language="python",
                start_rc=(0, 0),
                end_rc=(0, 5),
                start_bytes=0,
                end_bytes=5,
            )
        ]

    monkeypatch.setattr(Indexer.chunker, "chunk_file", fake_chunk_file)

    with mock.patch.object(sys, "argv", ["Indexer.py", "org/repo", "--branch", "feature-x"]):
        with mock.patch.object(Indexer.logger, "info") as info_spy:
            Indexer.main()

    assert seen_branch == ["feature-x"]
    assert persist.batches[0][0].branch == "feature-x"
    assert any("branch=%s" in str(call.args[0]) for call in info_spy.call_args_list)


def test_collect_changes_handles_rename_delete_and_modify(monkeypatch) -> None:
    payload = "R100\told.py\x00new.py\x00D\tgone.py\x00M\tkeep.py\x00"
    monkeypatch.setattr(Indexer, "_run_git", lambda args: payload)

    to_process, to_delete, actions = Indexer._collect_changes(("a", "b"))

    assert to_process == {"new.py", "keep.py"}
    assert to_delete == {"old.py", "gone.py"}
    assert {a["action"] for a in actions} == {"process", "delete"}


def test_resolve_range_fallbacks_to_empty_tree(monkeypatch) -> None:
    def fake_run_git(args):
        if args == ["rev-parse", "HEAD^"]:
            raise RuntimeError("no parent")
        if args[:3] == ["hash-object", "-t", "tree"]:
            return "emptytree\n"
        return ""

    monkeypatch.setattr(Indexer, "_run_git", fake_run_git)
    frm, to = Indexer._resolve_range()

    assert frm == "emptytree"
    assert to == "HEAD"


def test_filter_text_files_ignores_binary_and_errors() -> None:
    class Detector:
        def is_binary(self, p: str) -> bool:
            if p == "err.py":
                raise RuntimeError("boom")
            return p.endswith(".bin")

    out = Indexer._filter_text_files({"a.py", "b.bin", "err.py"}, detector=Detector())
    assert out == {"a.py"}


def test_run_git_surfaces_failures(monkeypatch) -> None:
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(Indexer.subprocess, "run", raise_missing)
    with pytest.raises(RuntimeError):
        Indexer._run_git(["status"])

    def raise_called(*args, **kwargs):
        raise Indexer.subprocess.CalledProcessError(1, ["git"], stderr="bad")

    monkeypatch.setattr(Indexer.subprocess, "run", raise_called)
    with pytest.raises(RuntimeError):
        Indexer._run_git(["status"])
