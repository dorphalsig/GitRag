from __future__ import annotations

import sys
from array import array
from unittest import mock

import pytest

import Chunker
from Chunker.Chunk import Chunk
import Indexer


class _Calc:
    @property
    def dimensions(self) -> int:
        return 2

    def calculate(self, _: str) -> bytes:
        return array("f", [1.0, 0.0]).tobytes()

    def calculate_batch(self, texts: list[str]) -> list[bytes]:
        return [self.calculate(t) for t in texts]


def test_run_indexing_forwards_branch_to_chunker_and_persists(monkeypatch) -> None:
    seen: list[tuple[str, str, str | None]] = []

    def fake_chunk_file(path: str, repo: str, branch: str | None = None):
        seen.append((path, repo, branch))
        yield Chunk(
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

    monkeypatch.setattr(Chunker.chunker, "chunk_file", fake_chunk_file)
    monkeypatch.setattr(Indexer, "_resolve_db_cfg", lambda: mock.Mock(provider="libsql"))

    persist = mock.Mock()
    persist.get_indexed_paths.return_value = set()
    monkeypatch.setattr(Indexer, "create_persistence_adapter", lambda *args, **kwargs: persist)
    monkeypatch.setattr(Indexer, "EmbeddingCalculator", _Calc)
    monkeypatch.setattr(Indexer, "BinaryDetector", lambda: mock.Mock(is_binary=lambda _: False))
    monkeypatch.setattr(
        Indexer,
        "_iter_selected_paths",
        lambda *args, **kwargs: iter([{"action": "process", "path": "src/a.py", "reason": "status=M"}]),
    )

    res = Indexer._run_indexing("org/repo", ("A", "B"), False, branch="feature-x")

    assert res.processed_chunks == 1
    assert res.processed_files == 1
    assert seen == [("src/a.py", "org/repo", "feature-x")]
    assert persist.persist_batch.call_count == 1
    assert persist.persist_batch.call_args.args[0][0].branch == "feature-x"


def test_main_threads_branch_argument_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(Indexer, "_resolve_range", lambda *a, **kw: ("HEAD^", "HEAD"))

    seen_branch: list[str | None] = []
    fake_res = Indexer.IndexingResult(processed_files=1, processed_chunks=1)

    def fake_run_indexing(repo: str, rng, is_full: bool, branch: str | None = None, start_time=None):
        seen_branch.append(branch)
        return fake_res

    monkeypatch.setattr(Indexer, "_run_indexing", fake_run_indexing)

    with mock.patch.object(sys, "argv", ["Indexer.py", "org/repo", "--branch", "feature-x"]):
        with mock.patch.object(Indexer.logger, "info") as info_spy:
            Indexer.main()

    assert seen_branch == ["feature-x"]
    assert any("branch=%s" in str(call.args[0]) for call in info_spy.call_args_list)


def test_collect_changes_handles_rename_delete_and_modify(monkeypatch) -> None:
    monkeypatch.setattr(
        Indexer,
        "_iter_git_changes",
        lambda rng: iter([("R100", "old.py", "new.py"), ("D", "gone.py", ""), ("M", "keep.py", "")]),
    )
    detector = mock.Mock(is_binary=lambda _: False)

    actions = list(Indexer._iter_selected_paths(("a", "b"), detector, False, set()))

    assert [a for a in actions if a["action"] == "delete"] == [
        {"action": "delete", "path": "old.py", "reason": "rename/copy"},
        {"action": "delete", "path": "gone.py", "reason": "status=D"},
    ]
    process_paths = {a["path"] for a in actions if a["action"] == "process"}
    assert process_paths == {"new.py", "keep.py"}


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


def test_iter_selected_paths_marks_binary_files_as_skipped() -> None:
    class Detector:
        def is_binary(self, p: str) -> bool:
            return p.endswith(".bin")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        Indexer,
        "_iter_git_changes",
        lambda rng: iter([("M", "a.py", ""), ("A", "b.bin", "")]),
    )
    try:
        actions = list(Indexer._iter_selected_paths(("a", "b"), Detector(), False, set()))
    finally:
        monkeypatch.undo()

    assert {"action": "skip", "path": "b.bin", "reason": "binary"} in actions
    assert {"action": "process", "path": "a.py", "reason": "status=M"} in actions


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
