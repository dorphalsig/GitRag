import os
import subprocess
import pytest
from unittest.mock import MagicMock, patch
from Indexer import (
    _collect_changes,
    _resolve_range,
    _filter_text_files,
    _run_git,
    ProcessFilesResult
)

def test_process_files_result_equality():
    res = ProcessFilesResult(5, ["failed.txt"])
    assert res == 5
    assert res[0] == 5
    assert res[1] == ["failed.txt"]

def test_resolve_range_initial(temp_git_repo, monkeypatch):
    monkeypatch.chdir(temp_git_repo)
    # HEAD has no parent, so it should fallback to empty tree
    from_sha, to_sha = _resolve_range()
    assert to_sha == "HEAD"
    # Empty tree hash for /dev/null
    assert len(from_sha) == 40

def test_resolve_range_delta(temp_git_repo, monkeypatch):
    monkeypatch.chdir(temp_git_repo)
    # Add a second commit
    (temp_git_repo / "file.txt").write_text("content")
    subprocess.run(["git", "add", "file.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "second"], check=True)

    from_sha, to_sha = _resolve_range()
    assert from_sha == "HEAD^"
    assert to_sha == "HEAD"

def test_collect_changes_add_delete_rename(temp_git_repo, monkeypatch):
    monkeypatch.chdir(temp_git_repo)
    # 1. Setup initial state
    (temp_git_repo / "to_delete.txt").write_text("delete me")
    (temp_git_repo / "to_rename.txt").write_text("rename me")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "base"], check=True)
    base_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    # 2. Perform changes
    (temp_git_repo / "new_file.txt").write_text("new")
    os.remove(temp_git_repo / "to_delete.txt")
    os.rename(temp_git_repo / "to_rename.txt", temp_git_repo / "renamed.txt")

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "add", "-u"], check=True)
    subprocess.run(["git", "commit", "-m", "changes"], check=True)
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    to_proc, to_del, actions = _collect_changes((base_sha, head_sha))

    assert "new_file.txt" in to_proc
    assert "renamed.txt" in to_proc
    assert "to_delete.txt" in to_del
    # rename results in old path deletion and new path process
    assert "to_rename.txt" in to_del

def test_filter_text_files(tmp_path):
    # Create some dummy files
    text_file = tmp_path / "text.txt"
    text_file.write_text("this is text")

    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff")

    # We mock BinaryDetector since it uses Magika
    mock_detector = MagicMock()
    def mock_is_binary(p):
        return p.endswith(".bin")
    mock_detector.is_binary.side_effect = mock_is_binary

    paths = {str(text_file), str(binary_file)}
    filtered = _filter_text_files(paths, detector=mock_detector)

    assert str(text_file) in filtered
    assert str(binary_file) not in filtered

def test_run_git_failure():
    with pytest.raises(RuntimeError, match="git invalid-command failed"):
        _run_git(["invalid-command"])

def test_collect_full_repo(temp_git_repo, monkeypatch):
    monkeypatch.chdir(temp_git_repo)
    # 1. Setup tracked and untracked files
    (temp_git_repo / "tracked.txt").write_text("tracked")
    subprocess.run(["git", "add", "tracked.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "add tracked"], check=True)

    (temp_git_repo / "untracked.txt").write_text("untracked")
    (temp_git_repo / "ignored.txt").write_text("ignored")
    (temp_git_repo / ".gitignore").write_text("ignored.txt")

    from Indexer import _collect_full_repo
    # Mock detector to accept everything
    mock_detector = MagicMock()
    mock_detector.is_binary.return_value = False

    text, skipped, actions = _collect_full_repo(mock_detector)

    assert "tracked.txt" in text
    assert "untracked.txt" in text
    assert "ignored.txt" not in text
    assert "README.md" in text  # Created in fixture
