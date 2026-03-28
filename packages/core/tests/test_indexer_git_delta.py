import sys
from unittest.mock import MagicMock, patch

import pytest
from Indexer import Indexer, main


def _make_indexer(repo_path, sha_a, sha_b, mock_db):
    """Instantiate Indexer with mocked persistence and known SHAs."""
    mock_cfg = MagicMock(provider="mock")
    with patch("Indexer._resolve_db_cfg", return_value=mock_cfg), \
         patch("Indexer.create_persistence_adapter", return_value=mock_db), \
         patch("Indexer._resolve_range", return_value=(sha_a, sha_b)):
        return Indexer("test/repo", "main", sha_a, sha_b)


def test_added_file_in_process_set(two_commit_repo, mock_db):
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "new.py" in to_process


def test_deleted_file_in_delete_set(two_commit_repo, mock_db):
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "delete.py" in to_delete


def test_rename_appears_as_delete_and_add(two_commit_repo, mock_db):
    # _iter_git_changes uses --no-renames: rename = D old + A new
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "to_rename.py" in to_delete
    assert "renamed.py" in to_process


def test_unmodified_file_excluded(two_commit_repo, mock_db):
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "keep.py" not in to_process
    assert "keep.py" not in to_delete


def test_skip_list_comma_separated(two_commit_repo, mock_db, monkeypatch):
    repo_path, sha_a, sha_b = two_commit_repo
    monkeypatch.setenv("GITRAG_IGNORE", "*.log,*.bin")
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "skip.log" not in to_process
    assert "skip.log" not in to_delete


def test_skip_list_semicolon_separator(two_commit_repo, mock_db, monkeypatch):
    repo_path, sha_a, sha_b = two_commit_repo
    monkeypatch.setenv("GITRAG_IGNORE", "*.log;*.bin")
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "skip.log" not in to_process
    assert "skip.log" not in to_delete


def test_binary_file_excluded(two_commit_repo, mock_db):
    # image.bin has a PNG header — magika identifies it as binary
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    to_process, to_delete = indexer._iter_git_changes()
    assert "image.bin" not in to_process


def test_index_continues_after_embedding_failure(two_commit_repo, mock_db):
    """index() logs and skips a batch when calculate_batch raises, then persists remaining."""
    repo_path, sha_a, sha_b = two_commit_repo
    mock_calc = MagicMock()
    mock_calc.calculate_batch.side_effect = RuntimeError("model exploded")

    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    indexer.calculator = mock_calc

    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 9999), \
         patch("Indexer.chunker.chunk_file") as mock_chunk:
        from Chunker.Chunk import Chunk
        mock_chunk.side_effect = lambda path, repo, branch: [
            Chunk(chunk="c", repo=repo, path=path, language="python",
                  start_rc=(0, 0), end_rc=(1, 0), start_bytes=0, end_bytes=1)
        ]
        indexer.index()  # must not raise


def test_index_pipeline_end_to_end(two_commit_repo, mock_db):
    """Smoke test: index() runs without error on a real two-commit delta."""
    repo_path, sha_a, sha_b = two_commit_repo
    mock_calc = MagicMock()
    mock_calc.calculate_batch.side_effect = lambda texts: [b"0" * 4096 for _ in texts]

    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    indexer.calculator = mock_calc

    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 9999), \
         patch("Indexer.chunker.chunk_file") as mock_chunk:
        from Chunker.Chunk import Chunk
        mock_chunk.side_effect = lambda path, repo, branch: [
            Chunk(
                chunk="content",
                repo=repo,
                path=path,
                language="python",
                start_rc=(0, 0),
                end_rc=(1, 0),
                start_bytes=0,
                end_bytes=7,
            )
        ]
        indexer.index()

    assert mock_db.persist_batch.called or mock_db.delete_batch.called


def test_main_constructs_indexer_and_calls_index(two_commit_repo, monkeypatch, mock_db):
    """main() parses argv, constructs Indexer, and calls index()."""
    repo_path, sha_a, sha_b = two_commit_repo
    monkeypatch.setattr(sys, "argv", ["indexer", "test/repo", "--from-sha", sha_a, "--to-sha", sha_b])

    mock_indexer = MagicMock()
    mock_cfg = MagicMock(provider="mock")
    with patch("Indexer._resolve_db_cfg", return_value=mock_cfg), \
         patch("Indexer.create_persistence_adapter", return_value=mock_db), \
         patch("Indexer._resolve_range", return_value=(sha_a, sha_b)), \
         patch.object(Indexer, "index", mock_indexer.index):
        main()

    mock_indexer.index.assert_called_once()
