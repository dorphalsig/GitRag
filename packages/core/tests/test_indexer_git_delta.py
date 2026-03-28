import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
from Indexer import Indexer, main


def _make_indexer(repo_path, sha_a, sha_b, mock_db):
    """Instantiate Indexer with mocked persistence and known SHAs."""
    mock_cfg = MagicMock(provider="mock")
    with patch("Indexer._resolve_db_cfg", return_value=mock_cfg), \
         patch("Indexer.create_persistence_adapter", return_value=mock_db), \
         patch.object(Indexer, "_resolve_range", return_value=(sha_a, sha_b)):
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
         patch.object(Indexer, "_resolve_range", return_value=(sha_a, sha_b)), \
         patch.object(Indexer, "index", mock_indexer.index):
        main()

    mock_indexer.index.assert_called_once()


def test_resolve_range_defaults_to_head_pair_when_shas_omitted(two_commit_repo, mock_db):
    """Delta indexing should resolve omitted SHAs to HEAD^..HEAD."""
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)

    resolved_from, resolved_to = indexer._resolve_range(None, None, False)

    assert resolved_from == sha_a
    assert resolved_to == sha_b


def test_resolve_range_preserves_explicit_shas(two_commit_repo, mock_db):
    """Explicit CLI SHA arguments must not be overwritten by fallback logic."""
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)

    resolved_from, resolved_to = indexer._resolve_range(sha_a, sha_b, False)

    assert resolved_from == sha_a
    assert resolved_to == sha_b


def test_resolve_range_uses_empty_tree_when_full_index_requested(two_commit_repo, mock_db):
    """Full indexing should diff from the empty tree to the resolved target SHA."""
    repo_path, sha_a, sha_b = two_commit_repo
    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)

    resolved_from, resolved_to = indexer._resolve_range(None, None, True)

    assert resolved_from == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    assert resolved_to == sha_b


def test_resolve_range_uses_empty_tree_for_single_commit_repo(tmp_path, monkeypatch, mock_db):
    """Delta indexing on a single-commit repo should fall back to the empty tree."""
    repo = tmp_path / "single_commit_repo"
    repo.mkdir()

    def git(*args):
        import subprocess

        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "T")
    (repo / "only.py").write_text("print('hello')\n")
    git("add", "only.py")
    git("commit", "-m", "initial")
    head_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.chdir(repo)

    indexer = _make_indexer(str(repo), None, None, mock_db)
    resolved_from, resolved_to = indexer._resolve_range(None, None, False)

    assert resolved_from == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    assert resolved_to == head_sha


def test_full_index_excludes_already_indexed_paths(two_commit_repo, mock_db):
    """Full indexing should subtract already-indexed paths from the process set."""
    repo_path, sha_a, sha_b = two_commit_repo
    mock_db.get_indexed_paths.return_value = {"new.py", "renamed.py"}
    mock_cfg = MagicMock(provider="mock")
    with patch("Indexer._resolve_db_cfg", return_value=mock_cfg), \
         patch("Indexer.create_persistence_adapter", return_value=mock_db), \
         patch.object(Indexer, "_resolve_range", return_value=(sha_a, sha_b)):
        indexer = Indexer("test/repo", "main", sha_a, sha_b, True)

    to_process, to_delete = indexer._iter_git_changes()

    assert "new.py" not in to_process
    assert "renamed.py" not in to_process
    assert mock_db.get_indexed_paths.called


from Indexer import IndexingResult


def test_indexing_result_tracks_compact_error_reports():
    result = IndexingResult()

    result.error_reports.append(
        {
            "message": "embedding failed",
            "path": "src/main.py",
            "start_rc": (10, 2),
            "end_rc": (12, 0),
            "signature": "main.func",
        }
    )

    assert result.error_reports == [
        {
            "message": "embedding failed",
            "path": "src/main.py",
            "start_rc": (10, 2),
            "end_rc": (12, 0),
            "signature": "main.func",
        }
    ]


def test_index_reports_embedding_failure_with_compact_chunk_details(two_commit_repo, mock_db):
    repo_path, sha_a, sha_b = two_commit_repo
    mock_calc = MagicMock()
    mock_calc.calculate_batch.side_effect = RuntimeError("model exploded")

    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    indexer.calculator = mock_calc

    from Chunker.Chunk import Chunk

    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 9999), \
         patch("Indexer.chunker.chunk_file") as mock_chunk:
        mock_chunk.side_effect = lambda path, repo, branch: [
            Chunk(
                chunk="c",
                repo=repo,
                path=path,
                language="python",
                start_rc=(3, 1),
                end_rc=(4, 2),
                start_bytes=0,
                end_bytes=1,
                signature="module.fn",
            )
        ]

        result = indexer.index()

    assert result.error_reports
    assert result.error_reports[0] == {
        "message": "model exploded",
        "path": "new.py",
        "start_rc": (3, 1),
        "end_rc": (4, 2),
        "signature": "module.fn",
    }


def test_index_logs_end_of_run_summary_without_aborting(two_commit_repo, mock_db):
    repo_path, sha_a, sha_b = two_commit_repo
    mock_calc = MagicMock()
    mock_calc.calculate_batch.side_effect = RuntimeError("embedding backend unavailable")

    indexer = _make_indexer(repo_path, sha_a, sha_b, mock_db)
    indexer.calculator = mock_calc

    from Chunker.Chunk import Chunk

    with patch("Indexer.SOFT_TIMEOUT_SECONDS", 9999), \
         patch("Indexer.chunker.chunk_file") as mock_chunk, \
         patch("Indexer.logger") as mock_logger:
        mock_chunk.side_effect = lambda path, repo, branch: [
            Chunk(
                chunk="c",
                repo=repo,
                path=path,
                language="python",
                start_rc=(0, 0),
                end_rc=(1, 0),
                start_bytes=0,
                end_bytes=1,
                signature="repo.fn",
            )
        ]

        result = indexer.index()

    assert result is not None
    assert result.error_reports
    assert any("Indexing error:" in call.args[0] for call in mock_logger.error.call_args_list)


