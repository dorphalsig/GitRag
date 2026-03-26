import json
import pytest
import os
import subprocess
from io import StringIO
from unittest.mock import MagicMock, patch
from Indexer import main

def test_main_full_indexing(temp_git_repo, monkeypatch, mock_calc, mock_persist, patch_env):
    monkeypatch.chdir(temp_git_repo)

    with patch("Indexer.EmbeddingCalculator", return_value=mock_calc), \
         patch("Indexer.create_persistence_adapter", return_value=mock_persist), \
         patch("Indexer.chunker.chunk_file") as mock_chunk:

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.path = "README.md"
        mock_chunk_obj.chunk = "content"
        mock_chunk.return_value = [mock_chunk_obj]

        monkeypatch.setattr("sys.argv", ["indexer", "test-repo", "--full"])

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            main()

        assert mock_chunk.called
        assert mock_persist.persist_batch.called

        summary = json.loads(stdout.getvalue())
        assert summary["repo"] == "test-repo"
        assert summary["mode"] == "full"
        assert summary["processed_files"] > 0
        assert summary["deleted_files"] == 0

def test_main_delta_indexing(temp_git_repo, monkeypatch, mock_calc, mock_persist, patch_env):
    monkeypatch.chdir(temp_git_repo)

    # Create a change
    new_file = temp_git_repo / "new.txt"
    new_file.write_text("new content")
    subprocess.run(["git", "add", "new.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "add new.txt"], check=True)

    with patch("Indexer.EmbeddingCalculator", return_value=mock_calc), \
         patch("Indexer.create_persistence_adapter", return_value=mock_persist), \
         patch("Indexer.chunker.chunk_file") as mock_chunk:

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.path = "new.txt"
        mock_chunk_obj.chunk = "content"
        mock_chunk.return_value = [mock_chunk_obj]

        monkeypatch.setattr("sys.argv", ["indexer", "test-repo"])

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            main()

        summary = json.loads(stdout.getvalue())
        assert summary["mode"] == "delta"
        assert summary["processed_files"] == 1

def test_main_error_handling(temp_git_repo, monkeypatch, mock_calc, mock_persist, patch_env):
    monkeypatch.chdir(temp_git_repo)

    with patch("Indexer.EmbeddingCalculator", return_value=mock_calc), \
         patch("Indexer.create_persistence_adapter", return_value=mock_persist), \
         patch("Indexer.chunker.chunk_file", side_effect=Exception("Chunking failed")):

        monkeypatch.setattr("sys.argv", ["indexer", "test-repo", "--full"])

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1

def test_process_files_batching_and_flushing(mock_calc, mock_persist):
    from Indexer import _process_files, EMBEDDING_BATCH_SIZE
    from Chunker.Chunk import Chunk
    import Indexer

    # Create a dummy repo path
    repo = "test-repo"

    # Setup mock chunker to return chunks for 2 files
    # total chunks = EMBEDDING_BATCH_SIZE + 10
    total_chunks_count = EMBEDDING_BATCH_SIZE + 10

    mock_chunks = []
    for i in range(total_chunks_count):
        c = MagicMock(spec=Chunk)
        c.path = f"file_{i // 10}.txt"
        c.chunk = f"content_{i}"
        mock_chunks.append(c)

    with patch("Indexer.chunker.chunk_file") as mock_chunk_file:
        # First file returns EMBEDDING_BATCH_SIZE + 5
        # Second file returns 5
        mock_chunk_file.side_effect = [
            mock_chunks[:EMBEDDING_BATCH_SIZE + 5],
            mock_chunks[EMBEDDING_BATCH_SIZE + 5:]
        ]

        paths = ["file_0.txt", "file_1.txt"]

        res = _process_files(paths, repo, mock_calc, mock_persist)

        assert res[0] == total_chunks_count
        assert len(res[1]) == 0

        # calculate_batch should be called twice:
        # 1. when count >= EMBEDDING_BATCH_SIZE (64)
        # 2. at the end for the remaining (10)
        assert mock_calc.calculate_batch.call_count == 2
        assert mock_persist.persist_batch.call_count == 2

        # Verify first batch size
        first_batch_text = mock_calc.calculate_batch.call_args_list[0][0][0]
        assert len(first_batch_text) == EMBEDDING_BATCH_SIZE

        # Verify second batch size
        second_batch_text = mock_calc.calculate_batch.call_args_list[1][0][0]
        assert len(second_batch_text) == 10

def test_process_files_persistence_only_on_success(mock_calc, mock_persist):
    from Indexer import _process_files
    from Chunker.Chunk import Chunk

    repo = "test-repo"
    c = MagicMock(spec=Chunk)
    c.path = "file.txt"
    c.chunk = "content"

    with patch("Indexer.chunker.chunk_file", return_value=[c]):
        # Simulate embedding failure
        mock_calc.calculate_batch.side_effect = Exception("GPU OOM")

        res = _process_files(["file.txt"], repo, mock_calc, mock_persist)

        assert res[0] == 0
        assert "file.txt" in res[1]
        # persist_batch should NOT be called if calculate_batch fails
        assert not mock_persist.persist_batch.called
