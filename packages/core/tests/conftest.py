import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Prevent EmbeddingCalculator._load_model from running when Indexer is imported.
# Indexer.calculator = EmbeddingCalculator() is a class-level attribute — it runs
# at import time. This patch must be .start()ed at module level (not inside a fixture)
# so it is active before pytest imports any test file.
patch("Calculators.EmbeddingCalculator.EmbeddingCalculator._load_model").start()


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    def run_git(*args):
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, check=True
        )

    run_git("init")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "user.name", "Test User")

    initial_file = repo_dir / "README.md"
    initial_file.write_text("# Test Repo")
    run_git("add", "README.md")
    run_git("commit", "-m", "Initial commit")

    return repo_dir


@pytest.fixture
def mock_calc():
    calc = MagicMock()
    calc.dimensions = 1024
    calc.calculate_batch.side_effect = lambda x: [b"0" * 4096 for _ in x]
    return calc


@pytest.fixture
def mock_persist():
    return MagicMock()


@pytest.fixture
def patch_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "libsql://test")
    monkeypatch.setenv("DB_PROVIDER", "libsql")
    monkeypatch.setenv("DB_AUTH_TOKEN", "test-token")


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_indexed_paths.return_value = set()
    return db


@pytest.fixture
def two_commit_repo(tmp_path, monkeypatch):
    """
    A temp git repo with two commits:
      sha_a — adds: keep.py, delete.py, to_rename.py, skip.log, image.bin (binary)
      sha_b — adds new.py, deletes delete.py, renames to_rename.py→renamed.py (delete+add),
              modifies skip.log

    Returns (repo_path, sha_a, sha_b). cwd is set to repo_path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "T")

    # Commit A
    (repo / "keep.py").write_text("keep")
    (repo / "delete.py").write_text("delete")
    (repo / "to_rename.py").write_text("rename me")
    (repo / "skip.log").write_text("log entry")
    # PNG header + null bytes — magika identifies this as binary
    (repo / "image.bin").write_bytes(
        bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(100)
    )
    git("add", ".")
    git("commit", "-m", "base")
    sha_a = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    # Commit B
    (repo / "new.py").write_text("new file")
    (repo / "delete.py").unlink()
    (repo / "to_rename.py").unlink()
    (repo / "renamed.py").write_text("renamed content")
    (repo / "skip.log").write_text("updated log")
    git("add", ".")
    git("add", "-u")
    git("commit", "-m", "changes")
    sha_b = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    monkeypatch.chdir(repo)
    return repo, sha_a, sha_b
