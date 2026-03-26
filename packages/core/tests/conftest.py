import pytest
import subprocess
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

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

    # Create an initial commit
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
    persist = MagicMock()
    return persist

@pytest.fixture
def patch_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "libsql://test")
    monkeypatch.setenv("DB_PROVIDER", "libsql")
    monkeypatch.setenv("DB_AUTH_TOKEN", "test-token")
