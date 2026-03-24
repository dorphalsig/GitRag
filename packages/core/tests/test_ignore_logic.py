import importlib
import os
import sys
from unittest.mock import patch

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.getcwd(), "packages", "core", "src"))
Indexer = importlib.import_module("Indexer")


def test_get_ignore_spec_parses_comma_and_semicolon_patterns():
    with patch.dict(os.environ, {"GITRAG_IGNORE": "node_modules/**;dist/**,*.md"}):
        spec = Indexer._get_ignore_spec()

    assert spec is not None
    assert spec.match_file("node_modules/foo/bar.js")
    assert spec.match_file("dist/bundle.js")
    assert spec.match_file("README.md")
    assert not spec.match_file("src/main.py")


def test_iter_selected_paths_respects_ignore_for_process_delete_and_rename():
    with patch.dict(os.environ, {"GITRAG_IGNORE": "ignored/"}):
        changes = iter(
            [
                ("M", "src/main.py", ""),
                ("D", "ignored/deleted.txt", ""),
                ("A", "ignored/new.txt", ""),
                ("R100", "old.txt", "ignored/renamed.txt"),
            ]
        )
        with patch.object(Indexer, "_iter_git_changes", return_value=changes):
            detector = type("Detector", (), {"is_binary": staticmethod(lambda _p: False)})()
            actions = list(Indexer._iter_selected_paths(("HEAD^", "HEAD"), detector, False, set()))

    assert {"action": "process", "path": "src/main.py", "reason": "status=M"} in actions
    assert {"action": "delete", "path": "old.txt", "reason": "rename/copy"} in actions
    assert not any(a["path"] == "ignored/new.txt" for a in actions)
    assert not any(a["path"] == "ignored/deleted.txt" for a in actions)
    assert not any(a["path"] == "ignored/renamed.txt" for a in actions)
