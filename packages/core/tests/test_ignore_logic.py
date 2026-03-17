import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import importlib

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.getcwd(), 'packages', 'core', 'src'))
Indexer = importlib.import_module("Indexer")
import pathspec

class IgnoreLogicTests(unittest.TestCase):
    @patch('os.environ.get')
    def test_get_ignore_patterns(self, mock_get):
        mock_get.side_effect = lambda k, d=None: "node_modules/*,dist/**,*.md" if k == "GITRAG_IGNORE" else d
        patterns = Indexer._get_ignore_patterns()
        self.assertEqual(patterns, ["node_modules/*", "dist/**", "*.md"])
        
        mock_get.side_effect = lambda k, d=None: "  foo  ;  bar  " if k == "GITRAG_IGNORE" else d
        patterns = Indexer._get_ignore_patterns()
        self.assertEqual(patterns, ["foo", "bar"])
        
        mock_get.side_effect = lambda k, d=None: "" if k == "GITRAG_IGNORE" else d
        patterns = Indexer._get_ignore_patterns()
        self.assertEqual(patterns, [])

    def test_is_path_ignored(self):
        patterns = ["node_modules/", "*.md", "dist/"]
        spec = pathspec.PathSpec.from_lines("gitignore", patterns)
        
        self.assertTrue(Indexer._is_path_ignored("node_modules/foo/bar.js", spec))
        self.assertTrue(Indexer._is_path_ignored("README.md", spec))
        self.assertTrue(Indexer._is_path_ignored("dist/bundle.js", spec))
        self.assertFalse(Indexer._is_path_ignored("src/main.py", spec))
        
        # In .gitignore, 'dist/' matches 'dist/file' but 'dist' (without slash) matches file named 'dist' OR dir 'dist'
        # Pathspec handles this correctly.
        
        spec2 = pathspec.PathSpec.from_lines("gitignore", ["build/"])
        self.assertTrue(Indexer._is_path_ignored("build/app.exe", spec2))
        self.assertFalse(Indexer._is_path_ignored("builder.py", spec2))

    @patch('Indexer._run_git')
    @patch('os.environ.get')
    def test_collect_full_repo_with_ignore(self, mock_get, mock_run_git):
        mock_get.side_effect = lambda k, d=None: "ignored/" if k == "GITRAG_IGNORE" else d
        
        def fake_run_git(args):
            if "ls-files" in args:
                if "--cached" in args:
                    return "src/main.py\nignored/secret.txt"
                if "--others" in args:
                    return "new_file.py"
            return ""
        
        mock_run_git.side_effect = fake_run_git
        
        detector = MagicMock()
        detector.is_binary.return_value = False
        
        text, skipped, actions = Indexer._collect_full_repo(detector)
        
        self.assertIn("src/main.py", text)
        self.assertIn("new_file.py", text)
        self.assertNotIn("ignored/secret.txt", text)

    @patch('Indexer._run_git')
    @patch('os.environ.get')
    def test_collect_changes_with_ignore(self, mock_get, mock_run_git):
        # The user mentioned git -z format: status\0path\0
        mock_get.side_effect = lambda k, d=None: "ignored/" if k == "GITRAG_IGNORE" else d
        
        # M\0path\0D\0path\0R100\0old\0new\0
        mock_run_git.return_value = "M\x00src/main.py\x00D\x00ignored/deleted.txt\x00A\x00ignored/new.txt\x00R100\x00old.txt\x00ignored/renamed.txt\x00"
        
        to_proc, to_del, actions = Indexer._collect_changes(("HEAD^", "HEAD"))
        
        self.assertIn("src/main.py", to_proc)
        self.assertNotIn("ignored/new.txt", to_proc)
        self.assertNotIn("ignored/renamed.txt", to_proc)
        self.assertNotIn("ignored/deleted.txt", to_del)
        
        # old.txt was renamed to ignored/renamed.txt. 
        self.assertIn("old.txt", to_del)

if __name__ == "__main__":
    unittest.main()
