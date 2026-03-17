import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import importlib

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.getcwd(), 'packages', 'core', 'src'))
Indexer = importlib.import_module("Indexer")

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

    def test_is_ignored(self):
        patterns = ["node_modules/*", "*.md", "dist"]
        self.assertTrue(Indexer._is_ignored("node_modules/foo/bar.js", patterns))
        self.assertTrue(Indexer._is_ignored("README.md", patterns))
        self.assertTrue(Indexer._is_ignored("dist/bundle.js", patterns))
        self.assertTrue(Indexer._is_ignored("dist", patterns))
        self.assertFalse(Indexer._is_ignored("src/main.py", patterns))
        
        patterns_with_slash = ["build/"]
        self.assertTrue(Indexer._is_ignored("build/app.exe", patterns_with_slash))
        self.assertFalse(Indexer._is_ignored("builder.py", patterns_with_slash))

    @patch('Indexer._run_git')
    @patch('os.environ.get')
    def test_collect_full_repo_with_ignore(self, mock_get, mock_run_git):
        mock_get.side_effect = lambda k, d=None: "ignored/*" if k == "GITRAG_IGNORE" else d
        
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
        
        # Check if it's in skipped (might be useful for reporting)
        # Actually if it's ignored via env var, it's different from being binary.
        # But for now, let's just ensure it's not in 'text'.

    @patch('Indexer._run_git')
    @patch('os.environ.get')
    def test_collect_changes_with_ignore(self, mock_get, mock_run_git):
        mock_get.side_effect = lambda k, d=None: "ignored/*" if k == "GITRAG_IGNORE" else d
        
        # Output of git diff --name-status -z
        # Format: STATUS\tPATH1\0[PATH2\0]
        # For M: M\tpath\0
        # For R: R100\told\0new\0
        mock_run_git.return_value = "M\tsrc/main.py\x00D\tignored/deleted.txt\x00A\tignored/new.txt\x00R100\told.txt\x00ignored/renamed.txt\x00"
        
        to_proc, to_del, actions = Indexer._collect_changes(("HEAD^", "HEAD"))
        
        self.assertIn("src/main.py", to_proc)
        self.assertNotIn("ignored/new.txt", to_proc)
        self.assertNotIn("ignored/renamed.txt", to_proc)
        self.assertNotIn("ignored/deleted.txt", to_del)
        
        # old.txt was renamed to ignored/renamed.txt. 
        # If we ignore renamed.txt, should we still delete old.txt?
        # Probably yes, because it's no longer there.
        self.assertIn("old.txt", to_del)

if __name__ == "__main__":
    unittest.main()
