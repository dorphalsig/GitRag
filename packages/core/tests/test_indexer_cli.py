import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Minimal stubs for heavy Indexer dependencies
def _stub_module(name, **kwargs):
    if name not in sys.modules:
        sys.modules[name] = types.SimpleNamespace(**kwargs)

_stub_module("sentence_transformers", SentenceTransformer=type("SentenceTransformer", (), {"__init__": lambda *a, **k: None, "encode": lambda *a, **k: []}))
_stub_module("transformers", AutoModel=None, AutoTokenizer=None)
_stub_module("tree_sitter", Node=None, Language=None, Parser=None, Query=None, QueryError=Exception, QueryCursor=None)
_stub_module("tree_sitter_language_pack", get_language=lambda *a: None, get_parser=lambda *a: None)
_stub_module("einops", rearrange=lambda *a, **k: None)

# Add SRC to path for imports
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Now we can import Indexer
import Indexer

class IndexerRefactorTests(unittest.TestCase):
    def setUp(self):
        self.repo = "test/repo"
        self.calc = mock.Mock()
        self.persist = mock.Mock()
        self.persist.get_indexed_paths.return_value = set()
        self.detector = mock.Mock()
        self.detector.is_binary.return_value = False
        # Patch environment for DB config
        self.env_patcher = mock.patch.dict("os.environ", {"DATABASE_URL": "libsql://test"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_resolve_range_fallback(self):
        with mock.patch("Indexer._run_git") as mock_git:
            def fake_git(args):
                if args[0] == "hash-object": return "empty_tree_sha"
                raise Exception("git failed")
            mock_git.side_effect = fake_git

            rng = Indexer._resolve_range()
            self.assertEqual(rng, ("empty_tree_sha", "HEAD"))

    def test_iter_git_changes_normal(self):
        with mock.patch("Indexer._run_git") as mock_git:
            mock_git.return_value = "A\x00file1.py\x00M\x00file2.py\x00"
            changes = list(Indexer._iter_git_changes(("A", "B")))
            self.assertEqual(changes, [("A", "file1.py", ""), ("M", "file2.py", "")])

    def test_iter_git_changes_rename(self):
        with mock.patch("Indexer._run_git") as mock_git:
            mock_git.return_value = "R100\x00old.py\x00new.py\x00"
            changes = list(Indexer._iter_git_changes(("A", "B")))
            self.assertEqual(changes, [("R100", "old.py", "new.py")])

    def test_iter_selected_paths_filters_binary(self):
        with mock.patch("Indexer._iter_git_changes") as mock_changes:
            mock_changes.return_value = iter([("A", "text.py", ""), ("A", "bin.exe", "")])
            self.detector.is_binary.side_effect = lambda p: p == "bin.exe"

            actions = list(Indexer._iter_selected_paths(("A", "B"), self.detector, False, set()))
            paths = [a["path"] for a in actions if a["action"] == "process"]
            self.assertIn("text.py", paths)
            self.assertNotIn("bin.exe", paths)

    def test_iter_selected_paths_filters_indexed_in_full(self):
        with mock.patch("Indexer._iter_git_changes") as mock_changes:
            mock_changes.return_value = iter([("A", "new.py", ""), ("A", "old.py", "")])

            actions = list(Indexer._iter_selected_paths(("A", "B"), self.detector, True, {"old.py"}))
            processed = [a["path"] for a in actions if a["action"] == "process"]
            skipped = [a["path"] for a in actions if a["action"] == "skip" and a["reason"] == "already-indexed"]

            self.assertIn("new.py", processed)
            self.assertIn("old.py", skipped)

    def test_run_indexing_batches_and_persists_complete_files(self):
        class MockChunk:
            def __init__(self, path):
                self.path = path
                self.embeddings = None
                self.repo = "repo"
                self.branch = "branch"
                self.chunk = "chunk"
            def id(self): return f"{self.path}_id"

        chunks_file1 = [MockChunk("file1.py"), MockChunk("file1.py")]
        chunks_file2 = [MockChunk("file2.py")]

        with mock.patch("Indexer.chunker.chunk_file") as mock_chunk_file, \
             mock.patch("Indexer.EmbeddingCalculator") as mock_calc_cls, \
             mock.patch("Indexer.create_persistence_adapter") as mock_persist_factory, \
             mock.patch("Indexer.BinaryDetector") as mock_detector_cls, \
             mock.patch("Indexer.EMBEDDING_BATCH_SIZE", 2):

            mock_chunk_file.side_effect = lambda path, *a, **k: chunks_file1 if path == "file1.py" else chunks_file2
            calc = mock_calc_cls.return_value
            calc.calculate_batch.side_effect = lambda texts: [b"emb"] * len(texts)
            persist = mock_persist_factory.return_value
            persist.get_indexed_paths.return_value = set()

            with mock.patch("Indexer._iter_selected_paths") as mock_select:
                mock_select.return_value = iter([
                    {"action": "process", "path": "file1.py", "reason": "test"},
                    {"action": "process", "path": "file2.py", "reason": "test"}
                ])

                res = Indexer._run_indexing("repo", ("A", "B"), False)

                self.assertEqual(res.processed_files, 2)
                self.assertEqual(res.processed_chunks, 3)
                self.assertEqual(persist.persist_batch.call_count, 2)
                persisted_paths = [call.args[0][0].path for call in persist.persist_batch.call_args_list]
                self.assertEqual(persisted_paths, ["file1.py", "file2.py"])

    def test_run_indexing_handles_lazy_chunk_iterables(self):
        class MockChunk:
            def __init__(self, path):
                self.path = path
                self.embeddings = None
                self.repo = "repo"
                self.branch = "branch"
                self.chunk = "chunk"
            def id(self): return f"{self.path}_id"

        def lazy_chunks(path, *args, **kwargs):
            if path == "VERSION":
                yield MockChunk(path)
            else:
                return

        with mock.patch("Indexer.chunker.chunk_file", side_effect=lazy_chunks), \
             mock.patch("Indexer.EmbeddingCalculator") as mock_calc_cls, \
             mock.patch("Indexer.create_persistence_adapter") as mock_persist_factory, \
             mock.patch("Indexer.BinaryDetector"), \
             mock.patch("Indexer.EMBEDDING_BATCH_SIZE", 4):
            calc = mock_calc_cls.return_value
            calc.calculate_batch.side_effect = lambda texts: [b"emb"] * len(texts)
            persist = mock_persist_factory.return_value
            persist.get_indexed_paths.return_value = set()

            with mock.patch("Indexer._iter_selected_paths") as mock_select:
                mock_select.return_value = iter([
                    {"action": "process", "path": "VERSION", "reason": "test"},
                ])
                res = Indexer._run_indexing("repo", ("A", "B"), False)

        self.assertEqual(res.processed_files, 1)
        self.assertEqual(res.processed_chunks, 1)
        persist.persist_batch.assert_called_once()
        self.assertEqual(persist.persist_batch.call_args.args[0][0].path, "VERSION")

    def test_persistence_requires_all_chunks_embedded(self):
        class MockChunk:
            def __init__(self, path):
                self.path = path
                self.embeddings = None
                self.repo = "repo"
                self.branch = "branch"
                self.chunk = "chunk"
            def id(self): return f"{self.path}_id"

        chunks = [MockChunk("a.py"), MockChunk("a.py"), MockChunk("b.py"), MockChunk("b.py")]

        with mock.patch("Indexer.chunker.chunk_file") as mock_chunk_file, \
             mock.patch("Indexer.EmbeddingCalculator") as mock_calc_cls, \
             mock.patch("Indexer.create_persistence_adapter") as mock_persist_factory, \
             mock.patch("Indexer.BinaryDetector"), \
             mock.patch("Indexer.EMBEDDING_BATCH_SIZE", 2):
            mock_chunk_file.side_effect = lambda path, *a, **k: chunks[0:2] if path == "a.py" else chunks[2:4]
            calc = mock_calc_cls.return_value
            calc.calculate_batch.side_effect = RuntimeError("embedding failure")
            persist = mock_persist_factory.return_value
            persist.get_indexed_paths.return_value = set()

            with mock.patch("Indexer._iter_selected_paths") as mock_select:
                mock_select.return_value = iter([
                    {"action": "process", "path": "a.py", "reason": "test"},
                    {"action": "process", "path": "b.py", "reason": "test"},
                ])
                res = Indexer._run_indexing("repo", ("A", "B"), False)

        self.assertIn("a.py", res.failed_paths)
        self.assertIn("b.py", res.failed_paths)
        persist.persist_batch.assert_not_called()

    def test_timeout_gate_before_batch(self):
        with mock.patch("Indexer.chunker.chunk_file") as mock_chunk_file, \
             mock.patch("Indexer.EmbeddingCalculator") as mock_calc_cls, \
             mock.patch("Indexer.create_persistence_adapter") as mock_persist_factory, \
             mock.patch("Indexer.BinaryDetector") as mock_detector_cls, \
             mock.patch("Indexer.EMBEDDING_BATCH_SIZE", 1), \
             mock.patch("Indexer.SOFT_TIMEOUT_SECONDS", 1), \
             mock.patch("Indexer.time.time") as mock_time:

            mock_chunk_file.return_value = [mock.Mock(path="f.py")]
            mock_time.side_effect = [1000.1, 1002.0]

            with mock.patch("Indexer._iter_selected_paths") as mock_select:
                mock_select.return_value = iter([{"action": "process", "path": "f.py", "reason": "test"}])

                res = Indexer._run_indexing("repo", ("A", "B"), False, start_time=1000.0)

                self.assertTrue(res.timed_out)
                self.assertEqual(res.processed_chunks, 0)
                mock_calc_cls.return_value.calculate_batch.assert_not_called()

    def test_main_returns_timeout_exit_code_75(self):
        with mock.patch.object(sys, "argv", ["Indexer.py", "org/repo"]), \
             mock.patch("Indexer._resolve_range", return_value=("A", "B")), \
             mock.patch("Indexer._run_indexing", return_value=Indexer.IndexingResult(timed_out=True)):
            with self.assertRaises(SystemExit) as raised:
                Indexer.main()

        self.assertEqual(raised.exception.code, 75)

if __name__ == "__main__":
    unittest.main()
