import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import Indexer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency missing
    Indexer = None  # type: ignore


class ProcessFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_process_files_batches_chunks(self):
        class StubChunk:
            def __init__(self, path: str):
                self.chunk = "body"
                self.repo = "repo"
                self.path = path
                self.language = "python"
                self.start_rc = (0, 0)
                self.end_rc = (0, 4)
                self.start_bytes = 0
                self.end_bytes = 4
                self.signature = "sig"
                self.embeddings = None

            def calculate_embeddings(self, calculator):
                self.embeddings = calculator.calculate(self.chunk)

        class StubCalc:
            def calculate(self, chunk: str) -> bytes:
                return chunk.encode("utf-8")

        class StubPersist:
            def __init__(self):
                self.received = []

            def persist_batch(self, batch):
                self.received.append(list(batch))

        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo):
            return [StubChunk(path)]

        Indexer.chunker.chunk_file = fake_chunk_file
        calc = StubCalc()
        persist = StubPersist()
        try:
            count = Indexer._process_files(["foo.py"], "repo", calc, persist)
        finally:
            Indexer.chunker.chunk_file = original_chunk_file

        self.assertEqual(count, 1)
        self.assertEqual(len(persist.received), 1)
        chunk = persist.received[0][0]
        self.assertIsInstance(chunk.embeddings, bytes)
        self.assertEqual(chunk.path, "foo.py")


class FullIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_collect_full_repo_filters_binary(self):
        class StubDetector:
            def __init__(self):
                self.calls: list[str] = []

            def is_binary(self, path: str) -> bool:
                self.calls.append(path)
                return path.endswith(".bin")

        detector = StubDetector()
        original_run_git = Indexer._run_git

        def fake_run_git(args: list[str]) -> str:
            mapping = {
                ("ls-files", "--cached"): "a.txt\nb.bin\n",
                ("ls-files", "--others", "--exclude-standard"): "c.md\n",
            }
            return mapping.get(tuple(args), "")

        Indexer._run_git = fake_run_git
        try:
            text, skipped, actions = Indexer._collect_full_repo(detector)
        finally:
            Indexer._run_git = original_run_git

        self.assertEqual(text, {"a.txt", "c.md"})
        self.assertEqual(skipped, ["b.bin"])
        self.assertEqual(actions, [
            {"action": "process", "path": "a.txt", "reason": "full-index"},
            {"action": "process", "path": "c.md", "reason": "full-index"},
        ])
        self.assertCountEqual(detector.calls, ["a.txt", "b.bin", "c.md"])


class EnvResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_prefers_clouflare_prefixed_envs(self):
        with mock.patch.dict(
            "os.environ",
            {
                "CLOUDFLARE_ACCOUNT_ID": "acct",
                "CLOUDFLARE_VECTORIZE_INDEX": "vec",
                "CLOUDFLARE_D1_DATABASE_ID": "db",
            },
            clear=True,
        ):
            cfg = Indexer._resolve_cf_ids()
        self.assertEqual(cfg.account_id, "acct")
        self.assertEqual(cfg.vectorize_index, "vec")
        self.assertEqual(cfg.d1_database_id, "db")

if __name__ == "__main__":
    unittest.main()
