import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


if "numpy" not in sys.modules:
    from array import array

    class _FakeArray(list):
        def __init__(self, data):
            super().__init__(data)

        @property
        def shape(self):
            return (len(self),)

        def tobytes(self):
            return array("f", self).tobytes()

    class _FakeMatrix:
        def __init__(self, rows: int, dim: int):
            self._rows = [_FakeArray([0.0] * dim) for _ in range(rows)]
            self.shape = (rows, dim)

        def __getitem__(self, idx):
            return self._rows[idx]

    def _fake_asarray(data, dtype=None):
        if isinstance(data, _FakeArray):
            return data
        return _FakeArray(list(data))

    fake_np = types.SimpleNamespace(asarray=_fake_asarray, float32="float32")
    sys.modules["numpy"] = fake_np

if "sentence_transformers" not in sys.modules:
    class _FakeSentenceTransformer:
        def __init__(self, repo, trust_remote_code=True, device=None):
            self._dim = 768

        def encode(self, texts, normalize_embeddings=True):
            return _FakeMatrix(len(texts), self._dim)

    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer
    )

if "chunker" not in sys.modules:
    sys.modules["chunker"] = types.SimpleNamespace(chunk_file=lambda path, repo, branch=None: [])

try:
    import text_detection  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency missing
    class _StubBinaryDetector:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["text_detection"] = types.SimpleNamespace(BinaryDetector=_StubBinaryDetector)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

Indexer = None


def _ensure_indexer():
    global Indexer
    if Indexer is not None:
        return
    import importlib

    try:
        Indexer = importlib.import_module("Indexer")  # type: ignore
    except Exception:  # pragma: no cover - dependencies missing
        Indexer = None  # type: ignore



class ProcessFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_indexer()
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
            @property
            def dimensions(self) -> int:
                return 4

            def calculate(self, chunk: str) -> bytes:
                return chunk.encode("utf-8")

            def calculate_batch(self, chunks: list[str]) -> list[bytes]:
                return [self.calculate(c) for c in chunks]

        class StubPersist:
            def __init__(self):
                self.received = []

            def persist_batch(self, batch):
                self.received.append(list(batch))

        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            return [StubChunk(path)]

        Indexer.chunker.chunk_file = fake_chunk_file
        calc = StubCalc()
        persist = StubPersist()
        try:
            count,_ = Indexer._process_files(["foo.py"], "repo", calc, persist)
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
        _ensure_indexer()
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
        _ensure_indexer()
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_resolve_db_cfg_reads_legacy_libsql_env(self):
        with mock.patch.dict(
            "os.environ",
            {
                "TURSO_DATABASE_URL": "libsql://example-db",
                "TURSO_AUTH_TOKEN": "secret",
                "LIBSQL_TABLE": "chunks_custom",
                "LIBSQL_FTS_TABLE": "chunks_fts_custom",
            },
            clear=True,
        ):
            cfg = Indexer._resolve_db_cfg()
        self.assertEqual(cfg.url, "libsql://example-db")
        self.assertEqual(cfg.provider, "libsql")
        self.assertEqual(cfg.auth_token, "secret")
        self.assertEqual(cfg.table, "chunks_custom")
        self.assertEqual(cfg.resolved_fts_table, "chunks_fts_custom")

    def test_resolve_db_cfg_requires_url(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                Indexer._resolve_db_cfg()

class CheckpointingTests(unittest.TestCase):
    """Task 8: Verify skip_paths filtering in _process_files."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_indexer()
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def _make_stubs(self):
        class StubCalc:
            @property
            def dimensions(self) -> int:
                return 4

            def calculate_batch(self, chunks: list[str]) -> list[bytes]:
                return [c.encode("utf-8") for c in chunks]

        class StubPersist:
            def __init__(self):
                self.received = []

            def persist_batch(self, batch):
                self.received.append(list(batch))

        return StubCalc(), StubPersist()

    def test_skip_paths_excludes_already_indexed(self):
        """Files listed in skip_paths must not be processed."""
        processed_paths = []
        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            processed_paths.append(path)
            return []

        Indexer.chunker.chunk_file = fake_chunk_file
        calc, persist = self._make_stubs()
        try:
            result = Indexer._process_files(
                ["a.py", "b.py", "c.py"],
                "repo",
                calc,
                persist,
                skip_paths={"b.py"},
            )
        finally:
            Indexer.chunker.chunk_file = original_chunk_file

        self.assertNotIn("b.py", processed_paths)
        self.assertIn("a.py", processed_paths)
        self.assertIn("c.py", processed_paths)

    def test_skip_paths_empty_set_processes_all(self):
        """When skip_paths is empty, all paths are processed."""
        processed_paths = []
        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            processed_paths.append(path)
            return []

        Indexer.chunker.chunk_file = fake_chunk_file
        calc, persist = self._make_stubs()
        try:
            Indexer._process_files(
                ["x.py", "y.py"],
                "repo",
                calc,
                persist,
                skip_paths=set(),
            )
        finally:
            Indexer.chunker.chunk_file = original_chunk_file

        self.assertCountEqual(processed_paths, ["x.py", "y.py"])

    def test_timed_out_false_when_no_timeout(self):
        """ProcessFilesResult.timed_out is False when processing completes normally."""
        original_chunk_file = Indexer.chunker.chunk_file
        Indexer.chunker.chunk_file = lambda path, repo, branch=None: []
        calc, persist = self._make_stubs()
        try:
            result = Indexer._process_files(["z.py"], "repo", calc, persist)
        finally:
            Indexer.chunker.chunk_file = original_chunk_file

        self.assertFalse(result.timed_out)


class SoftTimeoutTests(unittest.TestCase):
    """Task 9: Verify soft timeout detection in _process_files."""

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_indexer()
        if Indexer is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def _make_stubs(self):
        class StubCalc:
            @property
            def dimensions(self) -> int:
                return 4

            def calculate_batch(self, chunks: list[str]) -> list[bytes]:
                return [c.encode("utf-8") for c in chunks]

        class StubPersist:
            def __init__(self):
                self.received = []

            def persist_batch(self, batch):
                self.received.append(list(batch))

        return StubCalc(), StubPersist()

    def test_timeout_triggers_early_return(self):
        """When elapsed time exceeds SOFT_TIMEOUT_SECONDS, _process_files returns with timed_out=True."""
        processed_paths = []
        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            processed_paths.append(path)
            return []

        Indexer.chunker.chunk_file = fake_chunk_file
        calc, persist = self._make_stubs()

        # Simulate: start_time is 1000.0, first time.time() call returns 1001.0
        # SOFT_TIMEOUT_SECONDS patched to 1, so elapsed (1.0) >= timeout (1) triggers early exit
        time_calls = iter([1001.0])

        def fake_time():
            return next(time_calls)

        original_soft_timeout = Indexer.SOFT_TIMEOUT_SECONDS
        Indexer.SOFT_TIMEOUT_SECONDS = 1
        original_time = Indexer.time.time
        Indexer.time.time = fake_time
        try:
            result = Indexer._process_files(
                ["first.py", "second.py", "third.py"],
                "repo",
                calc,
                persist,
                start_time=1000.0,
            )
        finally:
            Indexer.chunker.chunk_file = original_chunk_file
            Indexer.SOFT_TIMEOUT_SECONDS = original_soft_timeout
            Indexer.time.time = original_time

        self.assertTrue(result.timed_out)
        # Only files processed before timeout should appear
        self.assertEqual(len(processed_paths), 0)

    def test_no_timeout_when_soft_timeout_zero(self):
        """When SOFT_TIMEOUT_SECONDS is 0, timeout check is disabled and all files process normally."""
        processed_paths = []
        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            processed_paths.append(path)
            return []

        Indexer.chunker.chunk_file = fake_chunk_file
        calc, persist = self._make_stubs()

        # Even if elapsed is huge, SOFT_TIMEOUT_SECONDS=0 disables the check
        original_soft_timeout = Indexer.SOFT_TIMEOUT_SECONDS
        Indexer.SOFT_TIMEOUT_SECONDS = 0
        try:
            result = Indexer._process_files(
                ["a.py", "b.py"],
                "repo",
                calc,
                persist,
                start_time=0.0,
            )
        finally:
            Indexer.chunker.chunk_file = original_chunk_file
            Indexer.SOFT_TIMEOUT_SECONDS = original_soft_timeout

        self.assertFalse(result.timed_out)
        self.assertCountEqual(processed_paths, ["a.py", "b.py"])

    def test_no_timeout_when_start_time_none(self):
        """When start_time is None, timeout check is disabled."""
        processed_paths = []
        original_chunk_file = Indexer.chunker.chunk_file

        def fake_chunk_file(path, repo, branch=None):
            processed_paths.append(path)
            return []

        Indexer.chunker.chunk_file = fake_chunk_file
        calc, persist = self._make_stubs()

        original_soft_timeout = Indexer.SOFT_TIMEOUT_SECONDS
        Indexer.SOFT_TIMEOUT_SECONDS = 1
        try:
            result = Indexer._process_files(
                ["a.py", "b.py"],
                "repo",
                calc,
                persist,
                start_time=None,
            )
        finally:
            Indexer.chunker.chunk_file = original_chunk_file
            Indexer.SOFT_TIMEOUT_SECONDS = original_soft_timeout

        self.assertFalse(result.timed_out)
        self.assertCountEqual(processed_paths, ["a.py", "b.py"])


if __name__ == "__main__":
    unittest.main()
