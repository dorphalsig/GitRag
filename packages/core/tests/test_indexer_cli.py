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
            def calculate(self, chunk: str) -> bytes:
                return chunk.encode("utf-8")

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

if __name__ == "__main__":
    unittest.main()
