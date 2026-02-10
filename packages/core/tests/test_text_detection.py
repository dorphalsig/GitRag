import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_detection import BinaryDetector
try:
    from Indexer import _filter_text_files  # type: ignore
except ImportError:  # pragma: no cover - optional dependency missing
    _filter_text_files = None  # type: ignore


class BinaryDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _filter_text_files is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_heuristic_binary_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            text = tmp / "text.txt"
            binary = tmp / "binary.bin"
            text.write_bytes(b"hello world\n")
            binary.write_bytes(b"\x00\xff\x00 bytes")

            detector = BinaryDetector(base_dir=tmp)
            self.assertFalse(detector.is_binary(text.name))
            self.assertTrue(detector.is_binary(binary.name))

    def test_git_attribute_overrides(self):
        calls = []

        def fake_git(args):
            calls.append(tuple(args))
            return f"{args[-1]}: binary: set"

        detector = BinaryDetector(git_runner=fake_git, base_dir=Path.cwd())
        self.assertTrue(detector.is_binary("anything.txt"))
        self.assertTrue(calls)

        def fake_git_text(args):
            return f"{args[-1]}: binary: unspecified"

        text_detector = BinaryDetector(git_runner=fake_git_text, base_dir=Path.cwd())
        # File is missing but heuristics fall back to False
        self.assertFalse(text_detector.is_binary("missing.txt"))


class FilterTextFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _filter_text_files is None:
            raise unittest.SkipTest("Indexer dependencies unavailable")

    def test_filter_uses_detector(self):
        class StubDetector:
            def __init__(self):
                self.calls = []
            def is_binary(self, path):
                self.calls.append(path)
                return path.endswith(".bin")

        detector = StubDetector()
        paths = {"a.txt", "b.bin", "c.md"}
        filtered = _filter_text_files(paths, detector=detector)
        self.assertEqual(filtered, {"a.txt", "c.md"})
        self.assertCountEqual(detector.calls, sorted(paths))


if __name__ == "__main__":
    unittest.main()
