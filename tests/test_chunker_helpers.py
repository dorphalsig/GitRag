# Pure-helper tests for:
#   _newline_aligned_ranges, _nearest_newline, _byte_to_point, _has_blank_line_between
# No Tree-sitter needed at runtime; we stub modules only to import chunker.

import sys
import types
import unittest
import random
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# ---------- Make 'src' importable -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------- Lightweight stubs so import chunker works without native deps ----------
# Prefer the real tree_sitter if available; otherwise provide minimal stubs
try:
    import tree_sitter  # type: ignore
except Exception:
    _ts = types.ModuleType("tree_sitter")
    class Node: ...
    class Query: ...
    class QueryCursor:
        def __init__(self, *a, **k): ...
        def captures(self, *a, **k): return {}
    class QueryError(Exception): ...
    _ts.Node = Node; _ts.Query = Query; _ts.QueryCursor = QueryCursor; _ts.QueryError = QueryError
    sys.modules["tree_sitter"] = _ts

# Prefer the real language pack if available; otherwise stub just enough for helpers
try:
    import tree_sitter_language_pack  # type: ignore
except Exception:
    _lp = types.ModuleType("tree_sitter_language_pack")
    def get_parser(*a, **k): raise RuntimeError("Tree-sitter not available for tests")
    def get_language(*a, **k): raise RuntimeError("Tree-sitter not available for tests")
    _lp.get_parser = get_parser; _lp.get_language = get_language
    sys.modules["tree_sitter_language_pack"] = _lp

# ---------- System under test -----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _import_chunker():
    """
    Try, in order:
    1) Regular `import chunker`
    2) Dynamically load the first src/**/chunker.py found
    """
    try:
        import chunker as m  # type: ignore
        return m
    except ImportError as exc:
        for p in SRC.rglob("chunker.py"):
            spec = spec_from_file_location("chunker", p)
            if spec and spec.loader:
                mod = module_from_spec(spec)
                sys.modules["chunker"] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod
        raise ImportError(f"Could not locate chunker.py under {SRC}") from exc

chunker = _import_chunker()


# ---------- Fixture helpers -------------------------------------------------------
from .fixtures import ensure_fixtures as _ensure_fixtures, load_bytes as _load_bytes


# ---------- Assertion mixin -------------------------------------------------------
class InvariantsMixin(unittest.TestCase):
    def assertCoverageNoGaps(self, ranges, start: int, end: int, overlap: int, hard_cap: int):
        """Verify full coverage [start,end), no gaps, overlap ≤ requested, and len ≤ hard_cap."""
        self.assertTrue(ranges, "no ranges returned")
        self.assertEqual(ranges[0][0], start)
        self.assertEqual(ranges[-1][1], end)
        ps, pe = ranges[0]
        self.assertLess(ps, pe)
        self.assertLessEqual(pe, end)
        self.assertLessEqual(pe - ps, hard_cap)
        for s, e in ranges[1:]:
            self.assertLessEqual(s, pe, f"gap detected: {pe}..{s}")
            ov = pe - s
            self.assertGreaterEqual(ov, 0, "negative overlap (gap)")
            self.assertLessEqual(ov, overlap, f"overlap {ov} exceeds {overlap}")
            self.assertLess(s, e)
            self.assertLessEqual(e, end)
            self.assertLessEqual(e - s, hard_cap)
            ps, pe = s, e


# ---------- Tests -----------------------------------------------------------------
class TestHelpers(InvariantsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load constants and ensure fixtures exist."""
        _ensure_fixtures()
        cls.HARD_CAP = int(chunker.HARD_CAP_BYTES)
        cls.SOFT_MAX = int(chunker.SOFT_MAX_BYTES)
        cls.NEWLINE_WINDOW = int(chunker.NEWLINE_WINDOW)
        cls.newline_aligned = staticmethod(chunker._newline_aligned_ranges)
        cls.nearest_newline = staticmethod(chunker._nearest_newline)
        cls.byte_to_point = staticmethod(chunker._byte_to_point)
        cls.has_blank_line_between = staticmethod(chunker._has_blank_line_between)

    def test_no_newlines_overlap0(self):
        """No-newline files split by size without gaps or overlaps."""
        for fname in ("no_newlines_small.txt", "no_newlines_large.txt"):
            data = _load_bytes(fname)
            ranges = self.newline_aligned(data, 0, len(data), overlap=0)
            self.assertCoverageNoGaps(ranges, 0, len(data), overlap=0, hard_cap=self.HARD_CAP)

    def test_crlf_and_nearest_newline_behavior(self):
        """CRLF files: nearest_newline returns an index at/near LF within the window; coverage holds."""
        data = _load_bytes("crlf.txt")
        start, end = 0, len(data)
        target = min(start + self.SOFT_MAX, end - 1)
        lo = max(start + 1, target - self.NEWLINE_WINDOW)
        hi = min(end - 1, target + self.NEWLINE_WINDOW)
        idx = self.nearest_newline(data, target, lo, hi)
        if idx is not None:
            # Accept either pointing at '\n' or immediately after it (implementation detail tolerant).
            near_lf = (data[idx:idx+1] == b"\n") or (data[idx-1:idx] == b"\n")
            self.assertTrue(near_lf and (lo <= idx <= hi + 1))
        ranges = self.newline_aligned(data, 0, len(data), overlap=64)
        self.assertCoverageNoGaps(ranges, 0, len(data), overlap=64, hard_cap=self.HARD_CAP)

    def test_tiny_file_single_range(self):
        """Tiny files (≤ a few bytes) should yield a single exact range."""
        data = _load_bytes("tiny.txt")
        ranges = self.newline_aligned(data, 0, len(data), overlap=0)
        self.assertEqual(ranges, [(0, len(data))])

    def test_huge_file_enforces_hard_cap(self):
        """Large file: every segment length must be ≤ HARD_CAP_BYTES."""
        data = _load_bytes("huge_random.txt")
        ranges = self.newline_aligned(data, 0, len(data), overlap=0)
        for s, e in ranges:
            self.assertLessEqual(e - s, self.HARD_CAP)

    def test_overlap_invariant_randomized(self):
        """Randomized: no gaps; overlap respected; full coverage; size ≤ HARD_CAP."""
        rnd = random.Random(1337)
        for _ in range(50):
            n = rnd.randint(1, 250_000)
            rate = rnd.random() * 0.05
            bs = bytearray()
            for _i in range(n):
                bs.append(rnd.choice(b"abcd efghijklmnopqrstuvwxyz"))
                if rnd.random() < rate:
                    bs.append(0x0A)  # '\n'
            data = bytes(bs)
            overlap = rnd.randint(0, min(1024, self.SOFT_MAX // 2))
            ranges = self.newline_aligned(data, 0, len(data), overlap=overlap)
            self.assertCoverageNoGaps(ranges, 0, len(data), overlap=overlap, hard_cap=self.HARD_CAP)

    @staticmethod
    def _naive_byte_to_point(data: bytes, index: int):
        """Naively compute (row, col) by scanning from start; reference for assertions."""
        row = data.count(b"\n", 0, index)
        last = data.rfind(b"\n", 0, index)
        col = index if last == -1 else index - (last + 1)
        return row, col

    def test_byte_to_point_matches_naive(self):
        """Compare helper with a naive reference across fixed and random indices."""
        data = _load_bytes("huge_random.txt")
        for idx in (0, len(data)//4, len(data)//2, len(data) - 1):
            self.assertEqual(self.byte_to_point(data, idx), self._naive_byte_to_point(data, idx))
        rnd = random.Random(9)
        for _ in range(100):
            i = rnd.randint(0, len(data) - 1)
            self.assertEqual(self.byte_to_point(data, i), self._naive_byte_to_point(data, i))

    def test_has_blank_line_between_variants(self):
        """Detect blank lines with LF and CRLF, including whitespace-only lines."""
        a = b"line1\n"
        between1 = b"\n"            # LF blank
        between2 = b"\r\n \t \r\n"  # CRLF with spaces
        b = b"line2"
        data1 = a + between1 + b
        data2 = a + between2 + b
        self.assertTrue(self.has_blank_line_between(data1, len(a) - 1, len(a) + len(between1) + 1))
        self.assertTrue(self.has_blank_line_between(data2, len(a) - 1, len(a) + len(between2) + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
