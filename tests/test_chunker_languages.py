import json
import unittest
import sys
from pathlib import Path

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Real Tree-sitter must be available for these tests
try:
    from tree_sitter_language_pack import get_parser
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    import pytest

    pytest.skip("tree_sitter_language_pack not installed", allow_module_level=True)

import chunker  # type: ignore

from tests.fixtures import ensure_fixtures, load_bytes


CODE_SAMPLES = [
    ("Java.java", "java"),
    ("Kotlin.kt", "kotlin"),
    ("Dart.dart", "dart"),
    ("Pascal.pas", "pascal"),
]


class TestChunkerWithTreeSitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fixtures()
        cls.config_path = Path(chunker.__file__).with_name("grammar_queries.json")
        with cls.config_path.open("r", encoding="utf-8") as handle:
            cls.grammar_config = json.load(handle)

    def _parse(self, lang: str, data: bytes):
        parser = get_parser(lang)
        return parser.parse(data)

    def _exec_nodes(self, lang: str, root):
        # Collect executable-like nodes for the given language using real queries
        caps = {
            "Method": "method",
            "Constructor": "constructor",
            "Initializer": "initializer",
            "Accessor": "accessor",
        }
        nodes = []
        for cat, cap in caps.items():
            sexprs = chunker.GRAMMAR_QUERIES[cat].get(lang, [])
            if not sexprs:
                continue
            nodes.extend(chunker._query_nodes(lang, root, sexprs, cap))
        # De-dup and sort by source order
        uniq = {}
        for n in nodes:
            uniq[(n.start_byte, n.end_byte, n.type)] = n
        out = list(uniq.values())
        out.sort(key=lambda n: (n.start_byte, n.end_byte))
        return out

    def _type_nodes(self, lang: str, root):
        sexprs = chunker.GRAMMAR_QUERIES["Type"].get(lang, [])
        if not sexprs:
            return []
        nodes = chunker._query_nodes(lang, root, sexprs, "type")
        # filter to top-level only
        return [n for n in nodes if chunker._is_top_level_type(n, lang)]

    def test_code_languages_chunks_align_with_tree_sitter_nodes(self):
        for fname, lang in CODE_SAMPLES:
            with self.subTest(language=lang, file=fname):
                p = ROOT / "tests" / "fixtures" / fname
                data = p.read_bytes()

                # Tree-sitter parsing
                tree = self._parse(lang, data)
                root = tree.root_node

                # Run chunker
                chunks = chunker.chunk_file(str(p), repo="TEST")
                self.assertTrue(chunks, "chunker returned no chunks")

                # All chunks should be tagged with the expected language
                self.assertTrue(all(c.language == lang for c in chunks))

                # Validate type metadata chunks: one per top-level type, matching exact span
                type_nodes = self._type_nodes(lang, root)
                meta_chunks = [c for c in chunks if c.signature.endswith("#metadata")]

                for t in type_nodes:
                    # Find a metadata chunk that exactly matches the type node span
                    matches = [c for c in meta_chunks if c.start_bytes == t.start_byte and c.end_bytes == t.end_byte]
                    self.assertTrue(matches, f"No metadata chunk matching top-level type at {t.start_byte}:{t.end_byte}")

                # Validate executable units (methods/ctors/inits/accessors)
                exec_nodes = self._exec_nodes(lang, root)
                for n in exec_nodes:
                    doc_start = chunker._leading_trivia_start(data, n)
                    # Gather non-metadata chunks that fall within the method span
                    method_chunks = [c for c in chunks if not c.signature.endswith("#metadata") and c.start_bytes >= doc_start and c.end_bytes <= n.end_byte]
                    self.assertTrue(method_chunks, f"No method-like chunks found for node {n.type} at {n.start_byte}:{n.end_byte}")

                    # The first chunk should start at the leading trivia start
                    first = min(method_chunks, key=lambda c: (c.start_bytes, c.end_bytes))
                    self.assertEqual(first.start_bytes, doc_start, f"First chunk for {n.type} should start at leading trivia")

                    # If unit size is small, it should end exactly at node end
                    unit_len = n.end_byte - doc_start
                    if unit_len <= chunker.HARD_CAP_BYTES:
                        # Find any chunk that starts at doc_start; should end at node end
                        at_doc = [c for c in method_chunks if c.start_bytes == doc_start]
                        self.assertTrue(at_doc, "Expected at least one chunk starting at doc_start")
                        # choose the one with max end
                        ender = max(at_doc, key=lambda c: c.end_bytes)
                        self.assertEqual(ender.end_bytes, n.end_byte, "Small unit should extend to node end")

                    # Check signature container markers (member vs top_level)
                    encl = chunker._enclosing_types(n, data)
                    if encl:
                        expected_suffix = f"|member|{lang}"
                    else:
                        expected_suffix = f"|top_level|{lang}"
                    # At least one chunk for this node should have the expected suffix
                    self.assertTrue(any(c.signature.endswith(expected_suffix) for c in method_chunks),
                                    f"No chunk had expected signature suffix {expected_suffix}")

    def test_json_configuration_matches_chunker_constants(self):
        data = self.grammar_config
        self.assertEqual(data["code_extensions"], chunker.CODE_EXTENSIONS)
        self.assertEqual(data["noncode_ts_grammar"], chunker.NONCODE_TS_GRAMMAR)
        self.assertEqual(data["grammar_queries"], chunker.GRAMMAR_QUERIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
