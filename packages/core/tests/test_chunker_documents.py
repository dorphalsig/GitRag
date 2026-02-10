import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.fixtures import ensure_fixtures  # type: ignore


class DocumentChunkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_fixtures()
        cls._ensure_tree_sitter_stubs()
        import chunker  # type: ignore

        cls.chunker = chunker
        cls.markdown_fixture = ROOT / "tests" / "fixtures" / "Markdown.md"

    @staticmethod
    def _ensure_tree_sitter_stubs() -> None:
        try:
            import tree_sitter  # type: ignore # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("tree_sitter")

            class Node:  # type: ignore
                pass

            class Query:  # type: ignore
                def __init__(self, *a, **k):
                    pass

                def captures(self, *a, **k):
                    return {}

            class QueryCursor:  # type: ignore
                def __init__(self, *a, **k):
                    self._captures = []

                def exec(self, *a, **k):
                    self._captures = []

                def captures(self, *a, **k):
                    return []

            class QueryError(Exception):
                pass

            stub.Node = Node
            stub.Query = Query
            stub.QueryCursor = QueryCursor
            stub.QueryError = QueryError
            sys.modules["tree_sitter"] = stub

        try:
            import tree_sitter_language_pack  # type: ignore # noqa: F401
        except ModuleNotFoundError:
            stub_lp = types.ModuleType("tree_sitter_language_pack")

            def _missing(*_a, **_k):
                raise RuntimeError("Tree-sitter not available for tests")

            stub_lp.get_parser = _missing  # type: ignore[attr-defined]
            stub_lp.get_language = _missing  # type: ignore[attr-defined]
            sys.modules["tree_sitter_language_pack"] = stub_lp

    def test_markdown_chunks_include_metadata(self) -> None:
        chunks = self.chunker.chunk_file(str(self.markdown_fixture), repo="docs")
        self.assertTrue(chunks, "expected markdown fixture to produce chunks")

        grammar_version = self.chunker.DOC_GRAMMAR_VERSION
        doc_chunks = [c for c in chunks if c.metadata.get("grammar_version") == grammar_version]
        self.assertTrue(doc_chunks, "markdown chunks should include doc grammar metadata")

        # All doc chunks should expose overlap metadata and breadcrumb/anchors lists
        for ch in doc_chunks:
            self.assertIn("overlap_bytes", ch.metadata)
            self.assertEqual(ch.metadata.get("grammar_version"), grammar_version)

        self.assertTrue(
            any(ch.metadata.get("heading_breadcrumb") for ch in doc_chunks),
            "at least one chunk should capture heading breadcrumbs",
        )

        self.assertTrue(
            any(ch.metadata.get("chunk_kind") == "fence" for ch in doc_chunks),
            "markdown code fences should yield fence chunks",
        )

        self.assertTrue(
            any("fence:ts" in ch.metadata.get("code_refs", []) for ch in doc_chunks),
            "code fence language tags should appear in code_refs",
        )

        requirement_chunks = [
            ch for ch in doc_chunks if ch.metadata.get("requirement_sentences")
        ]
        self.assertTrue(requirement_chunks, "must capture requirement sentences in markdown")
        self.assertTrue(
            any("must" in sentence.lower() for ch in requirement_chunks for sentence in ch.metadata["requirement_sentences"]),
            "markdown requirements should include 'must' sentence",
        )

    def test_json_chunks_track_breadcrumbs_and_requirements(self) -> None:
        payload = textwrap.dedent(
            """
            {
              "alpha": {
                "requirement": "The system shall log events for auditing."
              },
              "beta": [1, 2, 3]
            }
            """
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            chunks = self.chunker.chunk_file(str(temp_path), repo="docs")
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertTrue(chunks, "json document should yield chunks")
        self.assertTrue(
            all(ch.metadata.get("chunk_kind") == "json" for ch in chunks),
            "json chunks should be tagged as json",
        )
        self.assertTrue(
            any("alpha" in " / ".join(ch.metadata.get("heading_breadcrumb", [])) for ch in chunks),
            "top-level key names should appear in breadcrumbs",
        )
        self.assertTrue(
            any(
                "shall" in sentence.lower()
                for ch in chunks
                for sentence in ch.metadata.get("requirement_sentences", [])
            ),
            "json requirements should be detected",
        )

    def test_csv_chunks_note_delimiter(self) -> None:
        csv_content = "id,name\n1,Alice\n2,Bob\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write(csv_content)
            temp_path = Path(handle.name)
        try:
            chunks = self.chunker.chunk_file(str(temp_path), repo="docs")
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertTrue(chunks)
        meta = chunks[0].metadata
        self.assertEqual(meta.get("chunk_kind"), "table")
        self.assertEqual(meta.get("table_delimiter"), ",")
        self.assertEqual(chunks[0].language, "csv")

    def test_plaintext_bytes_fallback(self) -> None:
        data = b"\xff\xfe requirement must pass"
        with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        try:
            chunks = self.chunker.chunk_file(str(temp_path), repo="docs")
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertTrue(chunks)
        chunk = chunks[0]
        self.assertEqual(chunk.language, "text")
        kind = chunk.metadata.get("chunk_kind")
        self.assertIn(kind, {"text", "req"})
        self.assertIn("eol", chunk.metadata)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
