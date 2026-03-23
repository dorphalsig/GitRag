"""Tests for grouped Tree-sitter query matches."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Chunker import chunker


class TestQueryEngine(unittest.TestCase):
    """Validate grouped capture matches for scoped methods."""

    def test_query_matches_grouped_capture(self):
        """Ensure scoped identifiers preserve scope/name captures in grouped matches."""
        source = """
void Stack::push(int val) {
    if (top >= n - 1)
        cout << "Stack Overflow" << endl;
    else {
        top++;
        stack[top] = val;
    }
}
"""
        query = """
(function_definition
  declarator: (function_declarator
    declarator: (scoped_identifier
      scope: (type_identifier) @explicit_scope
      name: (identifier) @name
    )
  )
) @method
"""
        class _Node:
            def __init__(self, start: int, end: int, node_type: str = "", children: list["_Node"] | None = None):
                self.start_byte = start
                self.end_byte = end
                self.type = node_type
                self.named_children = children or []
                self.parent = None
                for child in self.named_children:
                    child.parent = self

            def child_by_field_name(self, name: str):
                return None

        class _Query:
            def matches(self, root):
                method = _Node(1, len(source.encode("utf-8")) - 1, "function_definition")
                scope = _Node(6, 11, "type_identifier")
                name = _Node(13, 17, "identifier")
                return [(0, {"method": [method], "explicit_scope": [scope], "name": [name]})]

        class _CompiledQuery:
            def __init__(self, lang, qsrc):
                self._query = _Query()

            def matches(self, root):
                return self._query.matches(root)

        fake_root = _Node(0, len(source.encode("utf-8")), "translation_unit")
        fake_lang = type("Lang", (), {"query": staticmethod(lambda _q: _Query())})()
        fake_tree = type("Tree", (), {"root_node": fake_root})()

        with mock.patch.object(chunker, "get_parser", return_value=type("Parser", (), {"parse": staticmethod(lambda _b: fake_tree)})()), \
             mock.patch.object(chunker, "get_language", return_value=fake_lang), \
             mock.patch.object(chunker, "Query", _CompiledQuery), \
             mock.patch.object(chunker, "QueryCursor", None):
            parser = chunker.get_parser("cpp")
            tree = parser.parse(source.encode("utf-8"))
            matches = chunker._query_matches(tree.root_node, "cpp", [query], "method")
        self.assertGreaterEqual(len(matches), 1)
        method_node, grouped = matches[0]
        self.assertIs(grouped["method"], method_node)
        self.assertEqual(_node_text(source, grouped["explicit_scope"]), "Stack")
        self.assertEqual(_node_text(source, grouped["name"]), "push")

    def test_fqn_prefers_explicit_scope(self):
        """Ensure explicit scope overrides parent-walked scopes."""
        contents = b"Stack.push"
        ctx = {"explicit_scope": _MockNode(0, 5)}
        node = _MockNode(6, 10, node_type="function")
        node.named_children = [_MockNode(6, 10, node_type="identifier")]
        result = chunker._fqn_for_node(node, contents, context=ctx)
        self.assertEqual(result, "Stack.push")


def _node_text(source: str, node) -> str:
    """Return node text from source bytes."""
    return source.encode("utf-8")[node.start_byte:node.end_byte].decode("utf-8")


class _MockNode:
    def __init__(self, start_byte: int, end_byte: int, node_type: str = "") -> None:
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.type = node_type
        self.parent = None
        self.named_children = []
