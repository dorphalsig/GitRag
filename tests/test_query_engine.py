"""Tests for grouped Tree-sitter query matches."""

import sys
import unittest
from pathlib import Path

from tree_sitter_language_pack import get_parser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import chunker  # type: ignore


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
        parser = get_parser("cpp")
        tree = parser.parse(source.encode("utf-8"))
        matches = chunker._query_matches(tree.root_node, "cpp", [query], "method")
        self.assertEqual(len(matches), 1)
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
