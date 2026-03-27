from pathlib import Path

from tree_sitter_language_pack import get_parser

from Chunker.chunker import GRAMMAR_QUERIES, _query_matches

FIXTURES = Path(__file__).parent / "fixtures"


def _parse_fixture(name: str, language: str):
    contents = (FIXTURES / name).read_bytes()
    parser = get_parser(language)
    tree = parser.parse(contents)
    return contents, tree.root_node


def _captured_texts(root, contents: bytes, language: str, category: str, capture: str):
    matches = _query_matches(root, language, GRAMMAR_QUERIES[category].get(language, []), capture)
    return [contents[node.start_byte:node.end_byte].decode("utf-8", errors="replace") for node, _ in matches]


def test_python_method_query_captures_real_function_boundaries():
    contents, root = _parse_fixture("fixture.py", "python")
    texts = _captured_texts(root, contents, "python", "Method", "method")
    assert any("def" in text for text in texts)
    assert all(text.lstrip().startswith(("def ", "async def ")) for text in texts)


def test_java_queries_capture_type_method_constructor_and_field_nodes():
    contents, root = _parse_fixture("fixture.java", "java")
    method_texts = _captured_texts(root, contents, "java", "Method", "method")
    ctor_texts = _captured_texts(root, contents, "java", "Constructor", "constructor")
    type_texts = _captured_texts(root, contents, "java", "Type", "type")
    field_texts = _captured_texts(root, contents, "java", "Field", "field")
    assert method_texts
    assert ctor_texts
    assert type_texts
    assert field_texts


def test_kotlin_accessor_query_captures_getter_or_setter_when_present():
    contents, root = _parse_fixture("fixture.kt", "kotlin")
    texts = _captured_texts(root, contents, "kotlin", "Accessor", "accessor")
    assert texts
    assert all("get" in text or "set" in text for text in texts)


def test_rust_type_and_field_queries_capture_structs_and_fields():
    contents, root = _parse_fixture("fixture.rs", "rust")
    type_texts = _captured_texts(root, contents, "rust", "Type", "type")
    field_texts = _captured_texts(root, contents, "rust", "Field", "field")
    assert type_texts
    assert field_texts
    assert any("struct" in t for t in type_texts)


def test_go_package_and_function_queries_capture_declared_units():
    contents, root = _parse_fixture("fixture.go", "go")
    packages = _captured_texts(root, contents, "go", "Package", "package")
    methods = _captured_texts(root, contents, "go", "Method", "method")
    assert packages
    assert methods
