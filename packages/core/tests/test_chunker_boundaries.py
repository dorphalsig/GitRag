from pathlib import Path

from Chunker import chunker

FIXTURES = Path(__file__).parent / "fixtures"


def _chunks(name: str):
    return chunker.chunk_file(str(FIXTURES / name), "repo")


def test_python_executable_chunks_do_not_absorb_siblings():
    chunks = _chunks("fixture.py")
    exec_chunks = [c for c in chunks if c.metadata.get("chunk_kind") == "method"]
    assert exec_chunks, "Expected chunks with chunk_kind='method'"
    for chunk in exec_chunks:
        text = chunk.chunk.strip()
        assert text.startswith(("def ", "async def ", '"""', "#"))
        assert text.count("def ") <= 1


def test_java_method_chunk_stops_before_next_method():
    chunks = _chunks("fixture.java")
    exec_chunks = [c for c in chunks if c.metadata.get("chunk_kind") == "method"]
    assert exec_chunks, "Expected chunks with chunk_kind='method'"
    assert all(chunk.chunk.count("(") >= 1 for chunk in exec_chunks)
    assert all(chunk.chunk.count("class ") == 0 for chunk in exec_chunks)


def test_top_level_type_metadata_chunk_excludes_method_bodies():
    chunks = _chunks("fixture.java")
    metadata_chunks = [c for c in chunks if c.metadata.get("chunk_kind") == "class_metadata"]
    assert metadata_chunks, "Expected chunks with chunk_kind='class_metadata'"
    for chunk in metadata_chunks:
        assert "class" in chunk.chunk or "interface" in chunk.chunk or "record" in chunk.chunk or "enum" in chunk.chunk
        assert "return " not in chunk.chunk


def test_chunk_signatures_include_scope_information():
    chunks = _chunks("fixture.py")
    signatures = [chunk.signature for chunk in chunks]
    assert any(sig for sig in signatures)
    assert any("fixture." in sig or "repo|" in sig for sig in signatures)
