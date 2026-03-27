from pathlib import Path

import pytest

from Chunker import chunker

FIXTURES = Path(__file__).parent / "fixtures"


def test_chunk_markdown_fixture_produces_markdown_chunk():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.md"), "repo")
    assert len(chunks) >= 1
    assert all(chunk.language == "markdown" for chunk in chunks)
    assert any("Markdown Fixture Document" in chunk.signature for chunk in chunks)
    assert any("Introduction" in chunk.chunk for chunk in chunks)


def test_chunk_json_fixture_produces_json_chunk_with_breadcrumb():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.json"), "repo")
    assert len(chunks) >= 1
    assert all(chunk.language == "json" for chunk in chunks)
    assert any("metadata" in chunk.signature for chunk in chunks)
    assert any("version" in chunk.chunk for chunk in chunks)


def test_chunk_csv_fixture_produces_csv_chunk():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.csv"), "repo")
    assert len(chunks) >= 1
    assert all(chunk.language == "csv" for chunk in chunks)
    assert any("id,name,role,requirement" in chunk.signature for chunk in chunks)
    assert any("The system shall log events for auditing" in chunk.chunk for chunk in chunks)


def test_chunk_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert chunker.chunk_file(str(path), "repo") == []


def test_chunk_unsupported_extension_falls_back_to_document(tmp_path):
    path = tmp_path / "notes.unknown"
    path.write_text("hello\nworld\n")
    chunks = chunker.chunk_file(str(path), "repo")
    assert len(chunks) >= 1
    assert any(chunk.language == "text" for chunk in chunks)
    assert any("hello" in chunk.chunk for chunk in chunks)


def test_chunk_python_fixture_returns_chunks():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.py"), "repo")
    assert chunks
    assert any(chunk.language == "python" for chunk in chunks)


def test_chunk_java_fixture_returns_chunks():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.java"), "repo")
    assert chunks
    assert any(chunk.language == "java" for chunk in chunks)


def test_chunk_markdown_fixture_keeps_heading_context():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.md"), "repo")
    assert chunks
    assert any(chunk.metadata.get("heading_breadcrumb") for chunk in chunks)
    assert any("Introduction" in chunk.chunk for chunk in chunks)


def test_chunk_jsonl_fixture_emits_line_based_signatures():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.jsonl"), "repo")
    assert chunks
    assert all(chunk.language == "jsonl" for chunk in chunks)
    assert any("line 1" in chunk.signature.lower() for chunk in chunks)


def test_chunk_tsv_fixture_preserves_header_in_signature():
    chunks = chunker.chunk_file(str(FIXTURES / "fixture.tsv"), "repo")
    assert chunks
    assert any("requirement" in chunk.signature.lower() for chunk in chunks)
