from pathlib import Path

import pytest

from Chunker import chunker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("name", "expected_language", "snippet"),
    [
        ("fixture.txt", "text", "shall"),
        ("fixture.tsv", "tsv", "requirement"),
        ("fixture.yaml", "yaml", "audit logging"),
        ("fixture.toml", "toml", "database"),
        ("fixture.xml", "xml", "<"),
        ("fixture.html", "html", "html"),
        ("fixture.css", "css", "color"),
        ("fixture.jsonl", "jsonl", "line 1"),
    ],
)
def test_non_code_fixtures_chunk_with_real_parsers(name, expected_language, snippet):
    chunks = chunker.chunk_file(str(FIXTURES / name), "repo")
    assert chunks
    assert all(chunk.language == expected_language for chunk in chunks)
    combined = "\n".join(chunk.chunk for chunk in chunks)
    signatures = "\n".join(chunk.signature for chunk in chunks)
    assert snippet.lower() in (combined + "\n" + signatures).lower()


@pytest.mark.parametrize(
    "name",
    [
        "fixture.kt", "fixture.go", "fixture.rs", "fixture.ts", "fixture.js",
        "fixture.dart", "fixture.cs", "fixture.cpp", "fixture.c", "fixture.php",
        "fixture.rb", "fixture.swift", "fixture.pas",
    ],
)
def test_code_fixtures_chunk_with_real_parsers(name):
    chunks = chunker.chunk_file(str(FIXTURES / name), "repo")
    assert chunks
    assert any(chunk.chunk for chunk in chunks)
