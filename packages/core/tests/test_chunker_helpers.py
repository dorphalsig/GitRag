from array import array

from Chunker.LineMapper import LineMapper
from Chunker.chunker import (
    _build_doc_context,
    _chunk_plaintext_bytes,
    _document_signature,
    _json_spans,
    _make_chunk,
    _newline_aligned_ranges,
    _slugify,
)


def test_slugify_normalizes_text():
    assert _slugify(" Hello, World! ") == "hello-world"
    assert _slugify("A  B---C") == "a-b-c"


def test_document_signature_truncates_preview_and_adds_version():
    sig = _document_signature(
        "repo",
        "path.md",
        0,
        10,
        ["Heading", "Section"],
        "x" * 100,
    )
    assert sig.startswith("repo|path.md|0-10|Heading/Section|")
    assert sig.endswith("doc-chunker-v1")
    assert "..." in sig


def test_build_doc_context_handles_ascii_and_utf8():
    ascii_bytes = b"a\nline\n"
    ascii_ctx = _build_doc_context(ascii_bytes, LineMapper(ascii_bytes))
    assert ascii_ctx is not None
    assert ascii_ctx.char_to_byte[:4] == [0, 1, 2, 3]
    assert ascii_ctx.eol == "\n"

    utf8_bytes = "å\nβ\n".encode("utf-8")
    utf8_ctx = _build_doc_context(utf8_bytes, LineMapper(utf8_bytes))
    assert utf8_ctx is not None
    assert utf8_ctx.char_to_byte[0] == 0
    assert utf8_ctx.char_to_byte[1] == len("å".encode("utf-8"))


def test_build_doc_context_returns_none_for_invalid_utf8():
    assert _build_doc_context(b"\xff\xfe", LineMapper(b"\xff\xfe")) is None


def test_json_spans_for_jsonl_and_array_and_invalid():
    jsonl = b'{"a":1}\n\n{"b":2}\n'
    ctx = _build_doc_context(jsonl, LineMapper(jsonl))
    spans = _json_spans(ctx, "jsonl")
    assert len(spans) == 2
    assert spans[0][2] == ["line 1"]
    assert spans[1][2] == ["line 3"]

    arr = b'[1, 2, 3]'
    ctx2 = _build_doc_context(arr, LineMapper(arr))
    spans2 = _json_spans(ctx2, "json")
    assert len(spans2) >= 1

    invalid = b'{invalid json}'
    ctx3 = _build_doc_context(invalid, LineMapper(invalid))
    spans3 = _json_spans(ctx3, "json")
    assert spans3 == [(0, 1, [])]


def test_newline_aligned_ranges_with_overlap_and_empty_range():
    content = ("line\n" * 100).encode()
    mapper = LineMapper(content)
    ranges = _newline_aligned_ranges(content, 0, len(content), mapper, soft_max=20, hard_cap=30, overlap=5)
    assert ranges
    assert ranges[0][0] == 0
    assert all(start < end for start, end in ranges)
    assert _newline_aligned_ranges(content, 5, 5, mapper) == []


def test_chunk_plaintext_bytes_extracts_requirement_metadata():
    text = (
        "The system shall log events for auditing.\n"
        "Use `foo()` when needed.\n"
        "Additional context here.\n"
    ).encode()
    mapper = LineMapper(text)
    chunks = _chunk_plaintext_bytes(text, "doc.txt", "text", "repo", mapper)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.metadata["chunk_kind"] == "req"
    assert any("shall" in sentence.lower() for sentence in chunk.metadata["requirement_sentences"])
    assert "foo()" in chunk.metadata["code_refs"]


def test_make_chunk_maps_bytes_to_points():
    content = b"abc\ndef\n"
    mapper = LineMapper(content)
    chunk = _make_chunk(content, 4, 7, "file.txt", "text", "repo", mapper, signature="sig", metadata={"k": "v"})
    assert chunk.start_rc == (1, 0)
    assert chunk.end_rc == (1, 3)
    assert chunk.chunk == "def"
    assert chunk.signature == "sig"
    assert chunk.metadata == {"k": "v"}


def test_chunk_plaintext_bytes_marks_requirement_chunks():
    text = (
        "The system shall log events for auditing.\n"
        "Use `foo()` when needed.\n"
        "Additional context here.\n"
    ).encode()
    mapper = LineMapper(text)
    chunks = _chunk_plaintext_bytes(text, "doc.txt", "text", "repo", mapper)
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_kind"] == "req"
    assert chunks[0].metadata["requirement_sentences"]
