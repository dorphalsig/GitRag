from Chunker.LineMapper import LineMapper
from Chunker.chunker import (
    DOC_HARD_CAP_BYTES,
    DocBlock,
    _build_doc_context,
    _chunk_plaintext,
    _extract_requirement_sentences,
    _find_requirement_sentences,
    _merge_small_segments,
    _parse_markdown_blocks,
    _split_block_if_needed,
)


def _ctx(text: str):
    data = text.encode()
    return _build_doc_context(data, LineMapper(data)), data, LineMapper(data)


def test_parse_markdown_blocks_detects_headings_fences_tables_and_lists():
    text = (
        "Title\n"
        "=====\n\n"
        "## Section\n\n"
        "```python\nprint('x')\n```\n\n"
        "|a|b|\n|1|2|\n\n"
        "- item one\n- item two\n\n"
        "> quote\n"
    )
    ctx, _, _ = _ctx(text)
    blocks = _parse_markdown_blocks(ctx)
    block_types = [block.type for block in blocks]
    assert "heading" in block_types
    assert "fence" in block_types
    assert "table" in block_types
    assert "list" in block_types
    assert "blockquote" in block_types
    headings = [block for block in blocks if block.type == "heading"]
    assert any(block.heading_title == "Title" for block in headings)
    assert any(block.heading_anchor == "section" for block in headings)


def test_split_block_if_needed_keeps_heading_and_splits_large_text():
    big_text = ("line\n" * (DOC_HARD_CAP_BYTES // 5 + 20))
    ctx, _, mapper = _ctx(big_text)
    heading = DocBlock(start_char=0, end_char=5, type="heading", heading_title="x", heading_anchor="x")
    assert _split_block_if_needed(ctx, heading, mapper) == [heading]

    block = DocBlock(start_char=0, end_char=len(ctx.text), type="paragraph")
    parts = _split_block_if_needed(ctx, block, mapper)
    assert len(parts) > 1
    assert parts[0].start_char == 0
    assert parts[-1].end_char == len(ctx.text)


def test_requirement_helpers_and_plaintext_chunking():
    text = "System shall log. System shall log. It should validate."
    reqs = _find_requirement_sentences(text)
    assert reqs == ["System shall log.", "It should validate."]

    ctx, data, mapper = _ctx(text)
    extracted = _extract_requirement_sentences(ctx, 0, len(ctx.text))
    assert extracted == reqs

    chunks = _chunk_plaintext(ctx, data, "doc.txt", "repo", mapper)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_kind"] == "text"
    assert len(chunks[0].metadata["requirement_sentences"]) == 2


def test_merge_small_segments_merges_adjacent_small_segments():
    ctx, _, _ = _ctx("abc\ndef\nghi\n")
    seg1 = {"start_char": 0, "end_char": 4, "blocks": [1], "block_types": {"text"}, "fence_langs": set()}
    seg2 = {"start_char": 4, "end_char": 8, "blocks": [2], "block_types": {"text"}, "fence_langs": set()}
    merged = _merge_small_segments(ctx, [seg1, seg2])
    assert len(merged) == 1
    assert merged[0]["blocks"] == [1, 2]


def test_parse_markdown_blocks_detects_headings_fences_tables_lists_and_blockquotes():
    text = (
        "Title\n"
        "=====\n\n"
        "## Section\n\n"
        "```python\nprint('x')\n```\n\n"
        "|a|b|\n|1|2|\n\n"
        "- item one\n- item two\n\n"
        "> quote\n"
    )
    ctx, _, _ = _ctx(text)
    blocks = _parse_markdown_blocks(ctx)
    assert {block.type for block in blocks} >= {"heading", "fence", "table", "list", "blockquote"}
