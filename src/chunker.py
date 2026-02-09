# STRICT SCOPE (updated per user fixes):
# - Use Tree-sitter (tree_sitter_language_pack) for EVERYTHING: code (Kotlin/Java/Dart)
#   and non-code (Markdown, JSON, YAML, XML, TOML, etc.) when size > threshold.
# - Parse RAW BYTES for all Tree-sitter parsing; compute all offsets/lines on BYTES.
#   Decode ONLY AFTER slicing the bytes for any human-facing text.
# - Non-code rule still applies:
#     * ≤ SOFT_MAX_BYTES  -> emit a single whole-file chunk (with provenance header + redaction)
#     *  > SOFT_MAX_BYTES -> chunk by top-level element using Tree-sitter (NO overlap).
# - Code fallback ONLY IF Tree-sitter fails: fixed-size chunks with 10% overlap. All (chunk+overlap) nudged to the closest newline,
# - Primary unit (code): method chunks (leading doc + signature + full body or expr body).
# - Supplemental (code): class_metadata (top-level types only; NO bodies).
# - Heuristic FQDN: package (no trailing ';') + enclosing types + name (heuristic).
# - Methods ≤ 40 LOC; methods ≥ 10 LOC have docstrings. DRY + SOLID + NO SCOPE CREEP.

from __future__ import annotations

import bisect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict, Any

from tree_sitter import Node
from tree_sitter_language_pack import get_parser, get_language
from Chunk import Chunk

logger = logging.getLogger(__name__)

# ---------------- Constants & Config ----------------
# Internal constants (no knobs).
SOFT_MAX_BYTES = 16_384  # packing target
HARD_CAP_BYTES = 24_576  # absolute per-chunk limit
NEWLINE_WINDOW = 2_048  # cut nudge window

FALLBACK_OVERLAP_RATIO = 0.10

_ROOT_ATOM_RE = re.compile(r"\(+\s*([A-Za-z_]\w*)\b")
_MD_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_SETEXT_RE = re.compile(r"^([=-]{3,})\s*$")
_RST_SETEXT_RE = re.compile(r"^([=\-~`:'\"^_*+#<>]{3,})\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE_RE = re.compile(r"^(```+|~~~+)(.*)$")
_REQUIREMENT_PATTERN = re.compile(r"\b(shall|must|should|required|use case|scenario)\b", re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _slugify(text: str) -> str:
    """Best-effort slug for heading anchors."""
    lowered = text.strip().lower()
    sanitized = re.sub(r"[^a-z0-9\s-]", "", lowered)
    collapsed = re.sub(r"[\s-]+", "-", sanitized).strip("-")
    return collapsed


@dataclass
class DocContext:
    text: str
    char_to_byte: List[int]
    lines: List[str]
    line_char_offsets: List[int]
    total_bytes: int
    eol: str


@dataclass
class DocBlock:
    start_char: int
    end_char: int
    type: str
    heading_level: Optional[int] = None
    heading_title: str = ""
    heading_anchor: str = ""
    fence_lang: Optional[str] = None


class LineMapper:
    """Map byte offsets to (row, col) using a newline index."""

    def __init__(self, contents: bytes) -> None:
        self.newlines = [i for i, byte in enumerate(contents) if byte == 10]

    def byte_to_point(self, offset: int) -> Tuple[int, int]:
        if offset <= 0:
            return (0, 0)

        idx = bisect.bisect_left(self.newlines, offset)
        if idx == 0:
            return (0, offset)

        prev_newline = self.newlines[idx - 1]
        return (idx, offset - prev_newline - 1)


def _clone_block(block: DocBlock, start_char: int, end_char: int) -> DocBlock:
    return DocBlock(
        start_char=start_char,
        end_char=end_char,
        type=block.type,
        heading_level=block.heading_level,
        heading_title=block.heading_title,
        heading_anchor=block.heading_anchor,
        fence_lang=block.fence_lang,
    )


def _load_grammar_config() -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, list[str]]]]:
    """Load extensions and Tree-sitter query patterns from JSON configuration."""
    cfg_path = Path(__file__).with_name("grammar_queries.json")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    code_ext = {k: str(v) for k, v in data.get("code_extensions", {}).items()}
    noncode_ext = {k: str(v) for k, v in data.get("noncode_ts_grammar", {}).items()}
    queries_raw = data.get("grammar_queries", {})
    grammar_queries: dict[str, dict[str, list[str]]] = {}
    for category, lang_map in queries_raw.items():
        grammar_queries[category] = {lang: list(patterns) for lang, patterns in lang_map.items()}
    return code_ext, noncode_ext, grammar_queries


CODE_EXTENSIONS, NONCODE_TS_GRAMMAR, GRAMMAR_QUERIES = _load_grammar_config()

# Document chunking parameters
DOC_SOFT_MAX_BYTES = 8_192
DOC_HARD_CAP_BYTES = 16_384
DOC_OVERLAP_BYTES = 256
DOC_MIN_CHUNK_BYTES = 2_048
DOC_GRAMMAR_VERSION = "doc-chunker-v1"

DOC_MARKDOWN_EXTS = {"md", "markdown", "rst"}
DOC_JSON_EXTS = {"json", "jsonl", "ndjson"}
DOC_YAML_EXTS = {"yaml", "yml"}
DOC_TOML_EXTS = {"toml"}
DOC_CSV_EXTS = {"csv"}
DOC_TSV_EXTS = {"tsv"}
def chunk_file(path: str, repo: str) -> List[Chunk]:
    path_obj = Path(path)
    file_extension = path_obj.suffix.lower().lstrip(".")
    contents = path_obj.read_bytes()
    mapper = LineMapper(contents)

    if lang := CODE_EXTENSIONS.get(file_extension, None):
        chunks = _chunk_code(contents, path, lang, repo, mapper)
    elif file_extension in (
        DOC_MARKDOWN_EXTS | DOC_JSON_EXTS | DOC_YAML_EXTS | DOC_TOML_EXTS | DOC_CSV_EXTS | DOC_TSV_EXTS
    ):
        chunks = _chunk_document(contents, path, file_extension, repo, mapper)
    elif lang := NONCODE_TS_GRAMMAR.get(file_extension, None):
        chunks = _chunk_non_code(contents, path, lang, repo, mapper)
    else:
        chunks = _chunk_document(contents, path, file_extension, repo, mapper)

    return chunks


def _chunk_non_code(
    contents: bytes, path: str, language: str, repo: str, mapper: LineMapper
) -> List[Chunk]:
    """
    Chunk a non-code file using Tree-sitter:
      1) Choose a significant base level (root or its dominant single child).
      2) Pack first-level siblings into contiguous chunks ≤ SOFT_MAX_BYTES (no overlap).
      3) If a single sibling > SOFT_MAX_BYTES: one child-level pass; if still too large or childless,
         split by size nudged to newlines.
      4) Preserve leading/trailing bytes by stitching gaps to ensure full coverage.
    """
    if not contents:
        raise ValueError("contents must be non-empty bytes")

    try:
        parser = get_parser(language)
        tree = parser.parse(contents)
        root = tree.root_node
    except Exception as exc:
        logger.warning(
            "Failed to parse %s as %s: %s. Falling back to text.", path, language, exc
        )
        return _chunk_plaintext_bytes(contents, path, "text", repo, mapper)

    # Whole-file small case: emit a single chunk from the **original contents** and stop.
    if (root.end_byte - root.start_byte) < SOFT_MAX_BYTES:
        return [_make_chunk(contents, root.start_byte, root.end_byte, path, language, repo, mapper)]

    base = _first_multi_child_level(root)
    siblings = list(base.named_children)
    if not siblings:
        return _chunk_fallback(contents, path, language, repo, mapper)

    groups: List[Tuple[int, int]] = _pack_siblings(contents, siblings)
    if not groups:
        return _chunk_fallback(contents, path, language, repo, mapper)

    return [
        _make_chunk(contents, s, e, path, language, repo, mapper)
        for (s, e) in _stitch_full_coverage(contents, groups)
    ]


def _stitch_full_coverage(contents: bytes, groups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Expand grouped sibling ranges to cover the entire file by inserting ranges for
    leading bytes, inter-group gaps, and trailing bytes. Gap ranges are split with
    `_split_bytes_with_newline_nudge` to respect SOFT/HARD limits.
    """
    result: List[Tuple[int, int]] = []
    cursor = 0
    for s, e in groups:
        if s > cursor:
            result.extend(_newline_aligned_ranges(contents, cursor, s))
        result.append((s, e))
        cursor = e
    if cursor < len(contents):
        result.extend(_newline_aligned_ranges(contents, cursor, len(contents)))
    return result


def _chunk_fallback(
    contents: bytes, path: str, language: str, repo: str, mapper: LineMapper
) -> List[Chunk]:
    """
    Fallback chunking without Tree-sitter:
      - Split by size with a soft step of SOFT_MAX_BYTES.
      - For each boundary, nudge to the nearest newline within NEWLINE_WINDOW bytes.
      - Guarantee no chunk exceeds HARD_CAP_BYTES.

    Args:
        contents: Raw file bytes.
        path: Repository-relative file path.
        language: Language tag (kept for uniformity/telemetry).
        repo: Repository identifier.

    Returns:
        Iterable[Chunk]: Size-nudged chunks covering the entire file.

    Raises:
        ValueError: If contents is None or empty.
    """
    if not contents:
        raise ValueError("contents must be non-empty bytes")

    overlap = int(FALLBACK_OVERLAP_RATIO * SOFT_MAX_BYTES)
    ranges = _newline_aligned_ranges(contents, 0, len(contents), overlap=overlap)
    return [_make_chunk(contents, s, e, path, language, repo, mapper) for s, e in ranges]


def _chunk_document(
    contents: bytes, path: str, ext: str, repo: str, mapper: LineMapper
) -> List[Chunk]:
    ctx = _build_doc_context(contents)
    if ctx is None:
        return _chunk_plaintext_bytes(contents, path, "text", repo, mapper)

    if ext in DOC_MARKDOWN_EXTS:
        return _chunk_markdown(ctx, contents, path, repo, "markdown", mapper)
    if ext in DOC_JSON_EXTS:
        return _chunk_json(ctx, contents, path, repo, ext, mapper)
    if ext in DOC_YAML_EXTS or ext in DOC_TOML_EXTS:
        return _chunk_yaml_toml(ctx, contents, path, repo, ext, mapper)
    if ext in DOC_CSV_EXTS or ext in DOC_TSV_EXTS:
        delimiter = ',' if ext in DOC_CSV_EXTS else '\t'
        return _chunk_csv(ctx, contents, path, repo, delimiter, mapper)

    return _chunk_plaintext(ctx, contents, path, repo, mapper)


def _chunk_markdown(
    ctx: DocContext,
    contents: bytes,
    path: str,
    repo: str,
    language: str,
    mapper: LineMapper,
) -> List[Chunk]:
    blocks = _parse_markdown_blocks(ctx)
    if not blocks:
        return _chunk_plaintext(ctx, contents, path, repo, mapper)

    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    heading_stack: List[Tuple[int, str, str]] = []

    for block in blocks:
        if block.type == "heading":
            level = block.heading_level or 1
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.heading_title, block.heading_anchor))

        for part in _split_block_if_needed(ctx, block):
            part_bytes = ctx.char_to_byte[part.end_char] - ctx.char_to_byte[part.start_char]
            if current is None:
                current = _new_segment(part.start_char, heading_stack)
            else:
                current_size = ctx.char_to_byte[part.end_char] - ctx.char_to_byte[current["start_char"]]
                if current["blocks"] and current_size > DOC_SOFT_MAX_BYTES:
                    segments.append(current)
                    current = _new_segment(part.start_char, heading_stack)

            current["blocks"].append(part)
            current["end_char"] = part.end_char
            current["block_types"].add(part.type)
            if part.fence_lang:
                current["fence_langs"].add(part.fence_lang)

    if current and current["blocks"]:
        segments.append(current)

    segments = _merge_small_segments(ctx, segments)
    return _segments_to_chunks(ctx, contents, path, repo, language, segments, mapper)


def _skip_ws(text: str, idx: int) -> int:
    n = len(text)
    while idx < n and text[idx] in " \t\r\n\uFEFF":
        idx += 1
    return idx


def _line_prefix_start(text: str, idx: int) -> int:
    start = idx
    while start > 0 and text[start - 1] in " \t":
        start -= 1
    return start


def _json_object_spans(decoder: json.JSONDecoder, text: str, start_idx: int) -> List[Tuple[int, int, List[str]]]:
    spans: List[Tuple[int, int, List[str]]] = []
    idx = start_idx + 1
    n = len(text)
    structure_end = start_idx
    while True:
        idx = _skip_ws(text, idx)
        if idx >= n:
            structure_end = n
            break
        ch = text[idx]
        if ch == '}':
            structure_end = idx + 1
            idx = structure_end
            break
        if ch != '"':
            break
        try:
            key, key_end = json.decoder.scanstring(text, idx + 1)
        except ValueError:
            break
        colon = text.find(':', key_end)
        if colon == -1:
            break
        value_start = _skip_ws(text, colon + 1)
        try:
            _, value_end = decoder.raw_decode(text, value_start)
        except json.JSONDecodeError:
            break
        entry_start = _line_prefix_start(text, idx)
        entry_end = value_end
        while entry_end < n and text[entry_end].isspace():
            entry_end += 1
        if entry_end < n and text[entry_end] == ',':
            entry_end += 1
        spans.append((entry_start, entry_end, [key]))
        idx = entry_end

    structure_end = _skip_ws(text, idx)
    if structure_end < n and text[structure_end] == '}':
        structure_end += 1

    if spans:
        first_start, first_end, first_breadcrumb = spans[0]
        if first_start > start_idx:
            spans[0] = (start_idx, first_end, first_breadcrumb)
        last_start, last_end, last_breadcrumb = spans[-1]
        if structure_end > last_end:
            spans[-1] = (last_start, structure_end, last_breadcrumb)
    else:
        if structure_end <= start_idx:
            try:
                _, structure_end = decoder.raw_decode(text, start_idx)
            except json.JSONDecodeError:
                structure_end = len(text)
        spans.append((start_idx, structure_end, []))

    return spans


def _json_array_spans(decoder: json.JSONDecoder, text: str, start_idx: int) -> List[Tuple[int, int, List[str]]]:
    spans: List[Tuple[int, int, List[str]]] = []
    idx = start_idx + 1
    n = len(text)
    item_index = 0
    structure_end = start_idx
    while True:
        idx = _skip_ws(text, idx)
        if idx >= n:
            structure_end = n
            break
        if text[idx] == ']':
            structure_end = idx + 1
            idx = structure_end
            break
        entry_start = _line_prefix_start(text, idx)
        try:
            _, value_end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        entry_end = value_end
        while entry_end < n and text[entry_end].isspace():
            entry_end += 1
        if entry_end < n and text[entry_end] == ',':
            entry_end += 1
        spans.append((entry_start, entry_end, [f"[{item_index}]"]))
        idx = entry_end
        item_index += 1

    structure_end = _skip_ws(text, idx)
    if structure_end < n and text[structure_end] == ']':
        structure_end += 1

    if spans:
        first_start, first_end, first_breadcrumb = spans[0]
        if first_start > start_idx:
            spans[0] = (start_idx, first_end, first_breadcrumb)
        last_start, last_end, last_breadcrumb = spans[-1]
        if structure_end > last_end:
            spans[-1] = (last_start, structure_end, last_breadcrumb)
    else:
        if structure_end <= start_idx:
            try:
                _, structure_end = decoder.raw_decode(text, start_idx)
            except json.JSONDecodeError:
                structure_end = len(text)
        spans.append((start_idx, structure_end, []))

    return spans


def _json_spans(ctx: DocContext, ext: str) -> List[Tuple[int, int, List[str]]]:
    text = ctx.text
    if ext in {"jsonl", "ndjson"}:
        spans: List[Tuple[int, int, List[str]]] = []
        for idx, line in enumerate(ctx.lines):
            if not line.strip():
                continue
            start_char = ctx.line_char_offsets[idx]
            end_char = start_char + len(line)
            spans.append((start_char, end_char, [f"line {idx + 1}"]))
        return spans

    decoder = json.JSONDecoder()
    idx = _skip_ws(text, 0)
    if idx >= len(text):
        return []
    try:
        if text[idx] == '{':
            return _json_object_spans(decoder, text, idx)
        if text[idx] == '[':
            return _json_array_spans(decoder, text, idx)
        _, end_idx = decoder.raw_decode(text, idx)
        end_idx = _skip_ws(text, end_idx)
        return [(idx, end_idx, [])]
    except (json.JSONDecodeError, ValueError):
        return []


def _chunk_json(
    ctx: DocContext, contents: bytes, path: str, repo: str, ext: str, mapper: LineMapper
) -> List[Chunk]:
    spans = _json_spans(ctx, ext)
    if not spans:
        return _chunk_plaintext(ctx, contents, path, repo, mapper)

    segments: List[Dict[str, Any]] = []
    for start_char, end_char, breadcrumb in spans:
        block = DocBlock(start_char=start_char, end_char=end_char, type="json")
        for part in _split_block_if_needed(ctx, block):
            seg = _new_segment(part.start_char, [])
            seg["breadcrumb"] = list(breadcrumb)
            seg["blocks"].append(part)
            seg["end_char"] = part.end_char
            seg["block_types"].add("json")
            segments.append(seg)

    segments = _merge_small_segments(ctx, segments)
    language = "jsonl" if ext in {"jsonl", "ndjson"} else "json"
    return _segments_to_chunks(ctx, contents, path, repo, language, segments, mapper)


def _yaml_toml_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and "]" in stripped:
            inner = stripped.strip("[]")
            return inner.strip()
        if ":" in stripped:
            return stripped.split(":", 1)[0].strip()
        if "=" in stripped:
            return stripped.split("=", 1)[0].strip()
        if stripped.startswith("-"):
            payload = stripped[1:].strip()
            if ":" in payload:
                return payload.split(":", 1)[0].strip()
            return payload or "-"
    return ""


def _chunk_yaml_toml(
    ctx: DocContext, contents: bytes, path: str, repo: str, ext: str, mapper: LineMapper
) -> List[Chunk]:
    block_tag = "yaml" if ext in DOC_YAML_EXTS else "toml"
    lines = ctx.lines
    offsets = ctx.line_char_offsets
    total = len(lines)
    i = 0
    blocks: List[DocBlock] = []

    while i < total:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        start_char = offsets[i]
        end_char = start_char + len(line)
        j = i
        while j + 1 < total:
            nxt = lines[j + 1]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                end_char = offsets[j + 1] + len(nxt)
                j += 1
                continue
            indent = len(nxt) - len(nxt.lstrip(" "))
            if indent == 0 and not nxt.lstrip().startswith("#"):
                break
            end_char = offsets[j + 1] + len(nxt)
            j += 1
        blocks.append(DocBlock(start_char=start_char, end_char=end_char, type=block_tag))
        i = j + 1

    if not blocks:
        return _chunk_plaintext(ctx, contents, path, repo, mapper)

    segments: List[Dict[str, Any]] = []
    for block in blocks:
        heading = _yaml_toml_heading(ctx.text[block.start_char:block.end_char])
        for part in _split_block_if_needed(ctx, block):
            seg = _new_segment(part.start_char, [])
            if heading:
                seg["breadcrumb"] = [heading]
            seg["blocks"].append(part)
            seg["end_char"] = part.end_char
            seg["block_types"].add(block_tag)
            segments.append(seg)

    segments = _merge_small_segments(ctx, segments)
    language = "yaml" if ext in DOC_YAML_EXTS else "toml"
    return _segments_to_chunks(ctx, contents, path, repo, language, segments, mapper)


def _chunk_csv(
    ctx: DocContext, contents: bytes, path: str, repo: str, delimiter: str, mapper: LineMapper
) -> List[Chunk]:
    if not ctx.text:
        return []
    block = DocBlock(start_char=0, end_char=len(ctx.text), type="table")
    segments: List[Dict[str, Any]] = []
    for part in _split_block_if_needed(ctx, block):
        seg = _new_segment(part.start_char, [])
        seg["blocks"].append(part)
        seg["end_char"] = part.end_char
        seg["block_types"].update({"table", "csv" if delimiter == ',' else "tsv"})
        seg["table_delimiter"] = delimiter
        segments.append(seg)

    segments = _merge_small_segments(ctx, segments)
    language = "csv" if delimiter == ',' else "tsv"
    return _segments_to_chunks(ctx, contents, path, repo, language, segments, mapper)


def _chunk_plaintext(
    ctx: DocContext, contents: bytes, path: str, repo: str, mapper: LineMapper
) -> List[Chunk]:
    if not ctx.text:
        return []
    block = DocBlock(start_char=0, end_char=len(ctx.text), type="text")
    segments: List[Dict[str, Any]] = []
    for part in _split_block_if_needed(ctx, block):
        seg = _new_segment(part.start_char, [])
        seg["blocks"].append(part)
        seg["end_char"] = part.end_char
        seg["block_types"].add("text")
        segments.append(seg)

    segments = _merge_small_segments(ctx, segments)
    return _segments_to_chunks(ctx, contents, path, repo, "text", segments, mapper)


def _doc_byte_ranges(contents: bytes, start: int, end: int, overlap: int = DOC_OVERLAP_BYTES) -> List[Tuple[int, int]]:
    if start >= end:
        return []
    ranges: List[Tuple[int, int]] = []
    cur = start
    while cur < end:
        soft_end = min(cur + DOC_SOFT_MAX_BYTES, end)
        lo = max(cur + 1, soft_end - NEWLINE_WINDOW)
        hi = min(end - 1, soft_end + NEWLINE_WINDOW)
        split = _nearest_newline(contents, soft_end, lo, hi)
        if split is None:
            split = soft_end
        if (split - cur) > DOC_HARD_CAP_BYTES:
            split = cur + DOC_HARD_CAP_BYTES
        if split <= cur:
            split = min(end, cur + DOC_HARD_CAP_BYTES)
        ranges.append((cur, split))
        if split >= end:
            break
        next_start = split - overlap
        if next_start <= cur:
            next_start = split
        cur = next_start
    return ranges


def _chunk_plaintext_bytes(
    contents: bytes, path: str, language: str, repo: str, mapper: LineMapper
) -> List[Chunk]:
    total = len(contents)
    if total == 0:
        return []
    eol = "\r\n" if b"\r\n" in contents else ("\n" if b"\n" in contents else "")
    ranges = _doc_byte_ranges(contents, 0, total)
    chunks: List[Chunk] = []
    for start, end in ranges:
        text = contents[start:end].decode("utf-8", errors="replace")
        requirement_sentences = _find_requirement_sentences(text)
        code_refs = {match.strip() for match in _INLINE_CODE_RE.findall(text) if match.strip()}
        chunk_kind = "req" if requirement_sentences else "text"
        metadata = {
            "heading_breadcrumb": [],
            "anchors": [],
            "requirement_sentences": requirement_sentences,
            "code_refs": sorted(code_refs),
            "chunk_kind": chunk_kind,
            "grammar_version": DOC_GRAMMAR_VERSION,
            "eol": eol,
            "overlap_bytes": DOC_OVERLAP_BYTES,
        }
        signature = _document_signature(repo, path, start, end, [], text[:120])
        chunk = _make_chunk(
            contents,
            start,
            end,
            path,
            language,
            repo,
            mapper,
            signature=signature,
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks


def _new_segment(start_char: int, heading_stack: List[Tuple[int, str, str]]) -> Dict[str, Any]:
    breadcrumb = [title for (_, title, _) in heading_stack if title]
    anchors = [anchor for (_, _, anchor) in heading_stack if anchor]
    return {
        "start_char": start_char,
        "end_char": start_char,
        "blocks": [],
        "breadcrumb": breadcrumb,
        "anchors": anchors,
        "block_types": set(),
        "fence_langs": set(),
    }


def _parse_markdown_blocks(ctx: DocContext) -> List[DocBlock]:
    blocks: List[DocBlock] = []
    lines = ctx.lines
    offsets = ctx.line_char_offsets
    total = len(lines)
    i = 0

    rst_level_map = {
        "=": 1,
        "-": 2,
        "~": 3,
        "`": 4,
        ":": 5,
        "'": 6,
        '"': 6,
        "^": 6,
        "*": 7,
        "+": 7,
        "#": 7,
        "<": 7,
        ">": 7,
    }

    while i < total:
        line = lines[i]
        stripped_line = line.rstrip("\r\n")
        content = stripped_line.strip()
        if not content:
            i += 1
            continue

        start_char = offsets[i]
        line_end_char = start_char + len(stripped_line) + (1 if line.endswith("\n") else 0)

        # Setext/RST headings (current line + underline)
        if i + 1 < total:
            next_line = lines[i + 1]
            next_clean = next_line.strip()
            if _MD_SETEXT_RE.match(next_clean) and content:
                level = 1 if next_clean.startswith("=") else 2
                title = content
                anchor = _slugify(title)
                end_char = offsets[i + 1] + len(next_line)
                blocks.append(
                    DocBlock(
                        start_char=start_char,
                        end_char=end_char,
                        type="heading",
                        heading_level=level,
                        heading_title=title,
                        heading_anchor=anchor,
                    )
                )
                i += 2
                continue
            if _RST_SETEXT_RE.match(next_clean) and content:
                char = next_clean[:1]
                level = rst_level_map.get(char, 3)
                anchor = _slugify(content)
                end_char = offsets[i + 1] + len(next_line)
                blocks.append(
                    DocBlock(
                        start_char=start_char,
                        end_char=end_char,
                        type="heading",
                        heading_level=level,
                        heading_title=content,
                        heading_anchor=anchor,
                    )
                )
                i += 2
                continue

        # ATX heading
        atx = _MD_ATX_HEADING_RE.match(content)
        if atx:
            level = len(atx.group(1))
            raw_title = atx.group(2).strip()
            title = raw_title.rstrip("#").strip()
            anchor = _slugify(title)
            end_char = start_char + len(line)
            blocks.append(
                DocBlock(
                    start_char=start_char,
                    end_char=end_char,
                    type="heading",
                    heading_level=level,
                    heading_title=title,
                    heading_anchor=anchor,
                )
            )
            i += 1
            continue

        # Code fence
        fence_match = _FENCE_RE.match(content)
        if fence_match:
            fence_marker = fence_match.group(1)
            info = fence_match.group(2).strip()
            fence_lang = info.split()[0] if info else ""
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            escaped = re.escape(fence_char)
            closing = re.compile("^" + escaped + "{" + str(fence_len) + ",}\\s*$")
            j = i + 1
            end_char = start_char + len(line)
            while j < total:
                close_content = lines[j].strip()
                if closing.match(close_content):
                    end_char = offsets[j] + len(lines[j])
                    j += 1
                    break
                j += 1
            else:
                j = total
                end_char = len(ctx.text)
            blocks.append(
                DocBlock(
                    start_char=start_char,
                    end_char=end_char,
                    type="fence",
                    fence_lang=fence_lang or None,
                )
            )
            i = j
            continue

        # Markdown table (pipe-form)
        if _TABLE_LINE_RE.match(line):
            j = i
            end_char = start_char
            while j < total and _TABLE_LINE_RE.match(lines[j]):
                end_char = offsets[j] + len(lines[j])
                j += 1
            blocks.append(
                DocBlock(
                    start_char=start_char,
                    end_char=end_char,
                    type="table",
                )
            )
            i = j
            continue

        # Paragraph/list/blockquote aggregation
        block_type = "paragraph"
        stripped_leading = stripped_line.lstrip()
        if stripped_leading.startswith(">"):
            block_type = "blockquote"
        elif re.match(r"^(?:[*+-]|\d+[.)])\s", stripped_leading):
            block_type = "list"

        j = i
        end_char = offsets[j] + len(lines[j])
        while j + 1 < total:
            nxt = lines[j + 1]
            nxt_content = nxt.strip()
            if not nxt_content:
                end_char = offsets[j + 1] + len(nxt)
                j += 1
                break
            if _MD_ATX_HEADING_RE.match(nxt_content):
                break
            if _FENCE_RE.match(nxt_content):
                break
            if _TABLE_LINE_RE.match(nxt):
                break
            if j + 1 < total and _MD_SETEXT_RE.match(nxt_content):
                break
            if block_type == "blockquote" and not nxt.lstrip().startswith(">"):
                break
            end_char = offsets[j + 1] + len(nxt)
            j += 1

        blocks.append(
            DocBlock(
                start_char=start_char,
                end_char=end_char,
                type=block_type,
            )
        )
        i = j + 1

    return blocks


def _split_block_if_needed(ctx: DocContext, block: DocBlock) -> List[DocBlock]:
    start_bytes = ctx.char_to_byte[block.start_char]
    end_bytes = ctx.char_to_byte[block.end_char]
    total = end_bytes - start_bytes
    if total <= DOC_HARD_CAP_BYTES or block.type == "heading":
        return [block]

    newline_positions: List[int] = []
    segment_text = ctx.text[block.start_char:block.end_char]
    absolute = block.start_char
    for ch in segment_text:
        absolute += 1
        if ch == "\n":
            newline_positions.append(absolute)
    if not newline_positions or newline_positions[-1] != block.end_char:
        newline_positions.append(block.end_char)

    parts: List[DocBlock] = []
    cur_start = block.start_char
    cur_start_bytes = ctx.char_to_byte[cur_start]
    idx = 0
    while cur_start < block.end_char:
        limit = cur_start_bytes + DOC_SOFT_MAX_BYTES
        chosen_char: Optional[int] = None

        while idx < len(newline_positions):
            candidate = newline_positions[idx]
            candidate_bytes = ctx.char_to_byte[candidate]
            if candidate_bytes - cur_start_bytes > DOC_HARD_CAP_BYTES:
                break
            chosen_char = candidate
            idx += 1
            if candidate_bytes >= limit:
                break

        if chosen_char is None:
            forced_bytes = min(cur_start_bytes + DOC_HARD_CAP_BYTES, ctx.char_to_byte[block.end_char])
            forced_char = _bytes_to_char(ctx, forced_bytes)
            if forced_char <= cur_start:
                forced_char = min(block.end_char, cur_start + 1)
            chosen_char = forced_char

        if chosen_char <= cur_start:
            chosen_char = min(block.end_char, cur_start + 1)

        parts.append(_clone_block(block, cur_start, chosen_char))
        cur_start = chosen_char
        cur_start_bytes = ctx.char_to_byte[cur_start]

    return parts


def _find_requirement_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    results: List[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        if _REQUIREMENT_PATTERN.search(candidate) and candidate not in seen:
            results.append(candidate)
            seen.add(candidate)
    return results


def _extract_requirement_sentences(ctx: DocContext, start_char: int, end_char: int) -> List[str]:
    snippet = ctx.text[start_char:end_char]
    return _find_requirement_sentences(snippet)


def _merge_small_segments(ctx: DocContext, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    merged: List[Dict[str, Any]] = []
    for seg in segments:
        seg_size = ctx.char_to_byte[seg["end_char"]] - ctx.char_to_byte[seg["start_char"]]
        if merged and seg_size < DOC_MIN_CHUNK_BYTES:
            prev = merged[-1]
            prev["blocks"].extend(seg["blocks"])
            prev["end_char"] = seg["end_char"]
            prev["block_types"].update(seg["block_types"])
            prev["fence_langs"].update(seg["fence_langs"])
        else:
            merged.append(seg)

    if len(merged) > 1:
        last = merged[-1]
        last_size = ctx.char_to_byte[last["end_char"]] - ctx.char_to_byte[last["start_char"]]
        if last_size < DOC_MIN_CHUNK_BYTES:
            prev = merged[-2]
            prev["blocks"].extend(last["blocks"])
            prev["end_char"] = last["end_char"]
            prev["block_types"].update(last["block_types"])
            prev["fence_langs"].update(last["fence_langs"])
            merged.pop()

    return merged


def _segments_to_chunks(
    ctx: DocContext,
    contents: bytes,
    path: str,
    repo: str,
    language: str,
    segments: List[Dict[str, Any]],
    mapper: LineMapper,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    prev_end_bytes: Optional[int] = None

    for seg in segments:
        start_char = seg["start_char"]
        end_char = seg["end_char"]
        start_bytes = ctx.char_to_byte[start_char]
        end_bytes = ctx.char_to_byte[end_char]

        if prev_end_bytes is not None:
            overlap_start = max(0, prev_end_bytes - DOC_OVERLAP_BYTES)
            if overlap_start < start_bytes:
                start_bytes = overlap_start
                start_char = _bytes_to_char(ctx, start_bytes)

        chunk_text = ctx.text[start_char:end_char]
        requirement_sentences = _extract_requirement_sentences(ctx, start_char, end_char)
        code_refs = {match.strip() for match in _INLINE_CODE_RE.findall(chunk_text) if match.strip()}
        code_refs.update(f"fence:{lang}" for lang in seg["fence_langs"] if lang)

        block_types = seg["block_types"]
        chunk_kind = "doc"
        if "table" in block_types:
            chunk_kind = "table"
        elif "fence" in block_types:
            chunk_kind = "fence"
        elif "json" in block_types:
            chunk_kind = "json"
        elif "yaml" in block_types:
            chunk_kind = "yaml"
        elif "toml" in block_types:
            chunk_kind = "toml"
        elif "csv" in block_types:
            chunk_kind = "csv"
        elif "tsv" in block_types:
            chunk_kind = "tsv"
        elif "text" in block_types:
            chunk_kind = "text"
        if chunk_kind == "doc" and requirement_sentences:
            chunk_kind = "req"

        metadata = {
            "heading_breadcrumb": seg["breadcrumb"],
            "anchors": seg["anchors"],
            "requirement_sentences": requirement_sentences,
            "code_refs": sorted(code_refs),
            "chunk_kind": chunk_kind,
            "grammar_version": DOC_GRAMMAR_VERSION,
            "eol": ctx.eol,
            "overlap_bytes": DOC_OVERLAP_BYTES,
        }
        table_delimiter = seg.get("table_delimiter")
        if table_delimiter:
            metadata["table_delimiter"] = table_delimiter

        signature = _document_signature(repo, path, start_bytes, end_bytes, seg["breadcrumb"], chunk_text[:120])
        chunk = _make_chunk(
            contents,
            start_bytes,
            end_bytes,
            path,
            language,
            repo,
            mapper,
            signature=signature,
            metadata=metadata,
        )
        chunks.append(chunk)
        prev_end_bytes = end_bytes

    return chunks


def _document_signature(
    repo: str,
    path: str,
    start_bytes: int,
    end_bytes: int,
    breadcrumb: List[str],
    preview: str,
) -> str:
    crumb = "/".join(breadcrumb)
    cleaned_preview = " ".join(preview.strip().split())
    if len(cleaned_preview) > 80:
        cleaned_preview = cleaned_preview[:77] + "..."
    parts: List[str] = []
    if repo:
        parts.append(repo)
    parts.extend([path, f"{start_bytes}-{end_bytes}"])
    if crumb:
        parts.append(crumb)
    if cleaned_preview:
        parts.append(cleaned_preview)
    parts.append(DOC_GRAMMAR_VERSION)
    return "|".join(parts)


def _build_doc_context(contents: bytes) -> Optional[DocContext]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        return None

    char_to_byte: List[int] = [0] * (len(text) + 1)
    byte_pos = 0
    for idx, ch in enumerate(text):
        char_to_byte[idx] = byte_pos
        byte_pos += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = byte_pos

    lines = text.splitlines(True)
    line_offsets: List[int] = []
    cursor = 0
    for line in lines:
        line_offsets.append(cursor)
        cursor += len(line)

    eol = "\r\n" if "\r\n" in text else "\n"
    return DocContext(
        text=text,
        char_to_byte=char_to_byte,
        lines=lines,
        line_char_offsets=line_offsets,
        total_bytes=len(contents),
        eol=eol,
    )


def _chunk_code(
    contents: bytes, path: str, language: str, repo: str, mapper: LineMapper
) -> Iterable[Chunk]:
    """
    Chunk code via Tree-sitter:
      - Emit one chunk per executable unit (methods/ctors/inits/accessors).
      - Emit one metadata chunk per top-level type (no bodies).
      - On parse failure, fall back to size-based chunking with 10% overlap.
      Returns a single iterable (list) containing all chunks.
    """
    try:
        parser = get_parser(language)
        tree = parser.parse(contents)
    except Exception:
        return list(_chunk_fallback(contents, path, language, repo, mapper))

    root = tree.root_node
    pkg = _get_package_name(contents, root, language)

    types = _query_nodes(root,language, GRAMMAR_QUERIES["Type"].get(language, []), "type")
    methods = _query_nodes(root,language, GRAMMAR_QUERIES["Method"].get(language, []), "method")
    ctors = _query_nodes(root,language, GRAMMAR_QUERIES["Constructor"].get(language, []), "constructor")
    inits = _query_nodes(root,language, GRAMMAR_QUERIES["Initializer"].get(language, []), "initializer")
    accessors = _query_nodes(root,language, GRAMMAR_QUERIES["Accessor"].get(language, []), "accessor")

    chunks: List[Chunk] = []

    exec_nodes = sorted(_unique_by_span(methods + ctors + inits + accessors),
                        key=lambda n: (n.start_byte, n.end_byte))
    for n in exec_nodes:
        for ch in _build_method_like_chunks(n, contents, path, language, repo, pkg, mapper):
            chunks.append(ch)

    top_level_types = [t for t in types if _is_top_level_type(t, language)]
    fields = _query_nodes(root,language, GRAMMAR_QUERIES["Field"].get(language, []), "field")
    enums = _query_nodes(root,language, GRAMMAR_QUERIES["EnumMember"].get(language, []), "enum_member")
    anno_elems = _query_nodes(root,language, GRAMMAR_QUERIES["AnnotationElement"].get(language, []),
                              "annotation_element")
    record_comps = _query_nodes(root,language, GRAMMAR_QUERIES["RecordComponent"].get(language, []),
                                "record_component")

    for t in sorted(top_level_types, key=lambda n: (n.start_byte, n.end_byte)):
        chunks.append(
            _build_class_metadata_chunk(
                t,
                contents,
                path,
                language,
                repo,
                pkg,
                mapper,
                methods=exec_nodes,
                fields=fields,
                enums=enums,
                anno_elems=anno_elems,
                record_comps=record_comps,
            )
        )

    # If Tree-sitter produced no actionable nodes, fall back to size-based chunking
    if not chunks:
        return list(_chunk_fallback(contents, path, language, repo, mapper))

    return chunks


def _root_types(patterns: list[str] | None) -> set[str]:
    out = set()
    for p in (patterns or []):
        p = p.strip()
        m = _ROOT_ATOM_RE.match(p)
        if m:
            out.add(m.group(1))
    return out


def _build_containers() -> tuple[set[str], set[str]]:
    """Compute container node-type sets from centralized GRAMMAR_QUERIES.

    Returns:
        (TYPE_CONTAINERS, ROUTINE_CONTAINERS): Two sets of node.type strings
        that represent, respectively, type declarations and routine containers
        (methods/functions/constructors/initializers/accessors).
    """

    def root_types_for(category: str) -> set[str]:
        acc: set[str] = set()
        for pats in GRAMMAR_QUERIES.get(category, {}).values():
            acc |= _root_types(pats)
        return acc

    type_cont = root_types_for("Type")
    routine_cont = set()
    for cat in ("Method", "Constructor", "Initializer", "Accessor"):
        routine_cont |= root_types_for(cat)
    return type_cont, routine_cont


TYPE_CONTAINERS, ROUTINE_CONTAINERS = _build_containers()


def _is_top_level_type(n: Node, language: str) -> bool:
    """
    Top-level iff no ancestor is a type container (class/interface/enum/record/etc.)
    or a routine container (method/function/constructor/initializer).
    Works across Java/Kotlin/Dart/Pascal.
    """
    cur = n.parent
    while cur is not None:
        t = cur.type
        if t in TYPE_CONTAINERS or t in ROUTINE_CONTAINERS:
            return False
        cur = cur.parent
    return True


from tree_sitter import Query, QueryError
try:  # Tree-sitter releases prior to 0.22 omit QueryCursor
    from tree_sitter import QueryCursor  # type: ignore
except ImportError:  # pragma: no cover - fallback for older runtimes
    QueryCursor = None  # type: ignore


def _query_nodes(root:Node, language: str, sexprs: list[str], capture: str):
    """
    Execute `sexprs` as a Tree-sitter Query and return nodes captured as `@{capture}`.
    """
    if not sexprs:
        return []

    try:
        lang = get_language(language)
        qsrc = "\n".join(s.strip() for s in sexprs if s.strip())
        query = Query(lang, qsrc)
    except QueryError as exc:
        logger.error("Tree-sitter query failed for language '%s': %s", language, exc)
        return []

    captures_nodes: List[Node] = []
    if QueryCursor is not None:
        cursor = QueryCursor()
        cursor.exec(query, root)
        for entry in cursor.captures():
            if isinstance(entry, tuple):
                node, cap_name = entry[0], entry[1] if len(entry) > 1 else None
                if cap_name == capture:
                    captures_nodes.append(node)
    else:
        captures = query.captures(root)
        if isinstance(captures, dict):  # older python bindings return {capture: [nodes]}
            captures_nodes.extend(captures.get(capture, []))
        else:
            for entry in captures:
                if not isinstance(entry, tuple):
                    continue
                node = entry[0]
                cap_name = entry[1] if len(entry) > 1 else None
                if cap_name == capture:
                    captures_nodes.append(node)

    nodes = list(captures_nodes)

    # (optional) ensure deterministic order & dedupe by byte range
    nodes.sort(key=lambda n: (n.start_byte, n.end_byte))
    deduped = []
    seen = set()
    for n in nodes:
        key = (n.start_byte, n.end_byte)
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


def _query_matches(
    root: Node,
    language: str,
    sexprs: list[str],
    capture: str,
) -> List[Tuple[Node, Dict[str, Node]]]:
    """Return capture-grouped matches keyed by capture name."""
    if not sexprs:
        return []
    try:
        lang = get_language(language)
    except QueryError as exc:
        logger.error("Tree-sitter query failed for language '%s': %s", language, exc)
        return []

    matches: List[Tuple[Node, Dict[str, Node]]] = []
    for qsrc in _query_variants(sexprs, language):
        try:
            query = Query(lang, qsrc)
        except QueryError as exc:
            logger.error("Tree-sitter query failed for language '%s': %s", language, exc)
            continue
        matches.extend(_collect_query_matches(query, root, capture))
    return matches


def _query_variants(sexprs: list[str], language: str) -> list[str]:
    """Return query source variants for language-specific node aliases."""
    qsrc = "\n".join(s.strip() for s in sexprs if s.strip())
    if language == "cpp" and ("scoped_identifier" in qsrc or "type_identifier" in qsrc):
        return [
            qsrc,
            qsrc.replace("scoped_identifier", "qualified_identifier").replace(
                "type_identifier", "namespace_identifier"
            ),
        ]
    return [qsrc]


def _collect_query_matches(
    query: Query,
    root: Node,
    capture: str,
) -> List[Tuple[Node, Dict[str, Node]]]:
    """Collect grouped capture matches from a prepared query."""
    if QueryCursor is not None and hasattr(QueryCursor, "matches"):
        cursor = QueryCursor()
        cursor.exec(query, root)
        raw_matches = cursor.matches()
    elif hasattr(query, "matches"):
        raw_matches = query.matches(root)
    else:
        raw_matches = []

    matches: List[Tuple[Node, Dict[str, Node]]] = []
    for match in raw_matches:
        captures = getattr(match, "captures", match[1] if isinstance(match, tuple) else match)
        if isinstance(captures, dict):
            captures = [(node, name) for name, nodes in captures.items() for node in nodes]
        grouped: Dict[str, Node] = {}
        for entry in captures or []:
            if isinstance(entry, tuple):
                node = entry[0]
                cap_name = entry[1] if len(entry) > 1 else None
                if cap_name:
                    grouped[cap_name] = node
        primary = grouped.get(capture)
        if primary is not None:
            matches.append((primary, grouped))
    return matches


def _unique_by_span(nodes: List[Node]) -> List[Node]:
    """Deduplicate nodes by (start_byte, end_byte, type)."""
    seen, uniq = set(), []
    for n in nodes:
        k = (n.start_byte, n.end_byte, n.type)
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return uniq


def _build_method_like_chunks(
    node: Node,
    contents: bytes,
    path: str,
    language: str,
    repo: str,
    pkg: str,
    mapper: LineMapper,
) -> List[Chunk]:
    """
    Emit 1..N chunks for a method/function/ctor/init/accessor.

    Logic:
      - Expand upward to include contiguous leading doc/annotation trivia (no blank line gaps).
      - If total size ≤ HARD_CAP_BYTES: emit one slice [doc_start..node.end].
      - Else: split the body by statement groups (child named nodes); if any group > cap,
              split that group by size with newline-nudge. Emit first part starting at doc_start,
              remaining parts as their exact body slices. Chunk text always equals sliced bytes.

    Params:
      node: TS node for the executable unit.  contents/path/language/repo/pkg: context.

    Returns:
      Iterable[Chunk] with precise byte spans; signature field stores an FQDN-like id.
    """
    doc_start = _leading_trivia_start(contents, node)
    body = _find_body_node(node)
    unit_start, unit_end = doc_start, node.end_byte
    if (unit_end - unit_start) <= HARD_CAP_BYTES:
        signature = _fqn_for_node(node, contents, path, language, pkg)
        return [
            _make_chunk(
                contents, unit_start, unit_end, path, language, repo, mapper, signature=signature
            )
        ]

    parts: List[Tuple[int, int]]
    if body is not None and body.named_children:
        parts = _pack_siblings(contents, list(body.named_children))
        parts = [(max(s, body.start_byte), min(e, body.end_byte)) for s, e in parts]
        fixed: List[Tuple[int, int]] = []
        for s, e in parts:
            fixed.extend(_newline_aligned_ranges(contents, s, e) if (e - s) > HARD_CAP_BYTES else [(s, e)])
        parts = fixed
    else:
        parts = _newline_aligned_ranges(contents, node.start_byte, node.end_byte)

    fqn = _fqn_for_node(node, contents, path, language, pkg)
    chunks = []
    for i, (s, e) in enumerate(parts):
        start = unit_start if i == 0 else s
        end = e
        chunks.append(
            Chunk(
                chunk=contents[start:end].decode("utf-8", errors="replace"),
                repo=repo,
                path=path,
                language=language,
                start_rc=mapper.byte_to_point(start),
                end_rc=mapper.byte_to_point(end),
                start_bytes=start,
                end_bytes=end,
                signature=(f"{fqn}#part{i + 1}" if len(parts) > 1 else fqn),
            )
        )
    return chunks


def _build_class_metadata_chunk(
    type_node: Node,
    contents: bytes,
    path: str,
    language: str,
    repo: str,
    pkg: str,
    mapper: LineMapper,
    methods: List[Node],
    fields: List[Node],
    enums: List[Node],
    anno_elems: List[Node],
    record_comps: List[Node],
) -> Chunk:
    """
    Emit one metadata chunk per top-level type (no bodies).

    Logic:
      - Slice the type header up to '{' (or end if bodyless).
      - Append signatures of methods/ctors (start..body_start), field/property decls (trim at '='),
        enum members, annotation elements, and record components contained within the type span.
      - Chunk byte span = full type span; text is the assembled metadata (signature-only).

    Returns:
      Chunk whose signature is "<pkg>.<Type>#metadata".
    """
    t_start, t_end = type_node.start_byte, type_node.end_byte
    header = contents[t_start:_header_end_byte(type_node, contents)].decode("utf-8", errors="replace")

    def in_type(n: Node) -> bool:
        return t_start <= n.start_byte and n.end_byte <= t_end

    sigs: List[str] = []
    for n in methods:
        if in_type(n):
            body = _find_body_node(n)
            sig_end = body.start_byte if body else n.end_byte
            sigs.append(contents[n.start_byte:sig_end].decode("utf-8", errors="replace").strip())
    for n in _unique_by_span(fields):
        if in_type(n):
            src = contents[n.start_byte:n.end_byte]
            eq = src.find(b'=')
            sigs.append((src[:eq] if eq != -1 else src).decode("utf-8", errors="replace").strip())
    for n in _unique_by_span(enums):
        if in_type(n):
            sigs.append(contents[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip())
    for n in _unique_by_span(anno_elems):
        if in_type(n):
            sigs.append(contents[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip())
    for n in _unique_by_span(record_comps):
        if in_type(n):
            sigs.append(contents[n.start_byte:n.end_byte].decode("utf-8", errors="replace").strip())

    type_name = _node_identifier_text(contents, type_node) or "<anonymous>"
    lines = ([f"package {pkg}"] if pkg else []) + [header.strip()] + [s for s in sigs if s]
    meta_text = ("\n".join(lines).strip() + "\n") if lines else ""
    return Chunk(
        chunk=meta_text,
        repo=repo,
        path=path,
        language=language,
        start_rc=mapper.byte_to_point(t_start),
        end_rc=mapper.byte_to_point(t_end),
        start_bytes=t_start,
        end_bytes=t_end,
        signature=(f"{pkg}.{type_name}#metadata" if pkg else f"{type_name}#metadata"),
    )


def _header_end_byte(type_node: Node, contents: bytes) -> int:
    """Return byte offset of type header end (first '{' or node end)."""
    start, end = type_node.start_byte, type_node.end_byte
    rel = contents[start:end]
    i = rel.find(b'{')
    return start + i if i != -1 else end


def _leading_trivia_start(contents: bytes, node: Node) -> int:
    """
    Find the start byte including contiguous leading doc/comments/annotations with no blank-line gap.

    Returns:
      Start byte offset to include as leading trivia.
    """
    start = node.start_byte
    sib = node.prev_named_sibling
    while sib is not None:
        t = sib.type
        if not ("comment" in t or "kdoc" in t or "documentation" in t or "doc" in t or "annotation" in t):
            break
        if _has_blank_line_between(contents, sib.end_byte, start):
            break
        start = sib.start_byte
        sib = sib.prev_named_sibling
    return start


def _has_blank_line_between(contents: bytes, a_end: int, b_start: int) -> bool:
    """Return True if a blank line exists between byte offsets a_end and b_start."""
    return bool(re.search(rb"\r?\n[ \t]*\r?\n", contents[a_end:b_start]))


def _find_body_node(node: Node) -> Optional[Node]:
    """Return the child node that represents the function/ctor body, if present."""
    for ch in node.named_children:
        if ch.type in ("block", "function_body", "constructor_body"):
            return ch
    return None


def _node_identifier_text(contents: bytes, node: Node) -> Optional[str]:
    """
    Extract a declaration identifier from child nodes named '*identifier' or 'name';
    fallback to regex on the head slice.
    """
    for ch in node.named_children:
        if "identifier" in ch.type or ch.type.endswith("_identifier") or ch.type == "name":
            return contents[ch.start_byte:ch.end_byte].decode("utf-8", errors="replace")
    head = contents[node.start_byte:min(node.end_byte, node.start_byte + 512)].decode("utf-8", errors="replace")
    m = re.search(r"([A-Za-z_][A-Za-z0-9_.$<>]*)\s*\(", head)
    return m.group(1) if m else None


def _enclosing_types(node: Node, contents: bytes) -> List[str]:
    """
    Collect enclosing type names (outer→inner) using TYPE_CONTAINERS derived
    from grammar queries, avoiding hardcoded language-specific node-type lists.
    """
    names: List[str] = []
    cur = node.parent
    while cur is not None:
        if cur.type in TYPE_CONTAINERS:
            nm = _node_identifier_text(contents, cur)
            if nm:
                names.append(nm)
        cur = cur.parent
    return list(reversed(names))


def _get_package_name(contents: bytes, root: Node, language: str) -> str:
    """
    Return a normalized package/library/module name for metadata/FQNs.

    Steps:
      1) Query PACKAGE_LIKE_SEXPR and take the first @package capture.
      2) Decode that slice and, if it contains a directive keyword, extract the
         identifier chain after it; otherwise use the raw slice (e.g., Pascal moduleName).
      3) Sanitize to a dotted identifier: keep [A-Za-z0-9_.], collapse repeats,
         trim leading/trailing dots, and drop a trailing ';'.
    """
    sexprs = GRAMMAR_QUERIES["Package"].get(language, [])
    nodes = _query_nodes(root,language, sexprs, "package")
    if not nodes:
        return ""

    raw = contents[nodes[0].start_byte:nodes[0].end_byte].decode("utf-8", errors="replace").strip()
    if raw.endswith(";"):
        raw = raw[:-1].rstrip()

    m = re.match(r"^(?:package|library|program)\s+([A-Za-z_][\w.]*)\s*$", raw)
    name = m.group(1) if m else raw

    name = re.sub(r"[^A-Za-z0-9_.]+", ".", name)
    name = re.sub(r"\.{2,}", ".", name).strip(".")
    return name


def _fqn_for_node(
    node: Node,
    contents: bytes,
    path: str = "",
    language: str = "",
    pkg: str = "",
    context: Optional[Dict[str, Node]] = None,
) -> str:
    """
    Build a best-effort FQDN: package + explicit/enclosing scope + name(params).

    If an explicit scope is provided in context, it overrides parent-walked scopes.
    """
    name = _node_identifier_text(contents, node) or ""
    explicit_scope = context.get("explicit_scope") if context else None
    explicit_name = (
        contents[explicit_scope.start_byte:explicit_scope.end_byte].decode("utf-8", errors="replace")
        if explicit_scope is not None
        else ""
    )
    encl = [] if explicit_name else _enclosing_types(node, contents)
    container = "member" if explicit_name or encl else "top_level"
    if not name:
        name = (encl[-1] if encl else Path(path).stem)
    head = contents[node.start_byte:min(node.end_byte, node.start_byte + 512)].decode("utf-8", errors="replace")
    m = re.search(r"\((.*?)\)", head, flags=re.DOTALL)
    params = m.group(1).strip() if m else ""
    scope = explicit_name or (".".join(encl) if encl else Path(path).stem)
    base = f"{pkg}.{scope}.{name}" if pkg else f"{scope}.{name}"
    if not language:
        return base
    return f"{base}({params})|{container}|{language}"


# -----------------------------
# Helpers (small, single-purpose)
# -----------------------------

def _first_multi_child_level(root) -> "Node":
    """
    Return the shallowest node on the root→leaf path whose **named children count > 1**.
    Walks down chains of single-child wrappers (common in JSON/YAML/XML) and stops at the
    first level that actually has siblings. If none exist, returns the leaf (so we fall back).
    """
    node = root
    while True:
        kids = list(node.named_children)
        if len(kids) != 1:
            return node
        node = kids[0]


def _pack_siblings(contents: bytes, siblings: List["Node"]) -> List[Tuple[int, int]]:
    """
    Pack sibling nodes sequentially up to SOFT_MAX_BYTES. Oversized siblings are handled
    via a one-level deeper pass or size-based splits. Produces contiguous, non-overlapping
    (start,end) byte ranges that preserve inter-sibling gaps.
    """
    ranges: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    last_end: Optional[int] = None

    for i, node in enumerate(siblings):
        size = node.end_byte - node.start_byte
        # Oversized single sibling: flush current group, then split this node alone.
        if size > HARD_CAP_BYTES:
            if cur_start is not None:
                ranges.append((cur_start, last_end))
                cur_start = last_end = None
            ranges.extend(_split_large_node(contents, node))
            continue

        if cur_start is None:
            cur_start = node.start_byte
            last_end = node.end_byte
            continue

        prospective_end = node.end_byte
        if (prospective_end - cur_start) <= SOFT_MAX_BYTES:
            last_end = prospective_end
        else:
            ranges.append((cur_start, last_end))
            cur_start, last_end = node.start_byte, node.end_byte

    if cur_start is not None:
        ranges.append((cur_start, last_end))
    return ranges


def _split_large_node(contents: bytes, node: "Node") -> List[Tuple[int, int]]:
    """
    One child-level pass: try packing named children inside the large node. If the node has no
    named children or any sub-range still violates caps, fall back to size-based splitting
    with newline nudging within the node span.
    """
    kids = list(node.named_children)
    if not kids:
        return _newline_aligned_ranges(contents, node.start_byte, node.end_byte)

    # Pack child siblings within node bounds.
    sub = _pack_siblings(contents, kids)
    if not sub:
        return _newline_aligned_ranges(contents, node.start_byte, node.end_byte)

    # Clamp first/last to include the node’s full span (to preserve inner leading/trailing bytes).
    sub[0] = (node.start_byte, sub[0][1])
    sub[-1] = (sub[-1][0], node.end_byte)

    # Enforce HARD_CAP_BYTES post-check; if any segment exceeds, resplit that segment by bytes.
    normalized: List[Tuple[int, int]] = []
    for s, e in sub:
        if (e - s) > HARD_CAP_BYTES:
            normalized.extend(_newline_aligned_ranges(contents, s, e))
        else:
            normalized.append((s, e))
    return normalized


def _newline_aligned_ranges(contents: bytes, start: int, end: int, overlap: int = 0) -> List[Tuple[int, int]]:
    """
    Split [start, end) into segments targeting SOFT_MAX_BYTES each, then apply
    **size + overlap + nudge**:
      - choose a size-based boundary (cur + SOFT_MAX_BYTES),
      - nudge that single boundary to the nearest newline within ±NEWLINE_WINDOW,
      - next segment starts at (split - overlap).
    Ensures no segment exceeds HARD_CAP_BYTES and guarantees forward progress.
    """
    if start >= end:
        return []
    ranges: List[Tuple[int, int]] = []
    cur = start
    n = len(contents)
    step = SOFT_MAX_BYTES

    while cur < end:
        soft_end = min(cur + step, end)
        lo = max(cur + 1, soft_end - NEWLINE_WINDOW)
        hi = min(end - 1, soft_end + NEWLINE_WINDOW)
        split = _nearest_newline(contents, soft_end, lo, hi) or soft_end

        # Enforce hard cap and ensure progress.
        if (split - cur) > HARD_CAP_BYTES:
            split = cur + HARD_CAP_BYTES
        if split <= cur:
            split = min(end, cur + min(step, HARD_CAP_BYTES))

        ranges.append((cur, split))
        if split >= end:
            break

        next_start = split - (overlap or 0)
        if next_start <= cur:  # avoid stalling when overlap is too large
            next_start = split
        cur = next_start

    return ranges


def _nearest_newline(contents: bytes, target: int, lo: int, hi: int) -> Optional[int]:
    """
    Return the byte index of the newline closest to target in [lo, hi], or None if no newline.
    """
    left, right = target, target + 1
    while left >= lo or right <= hi:
        if left >= lo and contents[left:left + 1] == b"\n":
            return left + 1  # split after newline to keep it in the previous chunk
        if right <= hi and contents[right - 1:right] == b"\n":
            return right  # right is already after the newline byte
        left -= 1
        right += 1
    return None


def _bytes_to_char(ctx: DocContext, byte_offset: int) -> int:
    index = bisect.bisect_left(ctx.char_to_byte, byte_offset)
    if index >= len(ctx.char_to_byte):
        return len(ctx.char_to_byte) - 1
    if ctx.char_to_byte[index] > byte_offset and index > 0:
        return index - 1
    return index


def _make_chunk(
    contents: bytes,
    start: int,
    end: int,
    path: str,
    language: str,
    repo: str,
    mapper: LineMapper,
    signature: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:
    """
    Build a Chunk from a byte range, computing row/col positions efficiently.
    """
    start_rc = mapper.byte_to_point(start)
    end_rc = mapper.byte_to_point(end)
    text = contents[start:end].decode("utf-8", errors="replace")
    meta = metadata or {}
    return Chunk(
        chunk=text,
        repo=repo,
        path=path,
        language=language,
        start_rc=start_rc,
        end_rc=end_rc,
        start_bytes=start,
        end_bytes=end,
        signature=signature,
        metadata=meta,
    )
