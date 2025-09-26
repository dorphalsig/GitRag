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

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

from tree_sitter import Node, Query, QueryError, QueryCursor
from tree_sitter_language_pack import get_parser, get_language

# ---------------- Constants & Config ----------------
# Internal constants (no knobs).
SOFT_MAX_BYTES = 16_384  # packing target
HARD_CAP_BYTES = 24_576  # absolute per-chunk limit
NEWLINE_WINDOW = 2_048  # cut nudge window

FALLBACK_OVERLAP_RATIO = 0.10

_ROOT_ATOM_RE = re.compile(r"\(+\s*([A-Za-z_]\w*)\b")

CODE_EXTENSIONS = {"kt": "kotlin", "kts": "kotlin", "java": "java", "dart": "dart", "pas": "pascal", "pp": "pascal"}

NONCODE_TS_GRAMMAR = {"md": "markdown", "json": "json", "jsonl": "json", "ndjson": "json", "arb": "json",
                      "yaml": "yaml", "yml": "yaml", "lock": "yaml", "xml": "xml", "plist": "xml",
                      "entitlements": "xml", "storyboard": "xml", "xib": "xml", "toml": "toml",
                      "properties": "properties", "html": "html", "css": "css", "svg": "xml", "pro": "properties",
                      "cfg": "properties"}

# ========= Package / module-ish =========
PACKAGE_LIKE_SEXPR = {
    "java": [
        "(package_declaration) @package",
    ],
    "kotlin": [
        # Both spellings exist across grammar versions
        "(package_directive) @package",
        "(package_header) @package",
    ],
    "dart": [
        # Dart has library directives instead of packages
        "(library_declaration) @package",
    ],
    "pascal": [
        "(program (moduleName) @package)",
        "(library (moduleName) @package)",
        "(unit (moduleName) @package)"
    ]
}

# ========= Types (top-level) =========
TYPE_LIKE_SEXPR = {
    "java": [
        "(class_declaration) @type",
        "(interface_declaration) @type",
        "(enum_declaration) @type",
        "(record_declaration) @type",
        "(annotation_type_declaration) @type",
    ],
    "kotlin": [
        "(class_declaration) @type",
        "(object_declaration) @type",  # includes companion objects
        "(interface_declaration) @type",
        # enums are classes in Kotlin; class_declaration covers them
    ],
    "dart": [
        "(class_declaration) @type",
        "(mixin_declaration) @type",
        "(extension_declaration) @type",
        "(enum_declaration) @type",
    ],
    "pascal": [
        "(declClass) @type",  # class/record/object/objc*
        "(declIntf) @type",  # interface/dispinterface
        "(declHelper) @type",  # type/class/record helpers
        "(declEnum) @type",  # enum types
    ]
}

# ========= Constructors =========
CONSTRUCTOR_LIKE_SEXPR = {
    "java": [
        "(constructor_declaration) @constructor",
        "(compact_constructor_declaration) @constructor",  # Java records
    ],
    "kotlin": [
        "(primary_constructor) @constructor",
        "(secondary_constructor) @constructor",
    ],
    "dart": [
        "(constructor_declaration) @constructor",
        "(factory_constructor_declaration) @constructor",
        # Some grammar variants expose "(constructor_signature)" as well:
        "(constructor_signature) @constructor",
    ],
    "pascal": []
}

# ========= Methods / functions =========
# (We include top-level functions where they exist in the grammar.)
METHOD_LIKE_SEXPR = {
    "java": [
        "(method_declaration) @method",
    ],
    "kotlin": [
        "(function_declaration) @method",  # covers top-level & member funcs
    ],
    "dart": [
        "(method_declaration) @method",
        "(function_declaration) @method",  # top-level functions
    ],
    "pascal": [
        "(defProc) @method",  # procedures/functions/constructors/destructors with bodies
    ]
}

# ========= Fields / properties =========
FIELD_LIKE_SEXPR = {
    "java": [
        "(field_declaration) @field",
        "(constant_declaration) @field",  # interface/annotation constants
    ],
    "kotlin": [
        "(property_declaration) @field",  # properties (vals/vars)
    ],
    "dart": [
        "(field_declaration) @field",  # class fields
        "(top_level_variable_declaration) @field",  # top-level vars
    ],
    "pascal": [
        "(declField) @field",  # fields in classes/records
        "(declProp) @field",  # properties (signature only)
        "(declVar) @field",  # class/record vars in sections
        "(declConst) @field",  # class/record consts in sections
    ]
}

# ========= Accessors (getters/setters) =========
# Note: In Java, accessors are ordinary methods; in Dart, getters/setters
# are represented as method/method-signature forms. Kotlin exposes explicit
# accessor nodes attached to property_declaration in current grammars.
ACCESSOR_LIKE_SEXPR = {
    "java": [
        # Intentionally empty; model them via METHOD_LIKE_SEXPR
    ],
    "kotlin": [
        "(getter) @accessor",
        "(setter) @accessor",
        # Some grammar versions nest accessors; these keep you safe:
        "(property_declaration (getter) @accessor)",
        "(property_declaration (setter) @accessor)",
    ],
    "dart": [
        # Treat as methods (get/set keywords appear inside method nodes)
        # so METHOD_LIKE_SEXPR will catch them.
    ],
    "pascal": []
}

# ========= Initializers (static/instance/init blocks) =========
INITIALIZER_LIKE_SEXPR = {
    "java": [
        "(static_initializer) @initializer",
        "(instance_initializer) @initializer",
    ],
    "kotlin": [
        # Kotlin "init { ... }" blocks
        "(init_declaration) @initializer",  # common in fwcd grammar
        "(initializer) @initializer",  # alt name seen in some revs
    ],
    "dart": [
        # Dart has no standalone init block nodes; constructors cover init.
    ],
    "pascal": [
        "(initialization) @initializer",
        "(finalization) @initializer",
    ]
}

# ========= Enum members =========
ENUM_MEMBER_LIKE_SEXPR = {
    "java": [
        "(enum_constant) @enum_member",
    ],
    "kotlin": [
        # Enum entries are class members; usually appear as declarations
        "(enum_entry) @enum_member",
    ],
    "dart": [
        "(enum_constant) @enum_member",
    ],
    "pascal": [
        "(declEnumValue) @enum_member",
    ]
}

# ========= Annotation elements (Java's @interface methods) =========
ANNOTATION_ELEMENT_LIKE_SEXPR = {
    "java": [
        "(annotation_type_element_declaration) @annotation_element",
        # Some older grammars used:
        "(annotation_method_declaration) @annotation_element",
    ],
    "pascal": []
}

# ========= Record components (Java) =========
RECORD_COMPONENT_LIKE_SEXPR = {
    "java": [
        "(record_component) @record_component",
        # In some revisions: components appear inside a list node
        "(record_component_list (record_component) @record_component)",
    ],
    "kotlin": [],
    "dart": [],
    "pascal": []
}

# Centralized grammar queries
GRAMMAR_QUERIES = {
    "Package": PACKAGE_LIKE_SEXPR,
    "Type": TYPE_LIKE_SEXPR,
    "Constructor": CONSTRUCTOR_LIKE_SEXPR,
    "Method": METHOD_LIKE_SEXPR,
    "Field": FIELD_LIKE_SEXPR,
    "Accessor": ACCESSOR_LIKE_SEXPR,
    "Initializer": INITIALIZER_LIKE_SEXPR,
    "EnumMember": ENUM_MEMBER_LIKE_SEXPR,
    "AnnotationElement": ANNOTATION_ELEMENT_LIKE_SEXPR,
    "RecordComponent": RECORD_COMPONENT_LIKE_SEXPR,
}


@dataclass(frozen=True)
class Chunk:
    chunk: str
    repo: str
    path: str
    language: str
    start_rc: tuple[int, int]
    end_rc: tuple[int, int]
    start_bytes: int
    end_bytes: int
    signature: str = ""

    def id(self):
        id = f"{self.repo}::{self.path}::{self.start_bytes}::{self.end_bytes}"
        return hashlib.sha256(id.encode()).hexdigest()


def chunk_file(path: str, repo: str) -> List[Chunk]:
    file_extension = Path(path).suffix[1:]
    contents = Path(path).read_bytes()

    if lang := CODE_EXTENSIONS.get(file_extension, None):
        chunks = _chunk_code(contents, path, lang, repo)
    elif lang := NONCODE_TS_GRAMMAR.get(file_extension, None):
        chunks = _chunk_non_code(contents, path, lang, repo)
    else:
        chunks = _chunk_fallback(contents, path, "plain text", repo)

    return chunks


def _chunk_non_code(contents: bytes, path: str, language: str, repo: str) -> List[Chunk]:
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

    parser = get_parser(language)
    tree = parser.parse(contents)
    root = tree.root_node

    # Whole-file small case: emit a single chunk from the **original contents** and stop.
    if (root.end_byte - root.start_byte) < SOFT_MAX_BYTES:
        return [_make_chunk(contents, root.start_byte, root.end_byte, path, language, repo)]

    base = _first_multi_child_level(root)
    siblings = list(base.named_children)
    if not siblings:
        return _chunk_fallback(contents, path, language, repo)

    groups: List[Tuple[int, int]] = _pack_siblings(contents, siblings)
    if not groups:
        return _chunk_fallback(contents, path, language, repo)

    return [_make_chunk(contents, s, e, path, language, repo) for (s, e) in _stitch_full_coverage(contents, groups)]


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


def _chunk_fallback(contents: bytes, path: str, language: str, repo: str) -> List[Chunk]:
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
    return [_make_chunk(contents, s, e, path, language, repo) for s, e in ranges]


def _chunk_code(contents: bytes, path: str, language: str, repo: str) -> Iterable[Chunk]:
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
        return list(_chunk_fallback(contents, path, language, repo))

    root = tree.root_node
    pkg = _get_package_name(contents, root, language)

    types = _query_nodes(language, root, GRAMMAR_QUERIES["Type"].get(language, []), "type")
    methods = _query_nodes(language, root, GRAMMAR_QUERIES["Method"].get(language, []), "method")
    ctors = _query_nodes(language, root, GRAMMAR_QUERIES["Constructor"].get(language, []), "constructor")
    inits = _query_nodes(language, root, GRAMMAR_QUERIES["Initializer"].get(language, []), "initializer")
    accessors = _query_nodes(language, root, GRAMMAR_QUERIES["Accessor"].get(language, []), "accessor")

    chunks: List[Chunk] = []

    exec_nodes = sorted(_unique_by_span(methods + ctors + inits + accessors),
                        key=lambda n: (n.start_byte, n.end_byte))
    for n in exec_nodes:
        for ch in _build_method_like_chunks(n, contents, path, language, repo, pkg):
            chunks.append(ch)

    top_level_types = [t for t in types if _is_top_level_type(t, language)]
    fields = _query_nodes(language, root, GRAMMAR_QUERIES["Field"].get(language, []), "field")
    enums = _query_nodes(language, root, GRAMMAR_QUERIES["EnumMember"].get(language, []), "enum_member")
    anno_elems = _query_nodes(language, root, GRAMMAR_QUERIES["AnnotationElement"].get(language, []), "annotation_element")
    record_comps = _query_nodes(language, root, GRAMMAR_QUERIES["RecordComponent"].get(language, []), "record_component")

    for t in sorted(top_level_types, key=lambda n: (n.start_byte, n.end_byte)):
        chunks.append(_build_class_metadata_chunk(
            t, contents, path, language, repo, pkg,
            methods=exec_nodes, fields=fields, enums=enums,
            anno_elems=anno_elems, record_comps=record_comps
        ))

    # If Tree-sitter produced no actionable nodes, fall back to size-based chunking
    if not chunks:
        return list(_chunk_fallback(contents, path, language, repo))

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

def _query_nodes(language: str, root: Node, sexprs: List[str], capture: str) -> List[Node]:
    """
    Execute `sexprs` as a Tree-sitter Query and return nodes captured as `@{capture}`.
    - Supports multiple py-tree-sitter calling conventions:
      * QueryCursor().exec(query, root) -> iterable of (node, cap[, pattern_idx])
      * Query.captures(root) -> iterable of (node, cap)
      * Test doubles that return a dict: {cap: [nodes...]}
    - Deduplicates by (start,end,type) and sorts by source order.
    """
    if not sexprs:
        return []
    try:
        lang = get_language(language)
        q = Query(lang, "\n".join(s.strip() for s in sexprs if s.strip()))
    except QueryError:
        return []

    result = None
    # Try modern cursor.exec API first
    try:
        cursor = QueryCursor(q)
        result = list(cursor.exec(q, root))  # type: ignore[attr-defined]
    except Exception:
        # Try Query.captures(root) API
        try:
            result = list(q.captures(root))  # type: ignore[attr-defined]
        except Exception:
            # As a last resort, allow test doubles to provide a dict
            result = {}

    nodes: List[Node] = []
    if isinstance(result, dict):
        nodes = list(result.get(capture, []))
    else:
        # Normalize tuples of len 2 or 3: (node, cap[, pattern_idx])
        for item in result:  # type: ignore[assignment]
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            n = item[0]
            cap = item[1]
            if cap == capture:
                nodes.append(n)

    nodes = _unique_by_span(nodes)
    nodes.sort(key=lambda n: (n.start_byte, n.end_byte))
    return nodes

def _unique_by_span(nodes: List[Node]) -> List[Node]:
    """Deduplicate nodes by (start_byte, end_byte, type)."""
    seen, uniq = set(), []
    for n in nodes:
        k = (n.start_byte, n.end_byte, n.type)
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return uniq


def _build_method_like_chunks(node: Node, contents: bytes, path: str, language: str, repo: str, pkg: str) \
        -> List[Chunk]:
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
        return [_make_chunk(contents, unit_start, unit_end, path, language, repo, signature=signature)]

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
        chunks.append(Chunk(
            chunk=contents[start:end].decode("utf-8", errors="replace"),
            repo=repo, path=path, language=language,
            start_rc=_byte_to_point(contents, start), end_rc=_byte_to_point(contents, end),
            start_bytes=start, end_bytes=end, signature=(f"{fqn}#part{i + 1}" if len(parts) > 1 else fqn),
        ))
    return chunks


def _build_class_metadata_chunk(
        type_node: Node,
        contents: bytes,
        path: str,
        language: str,
        repo: str,
        pkg: str,
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
        chunk=meta_text, repo=repo, path=path, language=language,
        start_rc=_byte_to_point(contents, t_start), end_rc=_byte_to_point(contents, t_end),
        start_bytes=t_start, end_bytes=t_end,
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
    nodes = _query_nodes(language, root, sexprs, "package")
    if not nodes:
        return ""

    raw = contents[nodes[0].start_byte:nodes[0].end_byte].decode("utf-8", errors="replace").strip()
    if raw.endswith(";"):
        raw = raw[:-1].rstrip()

    m = re.match(r"^(?:package|library|program)\s+([A-Za-z_][\w.]*)(?:\s*)$", raw)
    name = m.group(1) if m else raw

    name = re.sub(r"[^A-Za-z0-9_.]+", ".", name)
    name = re.sub(r"\.{2,}", ".", name).strip(".")
    return name


def _fqn_for_node(node: Node, contents: bytes, path: str, language: str, pkg: str) -> str:
    """
    Build a best-effort FQDN: package + enclosing types or file stem + name(params),
    and append '|member|lang' or '|top_level|lang'.
    """
    name = _node_identifier_text(contents, node) or ""
    encl = _enclosing_types(node, contents)
    container = "member" if encl else "top_level"
    if not name:
        name = (encl[-1] if encl else Path(path).stem)
    head = contents[node.start_byte:min(node.end_byte, node.start_byte + 512)].decode("utf-8", errors="replace")
    m = re.search(r"\((.*?)\)", head, flags=re.DOTALL)
    params = m.group(1).strip() if m else ""
    scope = ".".join(encl) if encl else Path(path).stem
    base = f"{pkg}.{scope}.{name}" if pkg else f"{scope}.{name}"
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


def _make_chunk(contents: bytes, start: int, end: int, path: str, language: str, repo: str, signature="") -> Chunk:
    """
    Build a Chunk from a byte range, computing row/col positions efficiently.
    """
    start_rc = _byte_to_point(contents, start)
    end_rc = _byte_to_point(contents, end)
    text = contents[start:end].decode("utf-8", errors="replace")
    return Chunk(
        chunk=text,
        repo=repo,
        path=path,
        language=language,
        start_rc=start_rc,
        end_rc=end_rc,
        start_bytes=start,
        end_bytes=end,
        signature=signature
    )


def _byte_to_point(contents: bytes, index: int) -> Tuple[int, int]:
    """
    Convert a byte offset to (row, col) with 0-based row/col using newline counts.
    """
    row = contents.count(b"\n", 0, index)
    last_nl = contents.rfind(b"\n", 0, index)
    col = index if last_nl == -1 else index - (last_nl + 1)
    return (row, col)
