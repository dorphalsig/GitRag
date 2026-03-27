from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

# Adjusting import for GitRag structure
from Chunker.chunker import CODE_EXTENSIONS, GRAMMAR_QUERIES, _query_matches

FIXTURES = Path(__file__).parent / "fixtures"

CAPTURE_NAMES = {
    "Package": "package",
    "Type": "type",
    "Method": "method",
    "Constructor": "constructor",
    "Field": "field",
    "Accessor": "accessor",
    "Initializer": "initializer",
    "EnumMember": "enum_member",
    "AnnotationElement": "annotation_element",
    "RecordComponent": "record_component",
}


def _fixture_languages():
    result = {}
    for path in FIXTURES.iterdir():
        ext = path.suffix.lower().lstrip(".")
        language = CODE_EXTENSIONS.get(ext)
        if language:
            result.setdefault(language, []).append(path)
    return result


def test_fixture_inventory_lists_query_categories_with_local_proofs():
    fixture_languages = _fixture_languages()
    proven = {}
    for category, lang_map in GRAMMAR_QUERIES.items():
        capture = CAPTURE_NAMES.get(category, category.lower())
        for language, patterns in lang_map.items():
            if not patterns or language not in fixture_languages:
                continue
            try:
                parser = get_parser(language)
            except Exception:
                continue
            total = 0
            for fixture in fixture_languages[language]:
                contents = fixture.read_bytes()
                root = parser.parse(contents).root_node
                total += len(_query_matches(root, language, patterns, capture))
            proven[(language, category)] = total
    assert proven


def test_fixture_inventory_reports_missing_language_category_pairs():
    fixture_languages = _fixture_languages()
    missing = []
    proven = []

    for category, lang_map in GRAMMAR_QUERIES.items():
        capture = CAPTURE_NAMES.get(category, category.lower())
        for language, patterns in lang_map.items():
            if not patterns:
                continue
            fixtures = fixture_languages.get(language, [])
            if not fixtures:
                missing.append((language, category, "no_fixture"))
                continue

            try:
                parser = get_parser(language)
            except Exception:
                continue
            count = 0
            for fixture in fixtures:
                contents = fixture.read_bytes()
                root = parser.parse(contents).root_node
                count += len(_query_matches(root, language, patterns, capture))

            if count:
                proven.append((language, category, count))
            else:
                missing.append((language, category, "no_capture"))

    assert proven
    if missing:
        pytest.fail(
            "Missing fixture/query coverage:\n"
            + "\n".join(f"  {lang}:{cat}:{reason}" for lang, cat, reason in missing)
        )
