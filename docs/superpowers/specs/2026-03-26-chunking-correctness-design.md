# Chunking Correctness Design

Date: 2026-03-26

## Context
Chunking correctness is the most critical part of this project. The core requirement is not just that files produce chunks, but that the grammars declared in `packages/core/src/Chunker/grammar_queries.json` actually capture the intended language structures with correct boundaries when executed with real Tree-sitter parsers.

The design goal is therefore to make grammar correctness the primary quality target for code chunking, while also ensuring strong correctness for non-code chunking behavior.

## Source of Truth
`packages/core/src/Chunker/grammar_queries.json` is the structural specification for code chunking.

It defines:
- supported code language mappings
- supported non-code grammar mappings
- the Tree-sitter queries for categories such as `Type`, `Method`, `Constructor`, `Initializer`, `Accessor`, `Field`, `EnumMember`, `AnnotationElement`, `RecordComponent`, and `Package`

Tests must treat this file as the canonical contract. We should not duplicate this contract in a separate hand-maintained spec.

## Correctness Standard

### Code languages
For code chunking, correctness means:
- the intended structures are actually captured by the configured grammar queries
- captured spans have the correct structural boundaries
- method-like chunks include intended leading documentation/comments where the chunker is designed to do so
- executable units stop at the correct end boundary
- adjacent methods/constructors/accessors/initializers do not collapse into each other
- top-level type metadata chunks remain separate from executable chunks and do not absorb method bodies
- fallback chunking is used only when the parser/query path or size rules require it

This is stricter than asserting that a query captures “something”. The tests must prove that the thing captured has the intended borders.

### Non-code
For non-code chunking, correctness means strong behavioral validation for:
- Markdown headings, fences, lists, tables, and blockquotes
- JSON / JSONL span extraction
- YAML / TOML sectioning behavior
- CSV / TSV table chunking
- plaintext and fallback chunking behavior
- requirement sentence extraction and related metadata where applicable

For non-code, the focus is on stable chunking behavior and metadata correctness, not Tree-sitter query categories.

## Testing Architecture

### 1. Query correctness tests
These tests validate the real grammar queries directly.

For each supported code language fixture:
1. parse the file with the real Tree-sitter parser
2. load the actual queries from `grammar_queries.json`
3. run the corresponding query category (`Type`, `Method`, `Constructor`, etc.)
4. assert that the expected structures are captured
5. assert that the captured spans match the intended structural unit boundaries

These are the primary grammar correctness tests.

### 2. Chunk boundary tests
These tests validate the transformation from captured structures into chunks.

They should assert:
- executable structures become isolated method-like chunks
- top-level types produce separate metadata chunks where expected
- signatures and scope markers reflect the intended container
- comment/doc inclusion begins at the intended leading boundary
- chunks stop at the correct border and do not absorb siblings
- fallback splitting only appears when size/cap conditions require it

These tests should remain semantic rather than snapshot-heavy, but must still be strict on span correctness.

### 3. Fixture completeness checks
We need visibility into whether the grammar spec is actually proven by fixtures.

This layer should:
- map the query categories declared in `grammar_queries.json` against the current local fixture set
- identify which language/category combinations are currently represented by proving fixtures
- identify missing structural cases and fixture gaps

This does not need to fail for every gap immediately, but it should give a clear inventory so missing cases can be filled deliberately.

### 4. Regression fixture runs
Keep broad real-fixture regression tests across supported languages.

These tests should verify:
- parsing does not unexpectedly regress
- chunk generation still succeeds for supported inputs
- broad invariants still hold

These are useful as smoke tests, but they are not sufficient on their own to prove grammar correctness.

## Fixture Strategy

### Current fixtures
The existing local fixtures in `packages/core/tests/fixtures/` should remain the base set because they are already integrated into the current repository and reflect current code assumptions.

Each fixture should ideally have a clear purpose:
- which categories it is meant to exercise
- which border conditions it is meant to prove
- which categories are intentionally absent

### Additional fixtures
Add new fixtures only when structurally necessary.

High-value additions include cases such as:
- adjacent methods that could merge incorrectly
- nested type/member cases
- constructor vs method ambiguities
- accessor/property edge forms
- enum / record / annotation specific forms
- comment/doc boundary edge cases
- language-specific structures that are valid but not currently covered

### External examples
External repositories can be used as curated sources for missing examples, but not as a blind corpus import.

Selection criteria:
- valid syntax
- minimal example files
- each file proves one or two structures clearly
- easy to reason about expected borders

The workflow should be:
1. map current local coverage
2. identify missing structure/border cases
3. add minimal new local fixtures
4. optionally seed those fixtures from external examples

## Failure Reporting
Failures should be actionable and specific.

Every grammar/chunking failure should identify:
- language
- fixture file
- query category
- whether the failure happened in parsing, query capture, or chunk materialization
- what boundary or structure was expected versus what was observed

This is important because correctness debugging in grammars is expensive if failures are vague.

## Success Criteria
The project should optimize for correctness confidence, not raw helper-branch coverage.

The acceptance standard is:
- strict grammar correctness tests for code languages
- strong non-code chunking behavior tests
- broad fixture-backed regression coverage
- explicit fixture-gap visibility for uncovered structures

This means we can be flexible on blanket `chunker.py` line coverage if the grammar correctness and non-code correctness suites are strong and specific.

## Recommended Next Implementation Steps
1. Adapt the strongest ideas from the deleted historical chunker tests into the current fixture layout.
2. Build dedicated query correctness tests that validate real Tree-sitter captures against `grammar_queries.json`.
3. Build strict chunk boundary tests for executable units and metadata chunks.
4. Expand non-code tests for Markdown, JSON/JSONL, YAML/TOML, CSV/TSV, and plaintext/fallback behavior.
5. Create a fixture coverage inventory for query-category completeness.
6. Add missing fixtures only where current examples do not prove the intended structures.

## Scope Check
This design is focused enough for a single implementation plan. It stays on chunking correctness and does not expand into unrelated refactoring.

## Ambiguity Resolution
A potential ambiguity was whether “correctness” means only “captures exist” or also “captured borders are correct”. This design resolves that explicitly in favor of span correctness.

A second ambiguity was whether to optimize for broad line coverage of `chunker.py`. This design resolves that in favor of correctness-centered testing rather than uniform branch chasing.
