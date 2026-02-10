# Shared test fixtures utilities.
# Provides deterministic byte generators and on-disk fixture management
# so multiple test modules can reuse the same data without duplication.

from __future__ import annotations

import random
from pathlib import Path

# Directory for on-disk byte fixtures used by tests
FIXTURES_DIR = Path(__file__).with_name("fixtures")
FIXTURES_DIR.mkdir(exist_ok=True)


def rand_bytes(n: int, rate: float = 0.05, crlf: bool = False, seed: int = 42) -> bytes:
    """Generate deterministic ASCII-ish bytes with occasional newlines.

    - n: total length in bytes
    - rate: probability of inserting a newline at each step
    - crlf: if True, inserts CRLF; otherwise LF
    - seed: RNG seed for determinism
    """
    rnd = random.Random(seed)
    out = bytearray()
    for _ in range(n):
        if rnd.random() < rate:
            out += (b"\r\n" if crlf else b"\n")
        else:
            out.append(rnd.choice(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"))
    return bytes(out)


def ensure_fixtures() -> None:
    """Create deterministic byte-file fixtures under tests/fixtures (idempotent).

    In addition to general-purpose text fixtures, generate a small random file for
    any non-code format supported by the chunker that doesn't already have a
    corresponding sample in tests/fixtures.
    """
    small = FIXTURES_DIR / "no_newlines_small.txt"
    if not small.exists():
        small.write_bytes(b"A" * 128)

    large = FIXTURES_DIR / "no_newlines_large.txt"
    if not large.exists():
        large.write_bytes(b"B" * 150_000)

    crlf = FIXTURES_DIR / "crlf.txt"
    if not crlf.exists():
        crlf.write_bytes(rand_bytes(16_384, rate=0.08, crlf=True, seed=7))

    tiny = FIXTURES_DIR / "tiny.txt"
    if not tiny.exists():
        tiny.write_bytes(b"XY")

    huge = FIXTURES_DIR / "huge_random.txt"
    if not huge.exists():
        huge.write_bytes(rand_bytes(300_000, rate=0.02, crlf=False, seed=99))

    # Create random files for each missing non-code grammar extension
    try:
        # Local import to avoid overhead outside tests; safe due to helper stubs in unit tests
        from src import Chunker as _chunker  # type: ignore
        noncode_exts = set(_chunker.NONCODE_TS_GRAMMAR.keys())
    except Exception:
        # Fallback to a hardcoded superset matching the current chunker.NONCODE_TS_GRAMMAR keys
        noncode_exts = {
            "md", "json", "jsonl", "ndjson", "arb",
            "yaml", "yml", "lock", "xml", "plist", "entitlements",
            "storyboard", "xib", "toml", "properties", "html", "css", "svg", "pro", "cfg"
        }

    # Identify already-present samples for these extensions
    present_exts = {p.suffix[1:].lower() for p in FIXTURES_DIR.glob("*.*")}
    missing = [ext for ext in noncode_exts if ext not in present_exts]

    for ext in missing:
        # Name files as random_noncode.<ext>
        fname = FIXTURES_DIR / f"random_noncode.{ext}"
        try:
            # Keep them reasonably small; Tree-sitter will still parse/error with these
            n = 2048 if ext not in {"md", "html", "xml"} else 4096
            fname.write_bytes(rand_bytes(n, rate=0.03, crlf=False, seed=13))
        except Exception:
            # Best-effort: ensure the file exists even if randomness fails
            fname.write_bytes(b"sample\n")


def load_bytes(name: str) -> bytes:
    """Load a named fixture file from tests/fixtures directory."""
    return (FIXTURES_DIR / name).read_bytes()


__all__ = [
    "FIXTURES_DIR",
    "rand_bytes",
    "ensure_fixtures",
    "load_bytes",
]
