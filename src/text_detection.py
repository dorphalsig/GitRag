from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


PRINTABLE_BYTES = set(b"\t\n\r\f\b" + bytes(range(32, 127)))


class BinaryDetector:
    """Classify files as binary/text using git attributes and byte heuristics."""

    def __init__(self, git_runner: Optional[Callable[[list[str]], str]] = None, base_dir: Path | str | None = None) -> None:
        self._git_runner = git_runner
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    def is_binary(self, path: str) -> bool:
        attr = self._git_attr_binary(path)
        if attr is True:
            return True
        if attr is False:
            return False

        sample = self._read_sample(path)
        if sample is None:
            return False
        if b"\x00" in sample:
            return True
        non_printable = sum(1 for b in sample if b not in PRINTABLE_BYTES)
        return non_printable / max(len(sample), 1) > 0.30

    def _git_attr_binary(self, path: str) -> Optional[bool]:
        if self._git_runner is None:
            return None
        try:
            out = self._git_runner(["check-attr", "binary", "--", path]).strip()
        except Exception:
            return None
        if not out:
            return None
        # Format: "path: binary: value"
        parts = [p.strip() for p in out.split(":")]
        if not parts:
            return None
        value = parts[-1].lower()
        if value == "set":
            return True
        if value in {"unset", "unspecified", "false"}:
            return False
        return None

    def _read_sample(self, path: str) -> Optional[bytes]:
        p = Path(path)
        if not p.is_absolute():
            p = self._base_dir / p
        try:
            with p.open("rb") as fh:
                return fh.read(8192)
        except FileNotFoundError:
            return None
        except OSError:
            return None
