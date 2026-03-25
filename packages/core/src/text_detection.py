from __future__ import annotations

import logging
from pathlib import Path

from magika import Magika

_magika = Magika()
logger = logging.getLogger(__name__)
class BinaryDetector:
    """Classify files as binary/text using git attributes and Magika AI."""


    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self._magika = _magika

    def is_binary(self, path: str) -> bool:
        p = Path(path)
        if not p.is_absolute():
            p = self._base_dir / p

        if not p.exists() or not p.is_file():
            return False

        try:
            result = self._magika.identify_path(p)
            return not result.output.is_text
        except Exception as e:
            logging.error("An error occured while identifying file as binary: %s", e, exc_info=True)
            return False