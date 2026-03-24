from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
class BinaryDetector:
    """Classify files as binary/text using git attributes and Magika AI."""
    _shared_magika = None

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        # Instantiate the model once during initialization to minimize overhead


    @classmethod
    def _get_magika(cls):
        """Lazy-load the Magika model only when first needed."""
        if cls._shared_magika is None:
            # Import here to avoid import-time penalty for the whole module
            from magika import Magika
            cls._shared_magika = Magika()
        return cls._shared_magika

    def is_binary(self, path: str) -> bool:
        p = Path(path)
        magika = self._get_magika()
        if not p.is_absolute():
            p = self._base_dir / p

        if not p.exists() or not p.is_file():
            return False

        try:
            result = magika.identify_path(p)
            return not result.output.is_text
        except Exception as e:
            logging.error("An error occured while identifying file as binary: %s", e, exc_info=True)
            return False