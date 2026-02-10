import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from Calculators.EmbeddingCalculator import EmbeddingCalculator


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
    branch: str | None = None
    signature: str = ""
    embeddings: Optional[bytes] = field(default=None, compare=False, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def id(self):
        branch = self.branch or ""
        id = f"{self.repo}::{branch}::{self.path}::{self.start_bytes}::{self.end_bytes}"
        return hashlib.sha256(id.encode()).hexdigest()

    def calculate_embeddings(self, calculator: EmbeddingCalculator) -> None:
        """
        Mutates this frozen instance by setting `self.embeddings` using the provided
        calculator. The calculator must implement `calculate(str) -> bytes` (or
        bytes-like). Returns None.
        """
        raw = calculator.calculate(self.chunk)
        if isinstance(raw, (bytearray, memoryview)):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            raise TypeError(
                "calculator.calculate(...) must return bytes or bytes-like"
            )
        object.__setattr__(self, "embeddings", raw)
