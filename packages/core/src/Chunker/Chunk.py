import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
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