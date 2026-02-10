from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingCalculator(Protocol):
    def calculate(self, chunk: str) -> bytes:
        pass