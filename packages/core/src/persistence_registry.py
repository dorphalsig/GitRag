"""Simple registry for persistence adapter factories."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from Persist import PersistenceAdapter, PersistConfig

AdapterFactory = Callable[..., "PersistenceAdapter"]

_REGISTRY: Dict[str, AdapterFactory] = {}


def register_persistence_adapter(name: str, factory: AdapterFactory) -> None:
    key = _normalize(name)
    if not key:
        raise ValueError("Adapter name must be non-empty")
    if not callable(factory):
        raise TypeError("Adapter factory must be callable")
    _REGISTRY[key] = factory


def unregister_persistence_adapter(name: str) -> None:
    key = _normalize(name)
    _REGISTRY.pop(key, None)


def get_persistence_adapter(name: str) -> Optional[AdapterFactory]:
    key = _normalize(name)
    return _REGISTRY.get(key)


def available_persistence_adapters() -> Iterable[str]:
    return sorted(_REGISTRY.keys())


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


__all__ = [
    "register_persistence_adapter",
    "unregister_persistence_adapter",
    "get_persistence_adapter",
    "available_persistence_adapters",
]
