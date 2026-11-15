"""Optional memoization of label-free upstream HELM parsing, never fitted features."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache


@contextmanager
def native_monomer_cache(settings: dict) -> Iterator[dict]:
    if not settings.get("cache_native_monomers", False):
        yield {"cache_native_monomers": False}
        return
    capacity = settings.get("max_entries", 262144)
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Native monomer cache capacity must be a positive integer")
    import oligogym.features as features

    original = features._extract_monomers

    @lru_cache(maxsize=capacity)
    def parsed(helm, strands):
        return original(helm, list(strands) if strands is not None else None)

    def copied(helm, strands=None):
        # Upstream feature methods mutate frames; callers must receive fresh copies.
        return parsed(helm, tuple(strands) if strands is not None else None).copy(
            deep=True
        )

    features._extract_monomers = copied
    info = {
        "cache_native_monomers": True,
        "max_entries": capacity,
        "scope": "raw HELM/strand parsing only; no labels or fitted dictionaries",
    }
    try:
        yield info
    finally:
        features._extract_monomers = original
        info.update(parsed.cache_info()._asdict())
        parsed.cache_clear()
