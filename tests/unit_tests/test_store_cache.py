"""Read-through store cache used by ``load_consciousness`` for static entries.

The reference image and style profile change only on upload, recalibration, or
deletion, so ``load_consciousness`` reads both through ``aget_through_cache``.
These tests cover the contract: one store round-trip per key until invalidation
or expiry, cached ``None`` misses, explicit per-entry and per-assistant
invalidation, and the least-recently-used eviction bound.
"""

import pytest

import src.anubis.utils.store_cache as store_cache
from src.anubis.utils.store_cache import (
    aget_through_cache,
    invalidate_store_cache_entry,
    invalidate_store_cache_for_assistant,
)


class _CountingStore:
    """Fake store recording every ``aget`` and serving from a plain dict."""

    def __init__(self, contents: dict | None = None):
        self.contents = contents or {}
        self.aget_calls: list[tuple[tuple, str]] = []

    async def aget(self, namespace: tuple, key: str):
        self.aget_calls.append((tuple(namespace), key))
        return self.contents.get((tuple(namespace), key))


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts and ends with an empty process-wide cache."""
    store_cache._store_cache.clear()
    yield
    store_cache._store_cache.clear()


NAMESPACE = ("creator-1", "assistant-1", "reference_image")
KEY = "assistant-1"


@pytest.mark.asyncio
async def test_second_read_is_served_from_cache():
    store = _CountingStore({(NAMESPACE, KEY): "image-item"})

    first = await aget_through_cache(store, NAMESPACE, KEY)
    second = await aget_through_cache(store, NAMESPACE, KEY)

    assert first == "image-item"
    assert second == "image-item"
    assert len(store.aget_calls) == 1


@pytest.mark.asyncio
async def test_missing_item_is_cached_as_none():
    store = _CountingStore()

    assert await aget_through_cache(store, NAMESPACE, KEY) is None
    assert await aget_through_cache(store, NAMESPACE, KEY) is None
    assert len(store.aget_calls) == 1


@pytest.mark.asyncio
async def test_invalidate_entry_forces_refetch():
    store = _CountingStore({(NAMESPACE, KEY): "old-image"})
    await aget_through_cache(store, NAMESPACE, KEY)

    store.contents[(NAMESPACE, KEY)] = "new-image"
    invalidate_store_cache_entry(NAMESPACE, KEY)

    assert await aget_through_cache(store, NAMESPACE, KEY) == "new-image"
    assert len(store.aget_calls) == 2


@pytest.mark.asyncio
async def test_invalidate_for_assistant_drops_only_that_assistant():
    other_namespace = ("assistant-2", "style_profile")
    store = _CountingStore(
        {
            (NAMESPACE, KEY): "image-item",
            (other_namespace, "style_profile"): "profile-item",
        }
    )
    await aget_through_cache(store, NAMESPACE, KEY)
    await aget_through_cache(store, other_namespace, "style_profile")

    invalidate_store_cache_for_assistant("assistant-1")

    await aget_through_cache(store, NAMESPACE, KEY)
    await aget_through_cache(store, other_namespace, "style_profile")
    # assistant-1's entry was refetched; assistant-2's entry was still cached.
    assert len(store.aget_calls) == 3


@pytest.mark.asyncio
async def test_expired_entry_is_refetched(monkeypatch):
    store = _CountingStore({(NAMESPACE, KEY): "image-item"})
    await aget_through_cache(store, NAMESPACE, KEY)

    cached_at, cached_item = store_cache._store_cache[(NAMESPACE, KEY)]
    store_cache._store_cache[(NAMESPACE, KEY)] = (
        cached_at - store_cache.STORE_CACHE_MAX_AGE_SECONDS - 1,
        cached_item,
    )

    await aget_through_cache(store, NAMESPACE, KEY)
    assert len(store.aget_calls) == 2


@pytest.mark.asyncio
async def test_least_recently_used_entry_is_evicted(monkeypatch):
    monkeypatch.setattr(store_cache, "STORE_CACHE_MAX_ENTRIES", 2)
    store = _CountingStore(
        {
            (("a",), "k"): 1,
            (("b",), "k"): 2,
            (("c",), "k"): 3,
        }
    )

    await aget_through_cache(store, ("a",), "k")
    await aget_through_cache(store, ("b",), "k")
    # Touch ("a",) so ("b",) becomes the least-recently-used entry.
    await aget_through_cache(store, ("a",), "k")
    await aget_through_cache(store, ("c",), "k")

    assert (("b",), "k") not in store_cache._store_cache
    assert (("a",), "k") in store_cache._store_cache
    assert (("c",), "k") in store_cache._store_cache
