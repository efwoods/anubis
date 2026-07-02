"""Process-wide read-through cache for single-key LangGraph store lookups.

``load_consciousness`` re-reads two effectively static store entries on every
message: the avatar's descriptive reference image
(``(creator_user_id, assistant_id, "reference_image")``) and the stylometric
style profile (``(assistant_id, "style_profile")``). Both entries change only
when media is uploaded, recalibrated, or deleted — never during normal
conversation — so re-fetching both entries per message wastes one store
round-trip each.

This module caches those exact-key ``store.aget`` results in process memory and
exposes explicit invalidation hooks that every write and delete site calls:

* reference image write — ``src/subgraphs/process_media_graph/utils/nodes.py``
* style profile write — ``src/subgraphs/process_media_graph/utils/calibrate_ground_truth.py``
* document / avatar deletion (raw SQL, bypasses the store client) —
  ``src/api/webapp.py`` (``/delete_avatar_document`` and ``/delete_avatar``)

The FastAPI webapp and every LangGraph graph run inside the same
``langgraph-api`` server process, so in-process invalidation reaches all
writers and readers. ``STORE_CACHE_MAX_AGE_SECONDS`` is a safety net for a
multi-replica deployment where a write in one replica cannot invalidate
another replica's cache: staleness is bounded by the maximum age.

Entries are evicted least-recently-used beyond ``STORE_CACHE_MAX_ENTRIES``.
The bound matters because a cached reference-image item carries the full
image data URI (multiple megabytes per avatar).
"""

import time
from collections import OrderedDict
from typing import Any

# Upper bound on staleness when another process wrote the store and could not
# invalidate this process's cache.
STORE_CACHE_MAX_AGE_SECONDS: float = 300.0

# Upper bound on resident entries; reference-image values are multi-megabyte,
# so the bound keeps the worst case at tens-of-megabytes scale.
STORE_CACHE_MAX_ENTRIES: int = 64

# Maps (namespace tuple, key) -> (monotonic time cached, store item).
# Ordered so the least-recently-used entry sits first for eviction.
_store_cache: OrderedDict[tuple[tuple, str], tuple[float, Any]] = OrderedDict()


def _cache_key(namespace: tuple, key: str) -> tuple[tuple, str]:
    return (tuple(namespace), key)


async def aget_through_cache(store, namespace: tuple, key: str) -> Any:
    """Read one store item through the cache, fetching on miss or expiry.

    A ``None`` result (the item does not exist) is cached as well, so avatars
    without a reference image or style profile do not pay the round-trip on
    every message; the corresponding write sites invalidate the cached miss
    the moment the item is first created.
    """
    cache_key = _cache_key(namespace, key)
    cached_entry = _store_cache.get(cache_key)
    if cached_entry is not None:
        cached_at, cached_item = cached_entry
        if time.monotonic() - cached_at < STORE_CACHE_MAX_AGE_SECONDS:
            _store_cache.move_to_end(cache_key)
            return cached_item

    fetched_item = await store.aget(namespace, key)
    _store_cache[cache_key] = (time.monotonic(), fetched_item)
    _store_cache.move_to_end(cache_key)
    while len(_store_cache) > STORE_CACHE_MAX_ENTRIES:
        _store_cache.popitem(last=False)
    return fetched_item


def invalidate_store_cache_entry(namespace: tuple, key: str) -> None:
    """Drop one cached entry after the underlying store item was written or deleted."""
    _store_cache.pop(_cache_key(namespace, key), None)


def invalidate_store_cache_for_assistant(assistant_id: str) -> None:
    """Drop every cached entry whose namespace references the assistant.

    Used by avatar-wide deletions (``/delete_avatar`` removes every store row
    mentioning the assistant via raw SQL, so no per-entry hook fires).
    """
    stale_cache_keys = [
        cache_key
        for cache_key in _store_cache
        if assistant_id in cache_key[0]
    ]
    for cache_key in stale_cache_keys:
        del _store_cache[cache_key]
