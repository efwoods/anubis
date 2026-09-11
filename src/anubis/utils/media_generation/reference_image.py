"""Store and read the avatar's reference image (its portrait).

The portrait lives in the LangGraph store under the namespace
``(user_id, assistant_id, "reference_image")`` with key ``assistant_id`` and the
value ``{"reference_image_data": <data URI>, "document": Document.to_json(),
**assessment_store_fields(...)}``. It is not a table row: the bytes are a
base64 data URI inside the store value, which is why the store's vector index
is configured to embed ``document.kwargs.page_content`` alone.

This module exists for one rule the raw ``store.aput`` could not express. The
deep-research acquisition may find a photograph of the subject and install it as
the portrait, and that acquisition runs at the same time as the creator's own
upload from the avatar-creation screen. A picture the creator chose must always
win, so the acquisition writes with ``replace=False`` and a per-avatar lock
serializes the two writers. Every other caller keeps the previous behaviour by
writing with ``replace=True``.

``source_url`` records where an acquired photograph came from. A portrait the
creator uploaded has no source URL; one the research downloaded does, so its
provenance stays auditable and a takedown request can be acted on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.anubis.utils.store_cache import invalidate_store_cache_entry

logger = logging.getLogger(__name__)

REFERENCE_IMAGE_NAMESPACE_CATEGORY = "reference_image"

_reference_image_locks: dict[tuple[str, str], asyncio.Lock] = {}


def reference_image_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the store namespace holding the avatar's portrait."""
    return (user_id, assistant_id, REFERENCE_IMAGE_NAMESPACE_CATEGORY)


def reference_image_lock(user_id: str, assistant_id: str) -> asyncio.Lock:
    """One lock per avatar so concurrent writers agree on a single portrait."""
    lock_key = (user_id, assistant_id)
    lock = _reference_image_locks.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _reference_image_locks[lock_key] = lock
    return lock


async def read_reference_image(
    store: Any, user_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Return the stored portrait's value, or ``None`` when the avatar has none.

    A row whose ``reference_image_data`` is empty counts as no portrait: that is
    what a half-written or cleared row looks like, and treating it as present
    would keep the avatar permanently faceless.
    """
    try:
        item = await store.aget(
            reference_image_namespace(user_id, assistant_id), assistant_id
        )
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Reference image lookup failed (continuing): %s", read_error)
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    value = value or {}
    if not str(value.get("reference_image_data") or "").strip():
        return None
    return dict(value)


async def store_reference_image(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    image_data_uri: str,
    document_json: Any,
    assessment_fields: dict[str, Any] | None = None,
    source_url: str | None = None,
    replace: bool = True,
    lock_already_held: bool = False,
) -> bool:
    """Write the portrait; return whether this call is the one that wrote it.

    With ``replace=True`` — every path where a person chose the picture — the
    portrait is written unconditionally, exactly as before this module existed.
    With ``replace=False`` the write happens only when the avatar has no
    portrait yet, and ``False`` is returned when one already exists; that is the
    deep-research acquisition declining to overwrite the creator's own choice.

    A caller already holding ``reference_image_lock`` for this avatar passes
    ``lock_already_held=True``; the lock is not re-entrant.
    """
    if lock_already_held:
        return await _write_reference_image(
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            image_data_uri=image_data_uri,
            document_json=document_json,
            assessment_fields=assessment_fields,
            source_url=source_url,
            replace=replace,
        )
    async with reference_image_lock(user_id, assistant_id):
        return await _write_reference_image(
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            image_data_uri=image_data_uri,
            document_json=document_json,
            assessment_fields=assessment_fields,
            source_url=source_url,
            replace=replace,
        )


async def _write_reference_image(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    image_data_uri: str,
    document_json: Any,
    assessment_fields: dict[str, Any] | None,
    source_url: str | None,
    replace: bool,
) -> bool:
    """Check-and-write without locking; callers hold ``reference_image_lock``."""
    namespace = reference_image_namespace(user_id, assistant_id)
    if not replace:
        existing_portrait = await read_reference_image(store, user_id, assistant_id)
        if existing_portrait is not None:
            logger.info(
                "Avatar %s already holds a portrait; leaving it in place.",
                assistant_id,
            )
            return False
    value: dict[str, Any] = {
        "reference_image_data": image_data_uri,
        "document": document_json,
        **(assessment_fields or {}),
    }
    if source_url:
        value["source_url"] = source_url
    await store.aput(namespace, key=assistant_id, value=value)
    invalidate_store_cache_entry(namespace, assistant_id)
    return True


__all__ = [
    "REFERENCE_IMAGE_NAMESPACE_CATEGORY",
    "read_reference_image",
    "reference_image_lock",
    "reference_image_namespace",
    "store_reference_image",
]
