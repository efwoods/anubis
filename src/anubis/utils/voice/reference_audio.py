"""Store and read the avatar's reference-audio clip.

The reference clip is one short, single-speaker recording the diarizer uses to
find the avatar's voice in later uploads.

The clip lives in the LangGraph store under the namespace
``(user_id, assistant_id, "reference_audio")`` with key ``assistant_id`` and the
value ``{"reference_audio_data": <data URI>, "document": Document.to_json()}``.
The serialized Document carries ``filename`` / ``namespace_filename`` metadata so
the clip shows up in ``/list_avatar_documents`` and can be deleted there.

Rules enforced here:

- The first audio or video upload for an avatar becomes the reference; later
  uploads never replace the clip (``store_reference_audio`` with
  ``replace=False``). A per-avatar lock serializes concurrent items of one
  batch so exactly one of them writes.
- The owner can point the reference at a different upload explicitly
  (``replace=True``), which only rewrites this store row.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from src.anubis.utils.store_cache import invalidate_store_cache_entry

logger = logging.getLogger(__name__)

REFERENCE_AUDIO_NAMESPACE_CATEGORY = "reference_audio"

_reference_audio_locks: dict[tuple[str, str], asyncio.Lock] = {}


def reference_audio_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the store namespace holding the avatar's reference clip."""
    return (user_id, assistant_id, REFERENCE_AUDIO_NAMESPACE_CATEGORY)


def reference_audio_lock(user_id: str, assistant_id: str) -> asyncio.Lock:
    """One lock per avatar so concurrent uploads agree on a single reference."""
    lock_key = (user_id, assistant_id)
    lock = _reference_audio_locks.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _reference_audio_locks[lock_key] = lock
    return lock


async def read_reference_audio(
    store: Any, user_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Return the stored reference clip, or ``None`` when the avatar has none.

    The returned dictionary has ``audio_data_uri``, ``transcript_text``,
    ``filename``, ``namespace_filename`` and ``duration_seconds``.
    """
    try:
        item = await store.aget(reference_audio_namespace(user_id, assistant_id), assistant_id)
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Reference audio lookup failed (continuing): %s", read_error)
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    value = value or {}
    document = value.get("document") or {}
    document_kwargs = document.get("kwargs") if isinstance(document, dict) else None
    if not isinstance(document_kwargs, dict):
        # Rows written before the Document shape was unified carried a plain
        # ``{"page_content", "metadata"}`` dictionary.
        document_kwargs = document if isinstance(document, dict) else {}
    metadata = document_kwargs.get("metadata") or {}
    return {
        "audio_data_uri": value.get("reference_audio_data") or None,
        "transcript_text": document_kwargs.get("page_content") or "",
        "filename": metadata.get("filename"),
        "namespace_filename": metadata.get("namespace_filename"),
        "duration_seconds": metadata.get("duration"),
    }


def build_reference_audio_document(
    *,
    user_id: str,
    assistant_id: str,
    transcript_text: str,
    filename: str | None,
    namespace_filename: str | None,
    duration_seconds: float | None,
    source: str,
) -> Document:
    """Build the Document stored beside the reference clip (listable, deletable)."""
    return Document(
        page_content=transcript_text or "",
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "processing_task_id": str(uuid4()),
            "type": "audio",
            "reference_audio": True,
            "duration": duration_seconds,
            "filename": filename,
            "namespace": REFERENCE_AUDIO_NAMESPACE_CATEGORY,
            "source": source,
            "vectorstore_acceptable": False,
            "adapter_acceptable": False,
            "analysis_acceptable": False,
            "namespace_filename": namespace_filename,
        },
    )


async def store_reference_audio(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    audio_data_uri: str,
    transcript_text: str,
    filename: str | None,
    namespace_filename: str | None,
    duration_seconds: float | None,
    source: str,
    replace: bool = False,
    lock_already_held: bool = False,
) -> Document | None:
    """Store the reference clip and return the Document written.

    With ``replace=False`` the write happens only when the avatar has no
    reference yet, and ``None`` is returned when one already exists. With
    ``replace=True`` the existing clip is overwritten. A caller that already
    holds ``reference_audio_lock`` for the avatar (the media pipeline, which
    isolates the clip under the lock) passes ``lock_already_held=True``; the
    lock is not re-entrant.
    """
    if lock_already_held:
        return await _write_reference_audio(
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            audio_data_uri=audio_data_uri,
            transcript_text=transcript_text,
            filename=filename,
            namespace_filename=namespace_filename,
            duration_seconds=duration_seconds,
            source=source,
            replace=replace,
        )
    async with reference_audio_lock(user_id, assistant_id):
        return await _write_reference_audio(
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            audio_data_uri=audio_data_uri,
            transcript_text=transcript_text,
            filename=filename,
            namespace_filename=namespace_filename,
            duration_seconds=duration_seconds,
            source=source,
            replace=replace,
        )


async def _write_reference_audio(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    audio_data_uri: str,
    transcript_text: str,
    filename: str | None,
    namespace_filename: str | None,
    duration_seconds: float | None,
    source: str,
    replace: bool,
) -> Document | None:
    """Check-and-write without locking; callers hold ``reference_audio_lock``."""
    namespace = reference_audio_namespace(user_id, assistant_id)
    if not replace:
        existing = await read_reference_audio(store, user_id, assistant_id)
        if existing is not None:
            return None
    document = build_reference_audio_document(
        user_id=user_id,
        assistant_id=assistant_id,
        transcript_text=transcript_text,
        filename=filename,
        namespace_filename=namespace_filename,
        duration_seconds=duration_seconds,
        source=source,
    )
    await store.aput(
        namespace,
        key=assistant_id,
        value={
            "reference_audio_data": audio_data_uri,
            "document": document.to_json(),
        },
    )
    invalidate_store_cache_entry(namespace, assistant_id)
    logger.info(
        "Reference audio for %s set to %s (%s)", assistant_id, filename, source
    )
    return document
