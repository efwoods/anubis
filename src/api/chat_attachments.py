"""The raw files a chat turn carried, kept for the identity-update tool.

``POST /message/{assistant_id}`` turns attachments into message content (images
become image blocks, documents become text) and the bytes are gone by the time
the graph runs. The in-chat ``update_avatar_identity_with_media`` tool needs
those bytes exactly as uploaded, so the message endpoint remembers them here
for the duration of the turn, keyed by conversation thread, and the tool reads
them back. One record per thread — a new turn replaces the previous turn's
record — and records expire after ``TURN_ATTACHMENT_TTL_SECONDS`` so an
abandoned turn cannot pin memory.

Process-local by design (same lifetime as ``src/api/media_jobs.py``): the graph
runs in the same process as the FastAPI app.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

TURN_ATTACHMENT_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class TurnAttachment:
    """One file exactly as the client attached it to the turn."""

    filename: str
    mime_type: str
    content: bytes

    @property
    def size_bytes(self) -> int:
        """Length of the raw payload."""
        return len(self.content)

    def describe(self) -> dict[str, Any]:
        """Return the public description the prompt and the tool result may carry."""
        return {
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }


@dataclass
class TurnAttachmentRecord:
    """What one turn left behind for the identity-update tool."""

    thread_id: str
    attachments: list[TurnAttachment]
    # The authenticated caller of the turn, so the tool can enforce and meter
    # the upload allotment the way ``/update_avatar_identity_with_media`` does.
    current_user: dict[str, Any]
    remembered_at: float = field(default_factory=time.monotonic)

    def is_expired(self, now: float | None = None) -> bool:
        """Whether the record is older than the TTL."""
        return ((now if now is not None else time.monotonic()) - self.remembered_at) > (
            TURN_ATTACHMENT_TTL_SECONDS
        )


_records: dict[str, TurnAttachmentRecord] = {}
_records_lock = threading.Lock()


def _evict_expired(now: float) -> None:
    for thread_id in [
        key for key, record in _records.items() if record.is_expired(now)
    ]:
        _records.pop(thread_id, None)


def remember_turn_attachments(
    thread_id: str,
    attachments: list[TurnAttachment],
    current_user: dict[str, Any],
) -> TurnAttachmentRecord:
    """Record the files of the turn that is about to run on ``thread_id``.

    Called even when the turn carried no files, because the tool also serves
    link-only requests ("learn from this video") and needs the caller identity
    to meter them.
    """
    record = TurnAttachmentRecord(
        thread_id=thread_id, attachments=list(attachments), current_user=current_user
    )
    with _records_lock:
        _evict_expired(time.monotonic())
        _records[thread_id] = record
    return record


def get_turn_attachments(thread_id: str | None) -> TurnAttachmentRecord | None:
    """Return the current turn's record for ``thread_id``; ``None`` when unknown or expired."""
    if not thread_id:
        return None
    with _records_lock:
        record = _records.get(thread_id)
        if record is None:
            return None
        if record.is_expired(time.monotonic()):
            _records.pop(thread_id, None)
            return None
        return record


def forget_turn_attachments(thread_id: str | None) -> None:
    """Drop the record for ``thread_id`` (the tool consumed it, or the turn ended)."""
    if not thread_id:
        return
    with _records_lock:
        _records.pop(thread_id, None)


def describe_turn_attachments(thread_id: str | None) -> list[dict[str, Any]]:
    """Public descriptions of the files attached to the current turn (for the prompt)."""
    record = get_turn_attachments(thread_id)
    if record is None:
        return []
    return [attachment.describe() for attachment in record.attachments]
