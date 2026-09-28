"""Per-provider voice slots on the ``avatar_voice`` row: what makes a provider switch revertible.

An instant clone id and a standard voice id mean something only to the vendor
that minted them, so each provider keeps its own **slot**: the instant clone
id, the seconds the clone was built from, the ``instant_*`` state keys (errors,
the ban mark, the rebuild markers) and the chosen ``standard_voice``.

- The **ElevenLabs slot is the row itself**: the ``instant_voice_id`` /
  ``instant_voice_seconds`` columns and the ``instant_*`` / ``standard_voice``
  keys of ``detail``, exactly as every row was written before a second provider
  existed. Existing rows therefore read unchanged and need no migration.
- **Every other provider's slot** lives in
  ``detail.provider_voices.<provider>`` as ``{"instant_voice_id",
  "instant_voice_seconds", "detail": {...}}``.

``voice_slot`` returns a slot shaped like a voice row (``instant_voice_id``,
``instant_voice_seconds``, ``detail``), so the clone code and helpers such as
``voice_record_blocked`` and ``standard_voice_of`` work on any provider's slot
without knowing which provider owns the slot. For ElevenLabs the slot **is**
the row; for another provider the slot is a copy, written back with
``store_voice_slot`` before the row is saved.

Switching ``VOICE_PROVIDER`` never clears a slot, so switching back finds the
earlier provider's voices exactly as they were.
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.voice.providers import ELEVENLABS_PROVIDER_NAME

PROVIDER_VOICES_DETAIL_KEY = "provider_voices"


def voice_slot(record: dict[str, Any], provider_name: str) -> dict[str, Any]:
    """Return the provider's slot of the voice row, shaped like a voice row."""
    if provider_name == ELEVENLABS_PROVIDER_NAME:
        record.setdefault("detail", {})
        return record
    provider_voices = (record.get("detail") or {}).get(PROVIDER_VOICES_DETAIL_KEY) or {}
    stored_slot = provider_voices.get(provider_name) or {}
    return {
        "assistant_id": record.get("assistant_id"),
        "user_id": record.get("user_id"),
        "instant_voice_id": stored_slot.get("instant_voice_id"),
        "instant_voice_seconds": float(stored_slot.get("instant_voice_seconds") or 0.0),
        "detail": dict(stored_slot.get("detail") or {}),
    }


def store_voice_slot(
    record: dict[str, Any], provider_name: str, slot: dict[str, Any]
) -> dict[str, Any]:
    """Write the provider's slot back onto the voice row and return the row."""
    if slot is record:
        return record
    detail = dict(record.get("detail") or {})
    provider_voices = dict(detail.get(PROVIDER_VOICES_DETAIL_KEY) or {})
    provider_voices[provider_name] = {
        "instant_voice_id": slot.get("instant_voice_id"),
        "instant_voice_seconds": float(slot.get("instant_voice_seconds") or 0.0),
        "detail": dict(slot.get("detail") or {}),
    }
    detail[PROVIDER_VOICES_DETAIL_KEY] = provider_voices
    record["detail"] = detail
    return record
