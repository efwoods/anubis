"""A stock vendor voice an avatar speaks with while it has no usable clone.

Cloning needs about two minutes of the avatar speaking, and a clone ElevenLabs
has banned never speaks again. In both situations the owner may pick one of
the vendor's own premade voices — any voice that matches the avatar's gender —
so the avatar is heard at all. The pick is a fallback: the moment a clone is
usable, the clone speaks and the standard voice is ignored.

The choice is kept on the avatar's ``avatar_voice`` row, under
``detail.standard_voice``, as ``{"voice_id", "name", "gender"}``. The premade
catalogue is read from the vendor and cached in this process for an hour: the
list changes rarely and is asked for every time the Voice panel opens.
"""

from __future__ import annotations

import time
from typing import Any

from src.anubis.utils.voice import elevenlabs_client

STANDARD_VOICE_DETAIL_KEY = "standard_voice"
STANDARD_VOICE_GENDERS = ("female", "male")
CATALOGUE_CACHE_SECONDS = 3600.0

_catalogue_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def normalize_gender(gender: str | None) -> str | None:
    """``"female"`` or ``"male"``; ``None`` for anything else."""
    candidate = str(gender or "").strip().lower()
    return candidate if candidate in STANDARD_VOICE_GENDERS else None


def standard_voice_of(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the standard voice chosen for the avatar, or ``None`` when none is set."""
    detail = (record or {}).get("detail") or {}
    chosen = detail.get(STANDARD_VOICE_DETAIL_KEY)
    if not isinstance(chosen, dict) or not str(chosen.get("voice_id") or "").strip():
        return None
    return {
        "voice_id": str(chosen.get("voice_id")),
        "name": str(chosen.get("name") or "").strip() or None,
        "gender": normalize_gender(chosen.get("gender")),
    }


def clear_catalogue_cache() -> None:
    """Forget the cached premade catalogue (tests, and a key change)."""
    _catalogue_cache.clear()


async def list_standard_voices(context: Any, *, gender: str) -> list[dict[str, Any]]:
    """Return the vendor's premade voices of one gender, cached for an hour.

    Each entry is ``{voice_id, name, gender, accent, age, description,
    preview_url}``. ``preview_url`` is the vendor's public sample of the voice,
    which the Voice panel plays so the owner can choose by ear.
    """
    normalized = normalize_gender(gender)
    if normalized is None:
        raise ValueError(
            "gender must be one of " + ", ".join(STANDARD_VOICE_GENDERS) + "."
        )
    cached = _catalogue_cache.get(normalized)
    now = time.monotonic()
    if cached is not None and now - cached[0] < CATALOGUE_CACHE_SECONDS:
        return list(cached[1])
    voices = await elevenlabs_client.list_premade_voices(context, gender=normalized)
    catalogue = sorted(
        (
            {
                "voice_id": str(voice.get("voice_id") or ""),
                "name": str(voice.get("name") or "").strip() or None,
                "gender": normalize_gender((voice.get("labels") or {}).get("gender"))
                or normalized,
                "accent": (voice.get("labels") or {}).get("accent"),
                "age": (voice.get("labels") or {}).get("age"),
                "description": (voice.get("labels") or {}).get("description")
                or voice.get("description"),
                "preview_url": voice.get("preview_url"),
            }
            for voice in voices
            if str(voice.get("voice_id") or "").strip()
        ),
        key=lambda entry: (entry["name"] or "").lower(),
    )
    _catalogue_cache[normalized] = (now, catalogue)
    return list(catalogue)


async def find_standard_voice(context: Any, *, voice_id: str) -> dict[str, Any] | None:
    """Return the catalogue entry for ``voice_id``, whichever gender it belongs to."""
    wanted = str(voice_id or "").strip()
    if not wanted:
        return None
    for gender in STANDARD_VOICE_GENDERS:
        for entry in await list_standard_voices(context, gender=gender):
            if entry["voice_id"] == wanted:
                return entry
    return None


async def set_standard_voice(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    voice: dict[str, Any] | None,
) -> dict[str, Any]:
    """Store (or, with ``None``, clear) the avatar's standard voice; return the row."""
    from src.anubis.utils.voice.corpus import _voice_record

    record = await _voice_record(repository, user_id, assistant_id)
    detail = dict(record.get("detail") or {})
    if voice is None:
        detail.pop(STANDARD_VOICE_DETAIL_KEY, None)
    else:
        detail[STANDARD_VOICE_DETAIL_KEY] = {
            "voice_id": str(voice["voice_id"]),
            "name": voice.get("name"),
            "gender": normalize_gender(voice.get("gender")),
        }
    record["detail"] = detail
    await repository.upsert_voice(record)
    return record
