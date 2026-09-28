"""A stock vendor voice an avatar speaks with instead of, or until, a usable clone.

Cloning needs about a minute of the avatar speaking, and a clone ElevenLabs
has banned never speaks again. In both situations the owner may pick one of
the vendor's own premade voices — any voice that matches the avatar's gender —
so the avatar is heard at all.

Which voice speaks is the owner's choice, kept as ``detail.voice_choice``:
``"standard"`` makes the chosen stock voice speak even when a usable clone
exists, and ``"custom"`` makes the clone speak whenever the clone is usable,
with the stock voice as the fallback. Picking a standard voice selects
``"standard"``; the owner switches back to the custom voice with
``set_voice_choice``. A row written before the choice existed has no
``voice_choice`` and behaves as ``"custom"``, which is how every avatar behaved
then.

The stock voice comes from the active voice provider (``VOICE_PROVIDER``) and
is kept in that provider's slot of the avatar's ``avatar_voice`` row
(``voice_slots.py``) as ``{"voice_id", "name", "gender"}``: for ElevenLabs
under ``detail.standard_voice``, as every row was written before a second
provider existed. A pick made on another provider stays stored, and keeps
speaking until the active provider has a pick of the avatar's own. The choice
between custom and standard is not per provider. The catalogue is read from the
active provider and cached in this process for an hour, per provider: the list
changes rarely and is asked for every time the Voice panel opens.
"""

from __future__ import annotations

import time
from typing import Any

from src.anubis.utils.voice.providers import (
    active_voice_provider_name,
    get_voice_provider,
    speaking_provider_order,
)
from src.anubis.utils.voice.voice_slots import store_voice_slot, voice_slot

STANDARD_VOICE_DETAIL_KEY = "standard_voice"
VOICE_CHOICE_DETAIL_KEY = "voice_choice"
VOICE_CHOICE_CUSTOM = "custom"
VOICE_CHOICE_STANDARD = "standard"
VOICE_CHOICES = (VOICE_CHOICE_CUSTOM, VOICE_CHOICE_STANDARD)
STANDARD_VOICE_GENDERS = ("female", "male")
CATALOGUE_CACHE_SECONDS = 3600.0

_catalogue_cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}


def normalize_gender(gender: str | None) -> str | None:
    """``"female"`` or ``"male"``; ``None`` for anything else."""
    candidate = str(gender or "").strip().lower()
    return candidate if candidate in STANDARD_VOICE_GENDERS else None


def standard_voice_of(
    record: dict[str, Any] | None, provider_name: str | None = None
) -> dict[str, Any] | None:
    """Return the standard voice chosen for the avatar, or ``None`` when none is set.

    ``record`` may be a voice row or one provider's slot of the row. With
    ``provider_name`` given, that provider's slot of the row is read.
    """
    if record is not None and provider_name is not None:
        record = voice_slot(dict(record), provider_name)
    detail = (record or {}).get("detail") or {}
    chosen = detail.get(STANDARD_VOICE_DETAIL_KEY)
    if not isinstance(chosen, dict) or not str(chosen.get("voice_id") or "").strip():
        return None
    return {
        "voice_id": str(chosen.get("voice_id")),
        "name": str(chosen.get("name") or "").strip() or None,
        "gender": normalize_gender(chosen.get("gender")),
    }


def any_standard_voice_of(
    record: dict[str, Any] | None, context: Any = None
) -> dict[str, Any] | None:
    """Return the first stored stock voice across ``speaking_provider_order``, or ``None``."""
    for provider_name in speaking_provider_order(context):
        standard_voice = standard_voice_of(record, provider_name)
        if standard_voice is not None:
            return standard_voice
    return None


def voice_choice_of(record: dict[str, Any] | None, context: Any = None) -> str:
    """Return ``"standard"`` when the owner chose the stock voice, else ``"custom"``.

    The stock voice is only a choice while one is stored (on any provider that
    may speak), so a ``"standard"`` choice with no stock voice reads as
    ``"custom"``.
    """
    detail = (record or {}).get("detail") or {}
    if (
        detail.get(VOICE_CHOICE_DETAIL_KEY) == VOICE_CHOICE_STANDARD
        and any_standard_voice_of(record, context) is not None
    ):
        return VOICE_CHOICE_STANDARD
    return VOICE_CHOICE_CUSTOM


def clear_catalogue_cache() -> None:
    """Forget the cached premade catalogue (tests, and a key change)."""
    _catalogue_cache.clear()


async def list_standard_voices(context: Any, *, gender: str) -> list[dict[str, Any]]:
    """Return the active provider's stock voices of one gender, cached for an hour.

    Each entry is ``{voice_id, name, gender, accent, age, description,
    preview_url, preview_requires_auth}``. ``preview_url`` is the vendor's
    public sample of the voice, which the Voice panel plays so the owner can
    choose by ear; when ``preview_requires_auth`` is true the vendor's sample
    needs the API key and the panel fetches the sample from
    ``GET /avatar_voice/standard_voices/{voice_id}/preview`` instead.
    """
    normalized = normalize_gender(gender)
    if normalized is None:
        raise ValueError(
            "gender must be one of " + ", ".join(STANDARD_VOICE_GENDERS) + "."
        )
    provider_name = active_voice_provider_name(context)
    cache_key = (provider_name, normalized)
    cached = _catalogue_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < CATALOGUE_CACHE_SECONDS:
        return list(cached[1])
    stock_voices = await get_voice_provider(context).list_stock_voices(
        context, gender=normalized
    )
    catalogue = sorted(
        (
            {
                "voice_id": str(stock_voice.get("voice_id") or ""),
                "name": stock_voice.get("name"),
                "gender": normalize_gender(stock_voice.get("gender")) or normalized,
                "accent": stock_voice.get("accent"),
                "age": stock_voice.get("age"),
                "description": stock_voice.get("description"),
                "preview_url": stock_voice.get("preview_url"),
                "preview_requires_auth": bool(stock_voice.get("preview_requires_auth")),
            }
            for stock_voice in stock_voices
            if str(stock_voice.get("voice_id") or "").strip()
        ),
        key=lambda entry: (entry["name"] or "").lower(),
    )
    _catalogue_cache[cache_key] = (now, catalogue)
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
    context: Any = None,
) -> dict[str, Any]:
    """Store (or, with ``None``, clear) the avatar's standard voice; return the row.

    The pick is stored in the active provider's slot (the ElevenLabs slot
    without a context). Picking a stock voice is the owner choosing to hear the
    stock voice, so the choice becomes ``"standard"``. Clearing the stock voice
    returns the choice to ``"custom"``.
    """
    from src.anubis.utils.voice.corpus import _voice_record

    provider_name = active_voice_provider_name(context)
    record = await _voice_record(repository, user_id, assistant_id)
    slot = voice_slot(record, provider_name)
    slot_detail = dict(slot.get("detail") or {})
    if voice is None:
        slot_detail.pop(STANDARD_VOICE_DETAIL_KEY, None)
    else:
        slot_detail[STANDARD_VOICE_DETAIL_KEY] = {
            "voice_id": str(voice["voice_id"]),
            "name": voice.get("name"),
            "gender": normalize_gender(voice.get("gender")),
        }
    slot["detail"] = slot_detail
    store_voice_slot(record, provider_name, slot)
    record["detail"] = {
        **(record.get("detail") or {}),
        VOICE_CHOICE_DETAIL_KEY: VOICE_CHOICE_CUSTOM
        if voice is None
        else VOICE_CHOICE_STANDARD,
    }
    await repository.upsert_voice(record)
    return record


async def set_voice_choice(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    choice: str,
    context: Any = None,
) -> dict[str, Any]:
    """Store which voice speaks, ``"custom"`` or ``"standard"``; return the row.

    Raises ``ValueError`` for an unknown choice, and for ``"standard"`` when no
    stock voice has been picked yet. The stock voice stays stored when the
    owner switches to the custom voice, so switching back needs no second pick.
    """
    from src.anubis.utils.voice.corpus import _voice_record

    normalized_choice = str(choice or "").strip().lower()
    if normalized_choice not in VOICE_CHOICES:
        raise ValueError("choice must be one of " + ", ".join(VOICE_CHOICES) + ".")
    record = await _voice_record(repository, user_id, assistant_id)
    if (
        normalized_choice == VOICE_CHOICE_STANDARD
        and any_standard_voice_of(record, context) is None
    ):
        raise ValueError("Pick a standard voice before choosing the standard voice.")
    record["detail"] = {
        **(record.get("detail") or {}),
        VOICE_CHOICE_DETAIL_KEY: normalized_choice,
    }
    await repository.upsert_voice(record)
    return record
