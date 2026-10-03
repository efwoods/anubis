"""Cartesia as a ``VoiceProvider``: stock voices, instant clones, and speech.

Plain HTTP against ``https://api.cartesia.ai`` (no SDK dependency), every
request carrying ``Authorization: Bearer <CARTESIA_API_KEY>`` and
``Cartesia-Version: <CARTESIA_API_VERSION>``.

- Instant clone: ``POST /voices/clone`` (multipart ``clip``, ``name``,
  ``language``, ``description``) → ``id``. Cartesia takes one clip of up to
  16 MB and uses about the first minute, so the corpus clips are joined, in the
  order given, into one mp3 capped at ``CARTESIA_INSTANT_VOICE_CLONE_MAXIMUM_SECONDS``.
- Delete: ``DELETE /voices/{id}``.
- Speech: ``POST /tts/bytes`` → mp3 bytes.
- Stock voices: ``GET /voices`` (paged by ``starting_after``), keeping public
  voices the account does not own. A Cartesia preview file needs the API key,
  so the Voice panel plays a stock sample through
  ``GET /avatar_voice/standard_voices/{voice_id}/preview`` (``stock_voice_preview``).
  Most public Cartesia voices have no preview file at all, so for those voices
  the sample is ``STOCK_VOICE_SAMPLE_TEXT`` spoken in the voice, synthesized
  once per voice and kept in this process. Cartesia publishes one voice
  several times — an emotion variant per tagline ("Angry Broadway Voice",
  "Sad Broadway Voice") under the same name and description — so the catalogue
  keeps one entry per name, gender and description (``_deduplicate_stock_voices``).
  ``CARTESIA_STANDARD_FEMALE_VOICE_ID`` / ``CARTESIA_STANDARD_MALE_VOICE_ID``
  name the standard voice of each gender (``default_stock_voice_id``).

Cartesia has no moderation ban on a voice, so ``voice_is_blocked`` is always
false. Professional clones are not offered through this provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from typing import Any

from src.anubis.utils.voice.provider_errors import (
    VoiceProviderCreditsExhaustedError,
    VoiceProviderError,
    VoiceProviderKeyRefusedError,
    VoiceProviderNotConfiguredError,
)

logger = logging.getLogger(__name__)

CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_CLONE_CLIP_MAXIMUM_BYTES = 16 * 1024 * 1024
_STOCK_VOICE_PAGE_SIZE = 100
_STOCK_VOICE_MAXIMUM_PAGES = 20
_CREDIT_EXHAUSTED_MARKERS = ("credit", "quota", "insufficient", "payment required")
STOCK_VOICE_SAMPLE_TEXT = (
    "Hello there. This is how I sound when I read a reply out loud for you."
)
_STOCK_VOICE_SAMPLE_CACHE_MAXIMUM_ENTRIES = 256
_synthesized_stock_voice_samples: dict[str, bytes] = {}

# Cartesia reports gender presentation as ``feminine`` / ``masculine``; the
# Voice panel's picker speaks ``female`` / ``male``.
_GENDER_BY_CARTESIA_VALUE = {
    "feminine": "female",
    "female": "female",
    "woman": "female",
    "masculine": "male",
    "male": "male",
    "man": "male",
}


def _api_key(context: Any) -> str:
    api_key = str(
        getattr(context, "cartesia_api_key", None)
        or getattr(context, "nn_cartesia_api_key", None)
        or ""
    ).strip()
    if not api_key:
        raise VoiceProviderNotConfiguredError(
            "CARTESIA_API_KEY is not configured; voices cannot be cloned or spoken."
        )
    return api_key


def _headers(context: Any) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key(context)}",
        "Cartesia-Version": str(
            getattr(context, "cartesia_api_version", None) or "2026-08-14"
        ),
    }


def _language(context: Any) -> str:
    return str(getattr(context, "cartesia_voice_language", None) or "en")


def _raise_for_cartesia_response(response: Any, action_description: str) -> None:
    """Raise the matching shared voice error for a Cartesia failure response."""
    if response.status_code < 400:
        return
    body_text = str(getattr(response, "text", "") or "")[:400]
    message = f"Cartesia refused to {action_description} ({response.status_code}): {body_text}"
    lowered_body = body_text.lower()
    if response.status_code == 401:
        raise VoiceProviderKeyRefusedError(message)
    if response.status_code == 402 or any(
        marker in lowered_body for marker in _CREDIT_EXHAUSTED_MARKERS
    ):
        raise VoiceProviderCreditsExhaustedError(message)
    raise VoiceProviderError(message)


@contextlib.asynccontextmanager
async def _http_client(timeout_seconds: float) -> AsyncIterator[Any]:
    """Yield a Cartesia HTTP client; a timeout or dropped connection becomes ``VoiceProviderError``.

    The routes answer ``VoiceProviderError`` with 502, so an ``httpx``
    transport error raised anywhere inside the ``async with`` block is
    converted rather than left to surface as a 500.
    """
    import httpx

    try:
        async with httpx.AsyncClient(
            base_url=CARTESIA_BASE_URL, timeout=timeout_seconds
        ) as http_client:
            yield http_client
    except httpx.HTTPError as network_error:
        raise VoiceProviderError(
            f"Cartesia could not be reached: {type(network_error).__name__}"
        ) from network_error


def join_clips_to_mp3(
    clips: list[tuple[str, bytes, str]], maximum_seconds: float
) -> tuple[bytes, float]:
    """Join clips in order into one mp3 of at most ``maximum_seconds``; return ``(bytes, seconds)``."""
    from moviepy import AudioFileClip
    from moviepy.audio.AudioClip import concatenate_audioclips

    temporary_paths: list[str] = []
    opened_clips: list[Any] = []
    pieces_to_join: list[Any] = []
    kept_seconds = 0.0
    try:
        for filename, payload, _mime_type in clips:
            remaining_seconds = maximum_seconds - kept_seconds
            if remaining_seconds <= 0:
                break
            suffix = os.path.splitext(filename or "")[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                temporary_paths.append(handle.name)
            source_clip = AudioFileClip(temporary_paths[-1])
            opened_clips.append(source_clip)
            clip_seconds = float(source_clip.duration or 0.0)
            if clip_seconds <= 0:
                continue
            if clip_seconds > remaining_seconds:
                pieces_to_join.append(source_clip.subclipped(0, remaining_seconds))
                clip_seconds = remaining_seconds
            else:
                pieces_to_join.append(source_clip)
            kept_seconds += clip_seconds
        if not pieces_to_join:
            return b"", 0.0
        joined_clip = concatenate_audioclips(pieces_to_join)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            output_path = handle.name
        temporary_paths.append(output_path)
        try:
            joined_clip.write_audiofile(
                output_path, codec="mp3", bitrate="128k", logger=None
            )
        finally:
            joined_clip.close()
        with open(output_path, "rb") as handle:
            return handle.read(), kept_seconds
    finally:
        for opened_clip in opened_clips:
            try:
                opened_clip.close()
            except Exception:  # noqa: BLE001
                pass
        for temporary_path in temporary_paths:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _normalize_stock_voice(cartesia_voice: dict[str, Any]) -> dict[str, Any] | None:
    voice_id = str(cartesia_voice.get("id") or "").strip()
    if not voice_id:
        return None
    gender = _GENDER_BY_CARTESIA_VALUE.get(
        str(cartesia_voice.get("gender") or "").strip().lower()
    )
    return {
        "voice_id": voice_id,
        "name": str(cartesia_voice.get("name") or "").strip() or None,
        "gender": gender,
        "accent": cartesia_voice.get("accent"),
        "age": None,
        "description": cartesia_voice.get("description")
        or cartesia_voice.get("tagline"),
        "preview_url": None,
        # A Cartesia preview file needs the API key, and most public voices have
        # no preview file, so the API serves (or synthesizes) every sample.
        "preview_requires_auth": True,
    }


def _normalized_description(cartesia_voice: dict[str, Any]) -> str:
    """Return the voice description lowercased with punctuation and spacing removed."""
    description = str(cartesia_voice.get("description") or "").lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", description).split())


def _plain_variant_rank(cartesia_voice: dict[str, Any]) -> tuple[int, int, str]:
    """Rank one copy of a voice: the plain variant first, then the single-accent copy, then the oldest.

    An emotion variant's tagline is the plain tagline with the emotion in front
    ("Angry Broadway Voice" beside "Broadway Voice"), so the shortest tagline is
    the plain variant. A copy listing a single accent is the voice in the
    catalogue language only, rather than the multilingual copy.
    """
    return (
        len(str(cartesia_voice.get("tagline") or "")),
        len(cartesia_voice.get("accents") or []),
        str(cartesia_voice.get("created_at") or ""),
    )


def _deduplicate_stock_voices(
    cartesia_voices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one Cartesia voice per name, gender and description, in catalogue order.

    Voices that only share a name (a British Benedict and a narrator Benedict)
    have different descriptions and are both kept.
    """
    best_voice_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cartesia_voice in cartesia_voices:
        voice_identity = (
            str(cartesia_voice.get("name") or "").strip().lower(),
            str(cartesia_voice.get("gender") or "").strip().lower(),
            _normalized_description(cartesia_voice),
        )
        kept_voice = best_voice_by_identity.get(voice_identity)
        if kept_voice is None or _plain_variant_rank(cartesia_voice) < (
            _plain_variant_rank(kept_voice)
        ):
            best_voice_by_identity[voice_identity] = cartesia_voice
    kept_voice_ids = {
        id(cartesia_voice) for cartesia_voice in best_voice_by_identity.values()
    }
    return [
        cartesia_voice
        for cartesia_voice in cartesia_voices
        if id(cartesia_voice) in kept_voice_ids
    ]


def _is_stock_voice(cartesia_voice: dict[str, Any]) -> bool:
    """Return whether the voice is public and not the account's own clone."""
    if cartesia_voice.get("is_owner") is True:
        return False
    access = str(cartesia_voice.get("access") or "public").strip().lower()
    return access != "private"


class CartesiaVoiceProvider:
    """Stock voices, instant clones, and speech from Cartesia."""

    name = "cartesia"
    display_name = "Cartesia"
    supports_professional_clone = False

    def is_configured(self, context: Any) -> bool:
        """Return whether the provider's API key is present."""
        return bool(
            str(
                getattr(context, "cartesia_api_key", None)
                or getattr(context, "nn_cartesia_api_key", None)
                or ""
            ).strip()
        )

    async def create_instant_voice(
        self,
        context: Any,
        *,
        name: str,
        clips: list[tuple[str, bytes, str]],
        description: str = "",
    ) -> str:
        """Build an instant clone from the clips and return the new voice id."""
        headers = _headers(context)
        maximum_seconds = self.instant_clone_max_seconds(context) or 60.0
        try:
            clip_bytes, clip_seconds = await asyncio.to_thread(
                join_clips_to_mp3, clips, maximum_seconds
            )
        except Exception as join_error:  # noqa: BLE001 - reported as a vendor failure
            raise VoiceProviderError(
                f"Could not prepare the clip for the Cartesia clone: {join_error}"
            ) from join_error
        if not clip_bytes:
            raise VoiceProviderError("No audio was available for the Cartesia clone.")
        if len(clip_bytes) > CARTESIA_CLONE_CLIP_MAXIMUM_BYTES:
            raise VoiceProviderError(
                f"The joined clip is {len(clip_bytes)} bytes; Cartesia accepts at most "
                f"{CARTESIA_CLONE_CLIP_MAXIMUM_BYTES}."
            )
        form_fields = {"name": name[:80], "language": _language(context)}
        if description:
            form_fields["description"] = description
        async with _http_client(120.0) as http_client:
            response = await http_client.post(
                "/voices/clone",
                headers=headers,
                data=form_fields,
                files={"clip": ("voice.mp3", clip_bytes, "audio/mpeg")},
            )
        _raise_for_cartesia_response(response, "create the instant clone")
        voice_id = str((response.json() or {}).get("id") or "").strip()
        if not voice_id:
            raise VoiceProviderError("Cartesia returned no voice id for the clone.")
        logger.info(
            "Cartesia instant clone %s built from a %.0fs clip", voice_id, clip_seconds
        )
        return voice_id

    async def delete_voice(self, context: Any, voice_id: str) -> None:
        """Delete a cloned voice at the vendor, logging rather than raising on failure."""
        try:
            async with _http_client(60.0) as http_client:
                response = await http_client.delete(
                    f"/voices/{voice_id}", headers=_headers(context)
                )
            _raise_for_cartesia_response(response, "delete the voice")
        except (VoiceProviderError, VoiceProviderNotConfiguredError) as delete_error:
            logger.info(
                "Could not delete Cartesia voice %s; leaving the voice in place: %s",
                voice_id,
                delete_error,
            )

    async def voice_is_blocked(self, context: Any, *, voice_id: str) -> bool:
        """Return whether the vendor has banned the voice."""
        # Cartesia applies no moderation ban to a voice.
        return False

    async def synthesize_speech(
        self, context: Any, *, voice_id: str, text: str
    ) -> bytes:
        """Speak the text in the voice and return MP3 bytes."""
        request_body = {
            "model_id": self.speech_model_name(context),
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "mp3",
                "sample_rate": 44100,
                "bit_rate": 128000,
            },
            "language": _language(context),
        }
        async with _http_client(120.0) as http_client:
            response = await http_client.post(
                "/tts/bytes", headers=_headers(context), json=request_body
            )
        _raise_for_cartesia_response(response, "synthesize speech")
        return bytes(response.content)

    async def _list_all_stock_voices(self, context: Any) -> list[dict[str, Any]]:
        headers = _headers(context)
        cartesia_voices: list[dict[str, Any]] = []
        starting_after: str | None = None
        async with _http_client(60.0) as http_client:
            for _page_index in range(_STOCK_VOICE_MAXIMUM_PAGES):
                query_parameters: dict[str, Any] = {
                    "limit": _STOCK_VOICE_PAGE_SIZE,
                    "language": _language(context),
                    "expand[]": "preview_file_url",
                }
                if starting_after:
                    query_parameters["starting_after"] = starting_after
                response = await http_client.get(
                    "/voices", headers=headers, params=query_parameters
                )
                _raise_for_cartesia_response(response, "list voices")
                page_body = response.json() or {}
                page_voices = (
                    page_body.get("data") if isinstance(page_body, dict) else page_body
                ) or []
                cartesia_voices.extend(
                    page_voice
                    for page_voice in page_voices
                    if isinstance(page_voice, dict)
                )
                has_more = isinstance(page_body, dict) and bool(
                    page_body.get("has_more")
                )
                if not has_more or not page_voices:
                    break
                starting_after = str(page_voices[-1].get("id") or "") or None
                if starting_after is None:
                    break
        return cartesia_voices

    async def list_stock_voices(
        self, context: Any, *, gender: str
    ) -> list[dict[str, Any]]:
        """Return the vendor's stock voices of one gender, normalized."""
        stock_voices: list[dict[str, Any]] = []
        public_voices = [
            cartesia_voice
            for cartesia_voice in await self._list_all_stock_voices(context)
            if _is_stock_voice(cartesia_voice)
        ]
        for cartesia_voice in _deduplicate_stock_voices(public_voices):
            stock_voice = _normalize_stock_voice(cartesia_voice)
            if stock_voice is not None and stock_voice["gender"] == gender:
                stock_voices.append(stock_voice)
        return stock_voices

    async def stock_voice_preview(
        self, context: Any, *, voice_id: str
    ) -> tuple[bytes, str] | None:
        """Return a stock voice sample the browser cannot fetch itself.

        The voice's Cartesia preview file when the voice has one; otherwise
        ``STOCK_VOICE_SAMPLE_TEXT`` synthesized in the voice, cached per voice
        so each voice is synthesized once per process.
        """
        headers = _headers(context)
        async with _http_client(60.0) as http_client:
            response = await http_client.get(
                f"/voices/{voice_id}",
                headers=headers,
                params={"expand[]": "preview_file_url"},
            )
            _raise_for_cartesia_response(response, "read the voice")
            preview_file_url = str(
                (response.json() or {}).get("preview_file_url") or ""
            ).strip()
            if not preview_file_url:
                return (
                    await self._synthesized_stock_voice_sample(
                        context, voice_id=voice_id
                    ),
                    "audio/mpeg",
                )
            preview_response = await http_client.get(
                preview_file_url, headers=headers, follow_redirects=True
            )
            _raise_for_cartesia_response(preview_response, "download the voice sample")
        mime_type = (
            preview_response.headers.get("content-type", "") or "audio/mpeg"
        ).split(";", 1)[0]
        return bytes(preview_response.content), mime_type

    async def _synthesized_stock_voice_sample(
        self, context: Any, *, voice_id: str
    ) -> bytes:
        """Speak ``STOCK_VOICE_SAMPLE_TEXT`` in the voice, once per voice per process."""
        cached_sample = _synthesized_stock_voice_samples.get(voice_id)
        if cached_sample is not None:
            return cached_sample
        sample = await self.synthesize_speech(
            context, voice_id=voice_id, text=STOCK_VOICE_SAMPLE_TEXT
        )
        if (
            len(_synthesized_stock_voice_samples)
            >= _STOCK_VOICE_SAMPLE_CACHE_MAXIMUM_ENTRIES
        ):
            _synthesized_stock_voice_samples.pop(
                next(iter(_synthesized_stock_voice_samples))
            )
        _synthesized_stock_voice_samples[voice_id] = sample
        return sample

    def default_stock_voice_id(self, context: Any, *, gender: str) -> str | None:
        """Return the configured standard Cartesia voice of one gender, or ``None``."""
        setting_name = {
            "female": "cartesia_standard_female_voice_id",
            "male": "cartesia_standard_male_voice_id",
        }.get(gender)
        if setting_name is None:
            return None
        return str(getattr(context, setting_name, None) or "").strip() or None

    def speech_model_name(self, context: Any) -> str:
        """Return the text-to-speech model the provider speaks with."""
        return str(
            getattr(context, "cartesia_text_to_speech_model", None) or "sonic-3.6"
        )

    def speech_cost_per_1000_characters_usd(self, context: Any) -> float:
        """Return the vendor cost per 1,000 characters of speech."""
        # Only an unset price falls back: a configured 0 (a plan whose
        # included credits cover the speech) is a real price and is kept.
        configured_price = getattr(
            context, "cartesia_text_to_speech_cost_per_1000_characters_usd", None
        )
        if configured_price is None or str(configured_price).strip() == "":
            return 0.04
        return float(configured_price)

    def instant_clone_max_seconds(self, context: Any) -> float | None:
        """Return the most corpus seconds an instant clone can use, or None for no cap."""
        return float(
            getattr(context, "cartesia_instant_voice_clone_maximum_seconds", None)
            or 60.0
        )


PROVIDER = CartesiaVoiceProvider()
