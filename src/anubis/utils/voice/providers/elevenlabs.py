"""ElevenLabs as a ``VoiceProvider``: a thin adapter over ``elevenlabs_client``.

Every method looks the ``elevenlabs_client`` function up at call time, so tests
that replace ``elevenlabs_client.synthesize_speech`` and friends keep working.
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.voice import elevenlabs_client

STANDARD_VOICE_GENDERS = ("female", "male")


def _normalize_gender(gender: Any) -> str | None:
    candidate = str(gender or "").strip().lower()
    return candidate if candidate in STANDARD_VOICE_GENDERS else None


class ElevenLabsVoiceProvider:
    """Stock voices, instant clones, and speech from ElevenLabs."""

    name = "elevenlabs"
    display_name = "ElevenLabs"
    supports_professional_clone = True

    def is_configured(self, context: Any) -> bool:
        """Return whether the provider's API key is present."""
        return bool(
            str(
                getattr(context, "elevenlabs_api_key", None)
                or getattr(context, "nn_elevenlabs_api_key", None)
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
        return await elevenlabs_client.create_instant_voice(
            context, name=name, clips=clips, description=description
        )

    async def delete_voice(self, context: Any, voice_id: str) -> None:
        """Delete a cloned voice at the vendor, logging rather than raising on failure."""
        await elevenlabs_client.delete_voice(context, voice_id)

    async def voice_is_blocked(self, context: Any, *, voice_id: str) -> bool:
        """Return whether the vendor has banned the voice."""
        return await elevenlabs_client.voice_is_blocked(context, voice_id=voice_id)

    async def synthesize_speech(
        self, context: Any, *, voice_id: str, text: str
    ) -> bytes:
        """Speak the text in the voice and return MP3 bytes."""
        return await elevenlabs_client.synthesize_speech(
            context,
            voice_id=voice_id,
            text=text,
            model_id=self.speech_model_name(context),
        )

    async def list_stock_voices(
        self, context: Any, *, gender: str
    ) -> list[dict[str, Any]]:
        """Return the vendor's stock voices of one gender, normalized."""
        premade_voices = await elevenlabs_client.list_premade_voices(
            context, gender=gender
        )
        stock_voices: list[dict[str, Any]] = []
        for premade_voice in premade_voices:
            voice_id = str(premade_voice.get("voice_id") or "").strip()
            if not voice_id:
                continue
            labels = premade_voice.get("labels") or {}
            stock_voices.append(
                {
                    "voice_id": voice_id,
                    "name": str(premade_voice.get("name") or "").strip() or None,
                    "gender": _normalize_gender(labels.get("gender")) or gender,
                    "accent": labels.get("accent"),
                    "age": labels.get("age"),
                    "description": labels.get("description")
                    or premade_voice.get("description"),
                    "preview_url": premade_voice.get("preview_url"),
                    # ElevenLabs serves premade samples from a public address.
                    "preview_requires_auth": False,
                }
            )
        return stock_voices

    async def stock_voice_preview(
        self, context: Any, *, voice_id: str
    ) -> tuple[bytes, str] | None:
        """Return a stock voice sample the browser cannot fetch itself, or None."""
        # The browser plays the public ``preview_url`` directly.
        return None

    def speech_model_name(self, context: Any) -> str:
        """Return the text-to-speech model the provider speaks with."""
        return str(
            getattr(context, "elevenlabs_text_to_speech_model", None)
            or "eleven_flash_v2_5"
        )

    def speech_cost_per_1000_characters_usd(self, context: Any) -> float:
        """Return the vendor cost per 1,000 characters of speech."""
        # Only an unset price falls back: a configured 0 (a plan whose
        # included credits cover the speech) is a real price and is kept.
        configured_price = getattr(
            context, "elevenlabs_text_to_speech_cost_per_1000_characters_usd", None
        )
        if configured_price is None or str(configured_price).strip() == "":
            return 0.05
        return float(configured_price)

    def instant_clone_max_seconds(self, context: Any) -> float | None:
        """Return the most corpus seconds an instant clone can use, or None for no cap."""
        # ElevenLabs accepts several files; the corpus target alone bounds the clone.
        return None

    def default_stock_voice_id(self, context: Any, *, gender: str) -> str | None:
        """Return the configured standard stock voice of one gender; ElevenLabs has none."""
        return None


PROVIDER = ElevenLabsVoiceProvider()
