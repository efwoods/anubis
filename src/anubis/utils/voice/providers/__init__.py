"""The switchable voice provider: which vendor supplies stock voices, instant clones and speech.

``VOICE_PROVIDER`` (``ELEVENLABS`` or ``CARTESIA``) names the active provider.
The active provider supplies the standard voice catalogue, builds new instant
clones, and speaks with the voices the active provider minted. Every stored
voice id records the provider that minted the voice (see ``voice_slots.py``),
so a voice minted by the other provider keeps speaking through that provider,
and switching ``VOICE_PROVIDER`` back restores the earlier voices untouched.

Professional clones and lip-sync video stay on ElevenLabs
(``elevenlabs_client``) whatever ``VOICE_PROVIDER`` says; only
``ElevenLabsVoiceProvider.supports_professional_clone`` is true.

Adding a provider is one module exposing a ``VoiceProvider`` instance named
``PROVIDER`` plus one line in ``_PROVIDER_MODULES``. Provider modules are
imported on first use, per the repository's cold-start rule.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ELEVENLABS_PROVIDER_NAME = "elevenlabs"
CARTESIA_PROVIDER_NAME = "cartesia"
DEFAULT_VOICE_PROVIDER_NAME = ELEVENLABS_PROVIDER_NAME

_PROVIDER_MODULES: dict[str, str] = {
    ELEVENLABS_PROVIDER_NAME: "src.anubis.utils.voice.providers.elevenlabs",
    CARTESIA_PROVIDER_NAME: "src.anubis.utils.voice.providers.cartesia",
}
VOICE_PROVIDER_NAMES: tuple[str, ...] = tuple(_PROVIDER_MODULES)

_warned_unknown_provider_values: set[str] = set()


class VoiceProvider(Protocol):
    """What every voice vendor module implements."""

    name: str
    display_name: str
    supports_professional_clone: bool

    def is_configured(self, context: Any) -> bool:
        """Whether the provider's API key is present."""
        ...

    async def create_instant_voice(
        self,
        context: Any,
        *,
        name: str,
        clips: list[tuple[str, bytes, str]],
        description: str = "",
    ) -> str:
        """Build an instant clone from ``(filename, bytes, mime_type)`` clips; return the voice id."""
        ...

    async def delete_voice(self, context: Any, voice_id: str) -> None:
        """Delete a cloned voice at the vendor, logging rather than raising on failure."""
        ...

    async def voice_is_blocked(self, context: Any, *, voice_id: str) -> bool:
        """Whether the vendor has banned the voice."""
        ...

    async def synthesize_speech(
        self, context: Any, *, voice_id: str, text: str
    ) -> bytes:
        """Speak ``text`` in the voice and return MP3 bytes."""
        ...

    async def list_stock_voices(
        self, context: Any, *, gender: str
    ) -> list[dict[str, Any]]:
        """Return the vendor's stock voices of one gender, normalized.

        Each entry is ``{voice_id, name, gender, accent, age, description,
        preview_url, preview_requires_auth}``.
        """
        ...

    async def stock_voice_preview(
        self, context: Any, *, voice_id: str
    ) -> tuple[bytes, str] | None:
        """Return ``(audio_bytes, mime_type)`` for a stock voice sample the browser cannot fetch itself."""
        ...

    def speech_model_name(self, context: Any) -> str:
        """Return the text-to-speech model the provider speaks with."""
        ...

    def speech_cost_per_1000_characters_usd(self, context: Any) -> float:
        """Vendor cost per 1,000 characters of speech, for ``api_metrics``."""
        ...

    def instant_clone_max_seconds(self, context: Any) -> float | None:
        """Most seconds of the corpus an instant clone can use, or ``None`` for no provider cap."""
        ...

    def default_stock_voice_id(self, context: Any, *, gender: str) -> str | None:
        """Return the configured standard stock voice of one gender, or ``None`` when none is configured."""
        ...


def normalize_voice_provider_name(value: Any) -> str | None:
    """``"elevenlabs"`` / ``"cartesia"`` for a known provider name, any case; else ``None``."""
    candidate = str(value or "").strip().lower().replace("_", "").replace("-", "")
    return candidate if candidate in _PROVIDER_MODULES else None


def active_voice_provider_name(context: Any = None) -> str:
    """Return the provider named by ``VOICE_PROVIDER``; ElevenLabs when unset or unknown."""
    configured_value = getattr(context, "voice_provider", None) if context else None
    provider_name = normalize_voice_provider_name(configured_value)
    if provider_name is not None:
        return provider_name
    if (
        configured_value
        and str(configured_value) not in _warned_unknown_provider_values
    ):
        _warned_unknown_provider_values.add(str(configured_value))
        logger.warning(
            "VOICE_PROVIDER=%s is not one of %s; using %s",
            configured_value,
            ", ".join(VOICE_PROVIDER_NAMES),
            DEFAULT_VOICE_PROVIDER_NAME,
        )
    return DEFAULT_VOICE_PROVIDER_NAME


def get_voice_provider(context: Any = None, name: str | None = None) -> VoiceProvider:
    """Return the provider named ``name``, or the active provider when ``name`` is ``None``."""
    provider_name = (
        normalize_voice_provider_name(name)
        if name is not None
        else active_voice_provider_name(context)
    )
    if provider_name is None:
        raise ValueError(
            f"Unknown voice provider {name!r}; expected one of "
            + ", ".join(VOICE_PROVIDER_NAMES)
        )
    provider_module = importlib.import_module(_PROVIDER_MODULES[provider_name])
    return provider_module.PROVIDER  # type: ignore[no-any-return]


def voice_provider_configured(context: Any, name: str | None = None) -> bool:
    """Whether the named provider (the active provider by default) has a key."""
    return get_voice_provider(context, name).is_configured(context)


def other_voice_provider_names(context: Any = None) -> list[str]:
    """Return every provider except the active provider, in registry order."""
    active_name = active_voice_provider_name(context)
    return [name for name in VOICE_PROVIDER_NAMES if name != active_name]


def speaking_provider_order(context: Any = None) -> list[str]:
    """Return the providers whose stored voices may speak: the active provider only.

    After ``VOICE_PROVIDER`` is switched, speech goes through the active
    provider alone. A voice another provider minted stays stored in that
    provider's slot and speaks again when ``VOICE_PROVIDER`` is switched back,
    but the voice never speaks through the switched-off provider, whose key may
    be refused. Without a context only the default provider is listed, which is
    how every row behaved before a second provider existed.
    """
    if context is None:
        return [DEFAULT_VOICE_PROVIDER_NAME]
    return [active_voice_provider_name(context)]
