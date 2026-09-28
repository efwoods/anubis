"""Vendor-neutral errors every voice provider raises.

Callers catch these instead of one vendor's exceptions, so switching
``VOICE_PROVIDER`` changes no ``except`` clause. Each vendor's own error classes
subclass the matching class here (``ElevenLabsVoiceBlockedError`` is a
``VoiceBlockedError``), so code and tests written against the ElevenLabs names
keep working.

This module imports nothing from the voice package, so every vendor module can
import it without a cycle.
"""

from __future__ import annotations


class VoiceProviderNotConfiguredError(RuntimeError):
    """The active voice provider has no API key configured."""


class VoiceProviderError(RuntimeError):
    """The voice provider refused or failed a request."""


class VoiceBlockedError(VoiceProviderError):
    """The provider has banned this voice; the voice can never speak again."""


class VoiceProviderKeyRefusedError(VoiceProviderError):
    """The provider rejected the configured API key."""


class VoiceProviderCreditsExhaustedError(VoiceProviderError):
    """The provider account has no remaining credits or quota."""
