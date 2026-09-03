"""The avatar's voice: cloning, speech synthesis, and the clip corpus behind them.

- :mod:`elevenlabs_client` — the vendor calls (instant clone, professional
  clone, verification, training, synthesis), wrapped so the rest of the
  codebase never touches the SDK directly.
- :mod:`corpus` — collecting target-only speech clips, the thresholds that
  create an instant clone and prepare a professional one, and which voice is
  active for an avatar.
"""

from src.anubis.utils.voice.corpus import (
    VoiceStatus,
    add_voice_clip,
    resolve_active_voice_id,
    voice_status_for,
)

__all__ = [
    "VoiceStatus",
    "add_voice_clip",
    "resolve_active_voice_id",
    "voice_status_for",
]
