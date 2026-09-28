"""The avatar's voice: cloning, speech synthesis, and the clip corpus behind them.

- :mod:`providers` — the switchable voice provider (``VOICE_PROVIDER``:
  ElevenLabs or Cartesia) for stock voices, instant clones, and speech;
  :mod:`voice_slots` keeps each provider's voices apart on the voice row so a
  switch can be reverted.
- :mod:`elevenlabs_client` — the ElevenLabs calls (instant clone, professional
  clone, verification, training, synthesis, lip-sync video), wrapped so the
  rest of the codebase never touches the SDK directly. Professional clones and
  lip-sync video always use ElevenLabs.
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
