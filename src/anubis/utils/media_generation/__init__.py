"""Generated emotion media for an avatar: stills and idle loops from a reference image.

- :mod:`prompts` — the seven base emotions and the editing / idle-loop prompts.
- :mod:`xai_client` — the xAI image-edit and image-to-video calls.
- :mod:`emotion_media` — the build: six stills and seven loops per avatar,
  persisted to the media tables, with per-call cost recorded.
"""

from src.anubis.utils.media_generation.prompts import (
    BASE_EMOTIONS,
    GENERATED_EMOTIONS,
    NEUTRAL_EMOTION,
)

__all__ = ["BASE_EMOTIONS", "GENERATED_EMOTIONS", "NEUTRAL_EMOTION"]
