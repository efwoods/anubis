"""Cut the target speaker's turns out of an already-diarized recording.

``isolate_dominant_speaker_audio_b64`` diarizes and cuts in one step, which is
right for a reference upload but would run the diarizer a second time on an
ordinary upload whose turns are already known. This module does only the
cutting: given the audio and the turns the media pipeline has already
attributed, it concatenates the target's windows (with short fades, like the
non-reference branch of the isolator) into one mp3 data URI and reports its
duration. moviepy + ffmpeg work is synchronous, so the whole thing runs in a
worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

_FADE_SECONDS = 0.025
_MINIMUM_WINDOW_SECONDS = 0.3


def _decode_to_temp_file(audio_data_uri: str) -> str:
    header, _, encoded = str(audio_data_uri).partition(",")
    mime_type = header[5:].split(";", 1)[0] or "audio/mpeg"
    suffix = ".wav" if "wav" in mime_type else ".m4a" if "mp4" in mime_type else ".mp3"
    payload = base64.b64decode(encoded)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        return handle.name


def target_windows(turns: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return the ``(start, end)`` windows of the target's turns, merged and ordered."""
    windows: list[tuple[float, float]] = []
    for turn in turns:
        if not turn.get("is_target"):
            continue
        try:
            start = float(turn.get("start"))
            end = float(turn.get("end"))
        except (TypeError, ValueError):
            continue
        if end - start < _MINIMUM_WINDOW_SECONDS:
            continue
        windows.append((start, end))
    windows.sort()
    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


async def cut_target_turns_to_mp3_data_uri(
    audio_data_uri: str, turns: list[dict[str, Any]]
) -> tuple[str | None, float]:
    """Return ``(mp3 data URI, seconds)`` of the target's speech, or ``(None, 0)``."""
    windows = target_windows(turns)
    if not windows:
        return None, 0.0

    def _produce() -> tuple[str | None, float]:
        from moviepy import AudioFileClip
        from moviepy.audio.AudioClip import concatenate_audioclips
        from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

        source_path = _decode_to_temp_file(audio_data_uri)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            output_path = handle.name
        clip = AudioFileClip(source_path)
        pieces = []
        try:
            total = float(clip.duration or 0.0)
            kept_seconds = 0.0
            for start, end in windows:
                end = min(end, total) if total else end
                if end - start < _MINIMUM_WINDOW_SECONDS:
                    continue
                piece = clip.subclipped(start, end)
                fade = min(_FADE_SECONDS, (end - start) / 4)
                if fade > 0:
                    piece = piece.with_effects([AudioFadeIn(fade), AudioFadeOut(fade)])
                pieces.append(piece)
                kept_seconds += end - start
            if not pieces:
                return None, 0.0
            glued = concatenate_audioclips(pieces)
            try:
                glued.write_audiofile(output_path, codec="mp3", logger=None)
            finally:
                glued.close()
            with open(output_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            return f"data:audio/mpeg;base64,{encoded}", kept_seconds
        finally:
            for piece in pieces:
                try:
                    piece.close()
                except Exception:  # noqa: BLE001
                    pass
            clip.close()
            for path in (source_path, output_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    try:
        return await asyncio.to_thread(_produce)
    except Exception as cut_error:  # noqa: BLE001 - never fail the upload
        logger.warning("Could not cut target turns for the voice corpus: %s", cut_error)
        return None, 0.0
