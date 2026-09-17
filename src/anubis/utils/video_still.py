"""A single JPEG still from a chat-attached video, for image description.

Chat turns currently turn a video into a ``[File: name - video/mp4]`` line.
One ffmpeg still feeds the existing image-description path so the avatar can
name what is in the attached clip without opening the live camera.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def still_jpeg_from_video_bytes(video_bytes: bytes, filename: str | None = None) -> bytes | None:
    """Return one JPEG frame from ``video_bytes``, or ``None`` if ffmpeg cannot."""
    if not video_bytes:
        return None
    from src.anubis.utils.utility import _ffmpeg_executable

    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    source_path = ""
    still_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
            source.write(video_bytes)
            source_path = source.name
        still_path = source_path + ".jpg"
        last_error = b""
        for seek in ("0.5", "0"):
            completed = subprocess.run(
                [
                    _ffmpeg_executable(),
                    "-v",
                    "error",
                    "-nostdin",
                    "-ss",
                    seek,
                    "-i",
                    source_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    "-y",
                    still_path,
                ],
                capture_output=True,
                timeout=20,
                check=False,
            )
            last_error = completed.stderr or b""
            if completed.returncode == 0 and os.path.isfile(still_path):
                with open(still_path, "rb") as jpeg:
                    still = jpeg.read()
                if still:
                    return still
        logger.info(
            "Could not take a still from %s: %s",
            filename or "attached video",
            last_error.decode("utf-8", errors="replace")[-300:],
        )
        return None
    except Exception:  # noqa: BLE001 - a still must never fail the chat turn
        logger.info("Could not take a still from %s", filename or "attached video", exc_info=True)
        return None
    finally:
        for path in (source_path, still_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
