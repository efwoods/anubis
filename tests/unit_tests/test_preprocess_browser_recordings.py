"""Browser ``MediaRecorder`` output must survive audio preprocessing.

Chromium records live-voice utterances and dictation as WebM/Opus streams with
no duration header (``Duration: N/A`` from ``ffmpeg -i``). moviepy refuses to
open such files, which used to surface in the web app as
"The recording could not be transcribed: Error passing `ffmpeg -i` command
output". ``preprocess_audio`` now transcodes with ffmpeg directly first.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.utility import (  # noqa: E402
    _ffmpeg_executable,
    preprocess_audio,
)


def _ffmpeg_available() -> bool:
    executable = _ffmpeg_executable()
    return bool(shutil.which(executable) or Path(executable).exists())


pytestmark = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg is not available in this environment"
)


def _media_recorder_like_webm(seconds: float) -> bytes:
    """A WebM/Opus stream written to a pipe, which leaves out the duration header."""
    completed = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _has_duration_header(webm_bytes: bytes) -> bool:
    completed = subprocess.run(
        [_ffmpeg_executable(), "-hide_banner", "-i", "-"],
        input=webm_bytes,
        capture_output=True,
        check=False,
    )
    return "Duration: N/A" not in completed.stderr.decode("utf-8", "replace")


def _data_uri(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def test_webm_utterance_without_duration_header_is_transcoded_to_mp3() -> None:
    webm_bytes = _media_recorder_like_webm(seconds=2.0)
    assert not _has_duration_header(webm_bytes), "fixture must lack a duration header"

    result = asyncio.run(
        preprocess_audio(
            _data_uri("audio/webm", webm_bytes),
            truncate_only=False,
            reference_audio=False,
            filename="utterance.webm",
            max_duration_seconds=None,
        )
    )

    assert result["audio_base64"].startswith("data:audio/mp3;base64,")
    assert result["duration_seconds"] == pytest.approx(2.0, abs=0.25)
    assert isinstance(result["sample_rate"], int) and result["sample_rate"] > 0
    mp3_bytes = base64.b64decode(result["audio_base64"].split(",", 1)[1])
    assert len(mp3_bytes) > 1000


def test_webm_utterance_with_wrong_filename_still_transcodes() -> None:
    # The caller used to default the temp suffix to ``.mp3``; ffmpeg probes the
    # container from the bytes, so a misleading name must not matter.
    webm_bytes = _media_recorder_like_webm(seconds=1.0)

    result = asyncio.run(
        preprocess_audio(
            _data_uri("audio/webm", webm_bytes),
            truncate_only=False,
            reference_audio=False,
            filename=None,
            max_duration_seconds=None,
        )
    )

    assert result["duration_seconds"] == pytest.approx(1.0, abs=0.25)


def test_reference_audio_is_clipped_to_max_seconds() -> None:
    webm_bytes = _media_recorder_like_webm(seconds=4.0)

    result = asyncio.run(
        preprocess_audio(
            _data_uri("audio/webm", webm_bytes),
            truncate_only=True,
            reference_audio=True,
            filename="reference.webm",
            max_duration_seconds=1.5,
        )
    )

    assert result["duration_seconds"] == pytest.approx(1.5, abs=0.25)
