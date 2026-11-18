"""Hosted diarizers other than OpenAI's, behind one provider switch.

``AUDIO_DIARIZATION_PROVIDER`` selects the diarizer ``transcribe_audio_diarize``
sends a recording to: ``OPENAI`` (the default, ``gpt-4o-transcribe-diarize``),
``ASSEMBLYAI``, ``DEEPGRAM`` or ``PYANNOTEAI``. OpenAI lists
``gpt-4o-transcribe-diarize`` and ``whisper-1`` for shutdown on 2027-02-26
(developers.openai.com/api/docs/deprecations, read 2026-11-17), and no
replacement OpenAI model documents ``diarized_json``, so the switch exists to
move the upload pipeline without a second rewrite.

Every provider function takes the path of one preprocessed mp3 file and
returns the dictionary shape ``transcribe_audio_diarize`` returns: ``segments``
(each with ``speaker``, ``text``, ``start`` and ``end`` in seconds), ``text``,
``duration``, ``model``, ``inference_type``, ``total_cost`` and ``latency_ms``.
The three hosted providers accept a whole recording in one request
(AssemblyAI 2.2 GB per upload, Deepgram 2 GB, pyannoteAI 1 GiB; vendor
documentation read 2026-11-17), so a recording is never cut into chunks and a
speaker label means the same person from the first second to the last.

None of the three hosted providers labels a known speaker from a reference
clip inside the transcription request (pyannoteAI's voiceprint identification
is a separate endpoint that returns no words). The avatar's turns are
therefore identified after the call by the transcript-reading attribution
pass in ``target_attribution.py``.

Request and response shapes follow each vendor's documentation as read on
2026-11-17. The response-normalizing functions are covered by unit tests; the
live requests have not been exercised, because no vendor key existed when the
module was written. ``scripts/diarization_bakeoff.py`` is the live check.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from time import time_ns
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

PROVIDER_OPENAI = "OPENAI"
PROVIDER_ASSEMBLYAI = "ASSEMBLYAI"
PROVIDER_DEEPGRAM = "DEEPGRAM"
PROVIDER_PYANNOTEAI = "PYANNOTEAI"
DIARIZATION_PROVIDERS = (
    PROVIDER_OPENAI,
    PROVIDER_ASSEMBLYAI,
    PROVIDER_DEEPGRAM,
    PROVIDER_PYANNOTEAI,
)

_UPLOAD_TIMEOUT_SECONDS = 1800.0
_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 3600.0


class DiarizationProviderError(RuntimeError):
    """A hosted diarizer refused the request or reported a failed job."""


def selected_diarization_provider(context: Any) -> str:
    """Return the configured provider name, defaulting to ``OPENAI``."""
    configured = str(getattr(context, "audio_diarization_provider", None) or "").strip().upper()
    if not configured:
        return PROVIDER_OPENAI
    if configured not in DIARIZATION_PROVIDERS:
        msg = (
            f"AUDIO_DIARIZATION_PROVIDER={configured!r} is not one of "
            f"{', '.join(DIARIZATION_PROVIDERS)}"
        )
        raise DiarizationProviderError(msg)
    return configured


def _required_key(context: Any, field_name: str) -> str:
    key = str(getattr(context, field_name, None) or "").strip()
    if not key:
        msg = f"{field_name.upper()} is not set"
        raise DiarizationProviderError(msg)
    return key


def _job_timeout_seconds(context: Any) -> float:
    return float(
        getattr(context, "audio_diarization_job_timeout_seconds", None)
        or _DEFAULT_JOB_TIMEOUT_SECONDS
    )


def _response_dictionary(
    *,
    provider: str,
    model: str,
    segments: List[Dict[str, Any]],
    duration_seconds: float,
    price_per_hour: float,
    started_at_nanoseconds: int,
    billing_currency: str = "USD",
) -> Dict[str, Any]:
    """Assemble the dictionary shape ``transcribe_audio_diarize`` returns.

    Hosted diarizers bill on the duration of the audio, so the cost is
    ``duration_seconds / 3600 × price_per_hour``. A ``price_per_hour`` of 0.0
    (the price setting is empty) records a cost of 0.0 rather than a guess.
    """
    return {
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "duration": duration_seconds,
        "task": "transcribe",
        "model": model,
        "provider": provider,
        "inference_type": "diarization",
        "total_cost": duration_seconds / 3600.0 * float(price_per_hour or 0.0),
        "billing_currency": billing_currency,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_ms": (time_ns() - started_at_nanoseconds) / 1e6,
    }


# ---------------------------------------------------------------------------
# AssemblyAI
# ---------------------------------------------------------------------------


def normalize_assemblyai_transcript(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn AssemblyAI ``utterances`` into segments with seconds.

    AssemblyAI reports ``start`` and ``end`` in milliseconds and names the
    speakers "A", "B", and so on.
    """
    segments: List[Dict[str, Any]] = []
    for utterance in transcript.get("utterances") or []:
        text = str(utterance.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "speaker": str(utterance.get("speaker") or "unknown"),
                "text": text,
                "start": float(utterance.get("start") or 0) / 1000.0,
                "end": float(utterance.get("end") or 0) / 1000.0,
            }
        )
    return segments


async def diarize_with_assemblyai(
    mp3_path: str, context: Any, *, expected_speaker_count: int | None = None
) -> Dict[str, Any]:
    """Diarize one mp3 file with AssemblyAI (upload, submit, poll)."""
    import httpx

    started_at_nanoseconds = time_ns()
    api_key = _required_key(context, "assemblyai_api_key")
    headers = {"authorization": api_key}
    base_url = "https://api.assemblyai.com"
    async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SECONDS) as client:
        with open(mp3_path, "rb") as audio_file:
            upload_response = await client.post(
                f"{base_url}/v2/upload",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=audio_file.read(),
            )
        upload_response.raise_for_status()
        request_body: Dict[str, Any] = {
            "audio_url": upload_response.json()["upload_url"],
            "speaker_labels": True,
        }
        if expected_speaker_count:
            request_body["speakers_expected"] = int(expected_speaker_count)
        speech_model = str(getattr(context, "assemblyai_speech_model", None) or "").strip()
        if speech_model:
            request_body["speech_model"] = speech_model
        submit_response = await client.post(
            f"{base_url}/v2/transcript", headers=headers, json=request_body
        )
        submit_response.raise_for_status()
        transcript_id = submit_response.json()["id"]

        deadline = asyncio.get_running_loop().time() + _job_timeout_seconds(context)
        while True:
            poll_response = await client.get(
                f"{base_url}/v2/transcript/{transcript_id}", headers=headers
            )
            poll_response.raise_for_status()
            transcript = poll_response.json()
            status = transcript.get("status")
            if status == "completed":
                break
            if status == "error":
                msg = f"AssemblyAI transcript {transcript_id} failed: {transcript.get('error')}"
                raise DiarizationProviderError(msg)
            if asyncio.get_running_loop().time() > deadline:
                msg = f"AssemblyAI transcript {transcript_id} did not finish in time"
                raise DiarizationProviderError(msg)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    segments = normalize_assemblyai_transcript(transcript)
    duration_seconds = float(
        transcript.get("audio_duration") or (segments[-1]["end"] if segments else 0.0)
    )
    return _response_dictionary(
        provider=PROVIDER_ASSEMBLYAI,
        model=f"assemblyai:{speech_model or 'default'}",
        segments=segments,
        duration_seconds=duration_seconds,
        price_per_hour=float(getattr(context, "assemblyai_price_per_hour", None) or 0.0),
        started_at_nanoseconds=started_at_nanoseconds,
    )


# ---------------------------------------------------------------------------
# Deepgram
# ---------------------------------------------------------------------------


def normalize_deepgram_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn Deepgram ``results.utterances`` into segments.

    Deepgram reports ``start`` and ``end`` in seconds and numbers the speakers
    0, 1, and so on; the label is rendered as ``speaker_<number>``.
    """
    segments: List[Dict[str, Any]] = []
    for utterance in (response.get("results") or {}).get("utterances") or []:
        text = str(utterance.get("transcript") or "").strip()
        if not text:
            continue
        speaker_number = utterance.get("speaker")
        segments.append(
            {
                "speaker": (
                    f"speaker_{speaker_number}" if speaker_number is not None else "unknown"
                ),
                "text": text,
                "start": float(utterance.get("start") or 0.0),
                "end": float(utterance.get("end") or 0.0),
            }
        )
    return segments


async def diarize_with_deepgram(
    mp3_path: str, context: Any, *, expected_speaker_count: int | None = None
) -> Dict[str, Any]:
    """Diarize one mp3 file with Deepgram (one synchronous request).

    Deepgram documents no expected-speaker-count parameter, so
    ``expected_speaker_count`` is accepted for a uniform signature and unused.
    """
    import httpx

    started_at_nanoseconds = time_ns()
    api_key = _required_key(context, "deepgram_api_key")
    model = str(getattr(context, "deepgram_diarization_model", None) or "nova-3").strip()
    async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SECONDS) as client:
        with open(mp3_path, "rb") as audio_file:
            http_response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params={
                    "model": model,
                    "diarize_model": "latest",
                    "utterances": "true",
                    "smart_format": "true",
                },
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/mpeg",
                },
                content=audio_file.read(),
            )
    http_response.raise_for_status()
    response = http_response.json()
    segments = normalize_deepgram_response(response)
    duration_seconds = float(
        (response.get("metadata") or {}).get("duration")
        or (segments[-1]["end"] if segments else 0.0)
    )
    return _response_dictionary(
        provider=PROVIDER_DEEPGRAM,
        model=f"deepgram:{model}",
        segments=segments,
        duration_seconds=duration_seconds,
        price_per_hour=float(getattr(context, "deepgram_price_per_hour", None) or 0.0),
        started_at_nanoseconds=started_at_nanoseconds,
    )


# ---------------------------------------------------------------------------
# pyannoteAI
# ---------------------------------------------------------------------------


def normalize_pyannoteai_output(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn pyannoteAI ``turnLevelTranscription`` into segments (seconds)."""
    segments: List[Dict[str, Any]] = []
    for turn in output.get("turnLevelTranscription") or []:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "speaker": str(turn.get("speaker") or "unknown"),
                "text": text,
                "start": float(turn.get("start") or 0.0),
                "end": float(turn.get("end") or 0.0),
            }
        )
    return segments


async def diarize_with_pyannoteai(
    mp3_path: str, context: Any, *, expected_speaker_count: int | None = None
) -> Dict[str, Any]:
    """Diarize and transcribe one mp3 file with pyannoteAI (upload, submit, poll).

    pyannoteAI prices are listed in euros, so ``total_cost`` is in euros unless
    ``PYANNOTEAI_PRICE_PER_HOUR`` was entered already converted;
    ``billing_currency`` on the response records ``PYANNOTEAI_BILLING_CURRENCY``.
    """
    import httpx

    started_at_nanoseconds = time_ns()
    api_key = _required_key(context, "pyannoteai_api_key")
    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = "https://api.pyannote.ai"
    model = str(getattr(context, "pyannoteai_diarization_model", None) or "precision-2").strip()
    media_address = f"media://diarization-{uuid.uuid4().hex}.mp3"
    async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SECONDS) as client:
        presign_response = await client.post(
            f"{base_url}/v1/media/input", headers=headers, json={"url": media_address}
        )
        presign_response.raise_for_status()
        with open(mp3_path, "rb") as audio_file:
            put_response = await client.put(
                presign_response.json()["url"], content=audio_file.read()
            )
        put_response.raise_for_status()

        request_body: Dict[str, Any] = {
            "url": media_address,
            "model": model,
            "transcription": True,
        }
        if expected_speaker_count:
            request_body["numSpeakers"] = int(expected_speaker_count)
        submit_response = await client.post(
            f"{base_url}/v1/diarize", headers=headers, json=request_body
        )
        submit_response.raise_for_status()
        job_id = submit_response.json()["jobId"]

        deadline = asyncio.get_running_loop().time() + _job_timeout_seconds(context)
        while True:
            poll_response = await client.get(f"{base_url}/v1/jobs/{job_id}", headers=headers)
            poll_response.raise_for_status()
            job = poll_response.json()
            status = str(job.get("status") or "").lower()
            if status == "succeeded":
                break
            if status in ("failed", "canceled", "cancelled"):
                msg = f"pyannoteAI job {job_id} ended with status {status}"
                raise DiarizationProviderError(msg)
            if asyncio.get_running_loop().time() > deadline:
                msg = f"pyannoteAI job {job_id} did not finish in time"
                raise DiarizationProviderError(msg)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    segments = normalize_pyannoteai_output(job.get("output") or {})
    duration_seconds = float(segments[-1]["end"] if segments else 0.0)
    return _response_dictionary(
        provider=PROVIDER_PYANNOTEAI,
        model=f"pyannoteai:{model}",
        segments=segments,
        duration_seconds=duration_seconds,
        price_per_hour=float(getattr(context, "pyannoteai_price_per_hour", None) or 0.0),
        started_at_nanoseconds=started_at_nanoseconds,
        billing_currency=str(
            getattr(context, "pyannoteai_billing_currency", None) or "EUR"
        ).upper(),
    )


_PROVIDER_FUNCTIONS = {
    PROVIDER_ASSEMBLYAI: diarize_with_assemblyai,
    PROVIDER_DEEPGRAM: diarize_with_deepgram,
    PROVIDER_PYANNOTEAI: diarize_with_pyannoteai,
}


async def diarize_mp3_with_hosted_provider(
    provider: str,
    mp3_path: str,
    context: Any,
    *,
    expected_speaker_count: int | None = None,
) -> Dict[str, Any]:
    """Send one preprocessed mp3 file to the named hosted diarizer."""
    provider_function = _PROVIDER_FUNCTIONS.get(provider)
    if provider_function is None:
        msg = f"{provider!r} is not a hosted diarization provider"
        raise DiarizationProviderError(msg)
    if not os.path.exists(mp3_path):
        msg = f"audio file {mp3_path} does not exist"
        raise DiarizationProviderError(msg)
    return await provider_function(
        mp3_path, context, expected_speaker_count=expected_speaker_count
    )
