"""ElevenLabs calls for voice cloning and speech synthesis.

The SDK is synchronous; every call here runs in a worker thread so a slow
vendor response never stalls the event loop that is streaming tokens to other
conversations. The SDK import is deferred into ``_client`` per the repository's
cold-start rule.

What is used (see ``_EMOTION_MEDIA_GENERATION_COST_REPORT.md`` §5.5–5.9):

- Instant voice clone: ``voices.ivc.create(name, files)`` → ``voice_id``.
- Professional voice clone: ``voices.pvc.create`` → ``voices.pvc.samples.create``
  → CAPTCHA (``verification.captcha.get`` / ``verify``) → ``voices.pvc.train``
  → poll ``voices.get(voice_id).fine_tuning.state``.
- Speech: ``text_to_speech.convert`` (bytes) for the speak button and voice mode.
- Lip-sync: ``assets.create`` + ``POST /v1/flows/video`` (raw HTTP, since the
  typed request models track individual vendor models).

Clips are passed as ``(filename, bytes, mime_type)`` tuples, which is the
``core.File`` shape the SDK accepts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"


class ElevenLabsNotConfiguredError(RuntimeError):
    """No ElevenLabs key is configured."""


class ElevenLabsError(RuntimeError):
    """The vendor refused or failed a request."""


def _api_key(context: Any) -> str:
    key = str(
        getattr(context, "elevenlabs_api_key", None)
        or getattr(context, "nn_elevenlabs_api_key", None)
        or ""
    ).strip()
    if not key:
        raise ElevenLabsNotConfiguredError(
            "ELEVENLABS_API_KEY is not configured; voices cannot be cloned or spoken."
        )
    return key


def _client(context: Any) -> Any:
    from elevenlabs.client import ElevenLabs

    return ElevenLabs(api_key=_api_key(context))


def _as_file(clip: tuple[str, bytes, str]) -> tuple[str, bytes, str]:
    filename, payload, mime_type = clip
    return (filename or "clip.mp3", payload, mime_type or "audio/mpeg")


async def _run(operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(operation, *args, **kwargs)
    except ElevenLabsNotConfiguredError:
        raise
    except Exception as vendor_error:  # noqa: BLE001 - normalized for callers
        raise ElevenLabsError(str(vendor_error)) from vendor_error


# --- cloning -------------------------------------------------------------------


async def create_instant_voice(
    context: Any,
    *,
    name: str,
    clips: list[tuple[str, bytes, str]],
    description: str = "",
) -> str:
    """Create an instant voice clone from one or more clean clips; return its id."""

    def _create() -> str:
        client = _client(context)
        response = client.voices.ivc.create(
            name=name,
            files=[_as_file(clip) for clip in clips],
            description=description or None,
        )
        return str(response.voice_id)

    return await _run(_create)


async def delete_voice(context: Any, voice_id: str) -> None:
    """Delete a cloned voice (when a better clone replaces it)."""

    def _delete() -> None:
        _client(context).voices.delete(voice_id)

    try:
        await _run(_delete)
    except ElevenLabsError:
        logger.info("Could not delete voice %s; leaving it in place", voice_id)


async def create_professional_voice(
    context: Any, *, name: str, language: str = "en", description: str = ""
) -> str:
    """Create an empty professional voice; samples and training follow."""

    def _create() -> str:
        client = _client(context)
        response = client.voices.pvc.create(
            name=name, language=language, description=description or None
        )
        return str(response.voice_id)

    return await _run(_create)


async def add_professional_samples(
    context: Any, *, voice_id: str, clips: list[tuple[str, bytes, str]]
) -> list[str]:
    """Attach target-only clips to a professional voice; return sample ids."""

    def _add() -> list[str]:
        client = _client(context)
        samples = client.voices.pvc.samples.create(
            voice_id, files=[_as_file(clip) for clip in clips]
        )
        return [str(getattr(sample, "sample_id", "") or "") for sample in samples or []]

    return await _run(_add)


async def get_verification_captcha(context: Any, *, voice_id: str) -> dict[str, Any]:
    """Return the CAPTCHA the owner must read aloud to verify the voice.

    The typed SDK method returns nothing for this endpoint, so the raw response
    is read: the body carries the text to read (and, in some versions, an
    image); whatever fields are present are returned as a dictionary.
    """

    def _get() -> dict[str, Any]:
        client = _client(context)
        raw = client.voices.pvc.verification.captcha.with_raw_response.get(voice_id)
        response = getattr(raw, "response", None)
        if response is None:
            return {}
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            return {"text": response.text}
        return body if isinstance(body, dict) else {"text": str(body)}

    return await _run(_get)


async def submit_verification_recording(
    context: Any, *, voice_id: str, recording: tuple[str, bytes, str]
) -> dict[str, Any]:
    """Submit the owner's spoken CAPTCHA recording."""

    def _verify() -> dict[str, Any]:
        client = _client(context)
        response = client.voices.pvc.verification.captcha.verify(
            voice_id, recording=_as_file(recording)
        )
        return _model_to_dict(response)

    return await _run(_verify)


async def request_manual_verification(
    context: Any,
    *,
    voice_id: str,
    files: list[tuple[str, bytes, str]],
    extra_text: str = "",
) -> dict[str, Any]:
    """Fallback: ask ElevenLabs to verify the voice manually."""

    def _request() -> dict[str, Any]:
        client = _client(context)
        response = client.voices.pvc.verification.request(
            voice_id,
            files=[_as_file(clip) for clip in files],
            extra_text=extra_text or None,
        )
        return _model_to_dict(response)

    return await _run(_request)


async def train_professional_voice(
    context: Any, *, voice_id: str, model_id: str | None = None
) -> dict[str, Any]:
    """Start professional-clone training (three to six hours)."""

    def _train() -> dict[str, Any]:
        client = _client(context)
        response = client.voices.pvc.train(voice_id, model_id=model_id or None)
        return _model_to_dict(response)

    return await _run(_train)


async def get_voice_fine_tuning_state(
    context: Any, *, voice_id: str, model_id: str | None = None
) -> str:
    """Return the fine-tuning state of a voice (``fine_tuned``, ``fine_tuning``, ``failed``).

    ElevenLabs reports one state per model; the configured training model's
    state is preferred, otherwise the first state present.
    """

    def _get() -> str:
        client = _client(context)
        voice = client.voices.get(voice_id)
        fine_tuning = getattr(voice, "fine_tuning", None)
        states = getattr(fine_tuning, "state", None) or {}
        if model_id and model_id in states:
            return str(states[model_id])
        for state in states.values():
            return str(state)
        return str(getattr(fine_tuning, "finetuning_state", None) or "not_started")

    return await _run(_get)


# --- speech -------------------------------------------------------------------------


async def synthesize_speech(
    context: Any,
    *,
    voice_id: str,
    text: str,
    model_id: str | None = None,
    output_format: str = "mp3_44100_128",
) -> bytes:
    """Speak ``text`` in the cloned voice and return the audio bytes."""

    def _convert() -> bytes:
        client = _client(context)
        chunks = client.text_to_speech.convert(
            voice_id,
            text=text,
            model_id=model_id or None,
            output_format=output_format,
        )
        return b"".join(chunk for chunk in chunks if chunk)

    return await _run(_convert)


# --- lip-sync (Phase 5) --------------------------------------------------------------


async def upload_asset(
    context: Any, *, payload: bytes, name: str, mime_type: str
) -> str:
    """Upload an image or audio as a reusable asset; return its id."""

    def _upload() -> str:
        client = _client(context)
        response = client.assets.create(asset=(name, payload, mime_type), name=name)
        return str(response.asset_id)

    return await _run(_upload)


async def create_lip_sync_video(
    context: Any,
    *,
    model_id: str,
    image_asset_id: str,
    audio_asset_id: str,
    resolution: str = "720p",
) -> str:
    """Start a lip-sync generation (image + audio → video); return the generation id."""
    import httpx

    payload = {
        "model_id": model_id,
        "image": {"type": "asset", "asset_id": image_asset_id},
        "audio": {"type": "asset", "asset_id": audio_asset_id},
        "resolution": resolution,
    }
    async with httpx.AsyncClient(base_url=ELEVENLABS_BASE_URL, timeout=120.0) as client:
        response = await client.post(
            "/v1/flows/video",
            json=payload,
            headers={
                "xi-api-key": _api_key(context),
                "Content-Type": "application/json",
            },
        )
    if response.status_code >= 400:
        raise ElevenLabsError(
            f"ElevenLabs refused the lip-sync generation ({response.status_code}): "
            f"{response.text[:400]}"
        )
    body = response.json()
    generation_id = body.get("id") or body.get("generation_id")
    if not generation_id:
        raise ElevenLabsError(
            "ElevenLabs returned no generation id for the lip-sync clip."
        )
    return str(generation_id)


async def get_lip_sync_video(context: Any, *, generation_id: str) -> dict[str, Any]:
    """Return ``{"status", "content_url", "content_mime_type"}`` for a generation."""
    import httpx

    async with httpx.AsyncClient(base_url=ELEVENLABS_BASE_URL, timeout=60.0) as client:
        response = await client.get(
            f"/v1/flows/video/{generation_id}",
            headers={"xi-api-key": _api_key(context)},
        )
    if response.status_code >= 400:
        raise ElevenLabsError(
            f"ElevenLabs failed the lip-sync status check ({response.status_code}): "
            f"{response.text[:300]}"
        )
    body = response.json()
    return {
        "status": str(body.get("status") or "").lower(),
        "content_url": body.get("content_url"),
        "content_mime_type": body.get("content_mime_type") or "video/mp4",
    }


async def download(url: str) -> tuple[bytes, str]:
    """Fetch a signed vendor URL immediately; such URLs expire within the hour."""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "")


def _model_to_dict(model: Any) -> dict[str, Any]:
    if model is None:
        return {}
    for attribute in ("model_dump", "dict"):
        method = getattr(model, attribute, None)
        if callable(method):
            try:
                return dict(method())
            except Exception:  # noqa: BLE001
                continue
    return {"result": str(model)}
