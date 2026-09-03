"""The xAI image-edit and image-to-video calls, and nothing else.

Two endpoints are used (see ``_EMOTION_MEDIA_GENERATION_COST_REPORT.md`` §5):

``POST /v1/images/edits``
    The reference image plus an editing instruction → one still. Requested as
    base64 so the bytes come back in the response; a URL result is downloaded
    immediately because xAI's URLs are signed and expire.

``POST /v1/videos/generations`` then ``GET /v1/videos/{request_id}``
    The still plus the idle-loop prompt → an asynchronous request that is
    polled until ``done``; the video is downloaded straight away for the same
    reason.

Every call goes through one ``httpx.AsyncClient`` created per operation, with
the bearer key from ``GlobalContext.xai_api_key``. Errors are raised as
:class:`XaiGenerationError` carrying the vendor's message, so the caller can
record the failure per asset and continue with the rest of the set.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai"


class XaiGenerationError(RuntimeError):
    """The vendor refused or failed a generation."""


class XaiNotConfiguredError(RuntimeError):
    """No xAI key is configured, so nothing can be generated."""


def _api_key(context: Any) -> str:
    key = str(getattr(context, "xai_api_key", None) or "").strip()
    if not key:
        raise XaiNotConfiguredError(
            "XAI_API_KEY is not configured; emotion media cannot be generated."
        )
    return key


def _headers(context: Any) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key(context)}",
        "Content-Type": "application/json",
    }


def _data_uri(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _decode_data_uri(data_uri: str) -> tuple[str, bytes]:
    """Split a ``data:<mime>;base64,<payload>`` string into its parts."""
    header, _, encoded = str(data_uri).partition(",")
    mime_type = header[5:].split(";", 1)[0] or "image/jpeg"
    return mime_type, base64.b64decode(encoded)


async def _download(client: Any, url: str) -> tuple[bytes, str]:
    response = await client.get(url, timeout=120.0)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


async def edit_image(
    context: Any, *, reference_image_data_uri: str, prompt: str
) -> dict[str, Any]:
    """Derive one emotion still from the reference image.

    Returns ``{"bytes", "mime_type", "request_id", "model"}``. Raises
    :class:`XaiGenerationError` when the vendor fails the edit.
    """
    import httpx

    model = str(
        getattr(context, "xai_image_edit_model", None) or "grok-imagine-image-2.0"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "image": {"url": reference_image_data_uri, "type": "image_url"},
        "response_format": "b64_json",
    }
    async with httpx.AsyncClient(base_url=XAI_BASE_URL, timeout=180.0) as client:
        try:
            response = await client.post(
                "/v1/images/edits", json=payload, headers=_headers(context)
            )
        except httpx.HTTPError as transport_error:
            raise XaiGenerationError(
                f"The image edit could not reach xAI: {transport_error}"
            ) from transport_error
        if response.status_code >= 400:
            raise XaiGenerationError(
                f"xAI refused the image edit ({response.status_code}): {response.text[:500]}"
            )
        body = response.json()
        entries = body.get("data") or []
        if not entries:
            raise XaiGenerationError("xAI returned no image for the edit.")
        entry = entries[0]
        if entry.get("b64_json"):
            image_bytes = base64.b64decode(entry["b64_json"])
            mime_type = entry.get("mime_type") or "image/jpeg"
        elif entry.get("url"):
            image_bytes, content_type = await _download(client, entry["url"])
            mime_type = content_type.split(";")[0] or "image/jpeg"
        else:
            raise XaiGenerationError("xAI returned an image entry with no payload.")
    return {
        "bytes": image_bytes,
        "mime_type": mime_type,
        "request_id": body.get("id") or body.get("request_id"),
        "model": model,
    }


async def generate_idle_loop(
    context: Any, *, still_image_data_uri: str, prompt: str
) -> dict[str, Any]:
    """Animate one still into an idle loop and return the finished video bytes.

    Submits the image-to-video request, polls until ``done`` (or the timeout),
    downloads the video, and returns
    ``{"bytes", "mime_type", "request_id", "model", "duration_seconds"}``.
    """
    import httpx

    model = str(getattr(context, "xai_video_model", None) or "grok-imagine-video-1.5")
    duration = int(getattr(context, "xai_idle_loop_duration_seconds", None) or 6)
    resolution = str(getattr(context, "xai_video_resolution", None) or "720p")
    aspect_ratio = str(getattr(context, "xai_video_aspect_ratio", None) or "9:16")
    poll_interval = float(
        getattr(context, "xai_video_poll_interval_seconds", None) or 5.0
    )
    poll_timeout = float(
        getattr(context, "xai_video_poll_timeout_seconds", None) or 600.0
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "image": {"url": still_image_data_uri, "type": "image_url"},
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    async with httpx.AsyncClient(base_url=XAI_BASE_URL, timeout=180.0) as client:
        try:
            response = await client.post(
                "/v1/videos/generations", json=payload, headers=_headers(context)
            )
        except httpx.HTTPError as transport_error:
            raise XaiGenerationError(
                f"The video generation could not reach xAI: {transport_error}"
            ) from transport_error
        if response.status_code >= 400:
            raise XaiGenerationError(
                f"xAI refused the video generation ({response.status_code}): "
                f"{response.text[:500]}"
            )
        request_id = response.json().get("request_id") or response.json().get("id")
        if not request_id:
            raise XaiGenerationError("xAI returned no request id for the video.")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + poll_timeout
        video_url: str | None = None
        while True:
            await asyncio.sleep(poll_interval)
            status_response = await client.get(
                f"/v1/videos/{request_id}",
                headers={"Authorization": _headers(context)["Authorization"]},
            )
            if status_response.status_code >= 400:
                raise XaiGenerationError(
                    f"xAI failed the status check for {request_id} "
                    f"({status_response.status_code}): {status_response.text[:300]}"
                )
            status_body = status_response.json()
            status = str(status_body.get("status") or "").lower()
            if status == "done":
                video_url = (status_body.get("video") or {}).get(
                    "url"
                ) or status_body.get("url")
                break
            if status in ("failed", "expired"):
                raise XaiGenerationError(
                    f"xAI reported the idle loop {request_id} as {status}."
                )
            if loop.time() > deadline:
                raise XaiGenerationError(
                    f"xAI did not finish the idle loop {request_id} within "
                    f"{poll_timeout:.0f} seconds."
                )
        if not video_url:
            raise XaiGenerationError("xAI finished the idle loop without a video URL.")
        video_bytes, content_type = await _download(client, video_url)
    return {
        "bytes": video_bytes,
        "mime_type": content_type.split(";")[0] or "video/mp4",
        "request_id": request_id,
        "model": model,
        "duration_seconds": float(duration),
    }


__all__ = [
    "XaiGenerationError",
    "XaiNotConfiguredError",
    "edit_image",
    "generate_idle_loop",
    "_data_uri",
    "_decode_data_uri",
]
