"""Describe a capture of the web page and keep a thumbnail of the capture.

The browser renders the Neural Nexus document to a canvas and posts the
result. This module downsizes the capture to a thumbnail (Pillow), asks the
image-description model what the person is doing with the
``DESCRIBE_USAGE_ANALYTICS_SCREENSHOT_PROMPT`` instructions and the recent
actions as context, stores the outcome on the screenshot row, and mirrors the
description into the LangGraph store under ``(user_id, "usage_analytics")``
so the analytics tools can recall captures by similarity later.

Describing runs after the route has answered (``describe_and_store`` is
scheduled as a task): a vision call takes seconds and the browser must not
wait on the call.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.usage_analytics.prompts import (
    DESCRIBE_USAGE_ANALYTICS_SCREENSHOT_PROMPT,
    RECENT_ACTIONS_HEADER,
)

logger = logging.getLogger(__name__)

USAGE_ANALYTICS_NAMESPACE_SUFFIX = "usage_analytics"
DEFAULT_THUMBNAIL_WIDTH = 640
THUMBNAIL_JPEG_QUALITY = 70

STATUS_PENDING = "pending"
STATUS_DESCRIBED = "described"
STATUS_FAILED = "failed"


def usage_analytics_namespace(user_id: str) -> tuple[str, str]:
    """Return the store namespace holding one user's capture descriptions."""
    return (str(user_id), USAGE_ANALYTICS_NAMESPACE_SUFFIX)


def make_thumbnail(
    image_bytes: bytes, *, max_width: int = DEFAULT_THUMBNAIL_WIDTH
) -> tuple[bytes, str, int, int]:
    """Downscale ``image_bytes`` to a JPEG no wider than ``max_width``.

    Returns ``(jpeg_bytes, mime, width, height)`` of the thumbnail. A capture
    already narrower than ``max_width`` is re-encoded as JPEG without scaling.
    """
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        width, height = image.size
        if max_width and width > max_width:
            ratio = max_width / float(width)
            image = image.resize((max_width, max(1, int(height * ratio))))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
        return buffer.getvalue(), "image/jpeg", image.size[0], image.size[1]


def image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Return the capture's width and height, or ``None`` when Pillow cannot read the bytes."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as source:
            return source.size
    except Exception:  # noqa: BLE001 - dimensions are a nicety
        return None


def image_data_url(image_bytes: bytes, mime: str) -> str:
    """Encode the capture as the data URL the describing model accepts."""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def describer_system_prompt(recent_actions: str | None) -> str:
    """Return the describing instructions with the recent actions appended."""
    text = (recent_actions or "").strip()
    if not text:
        return DESCRIBE_USAGE_ANALYTICS_SCREENSHOT_PROMPT
    return f"{DESCRIBE_USAGE_ANALYTICS_SCREENSHOT_PROMPT}\n\n{RECENT_ACTIONS_HEADER}\n{text}"


def build_store_document(
    *,
    user_id: str,
    screenshot_id: str,
    description: str,
    route: str | None,
    assistant_id: str | None,
    session_id: str | None,
    occurred_at: datetime | None,
) -> dict[str, Any]:
    """Build the store value for one description, shaped like the identity documents."""
    from langchain_core.documents import Document

    stamp = (occurred_at or datetime.now(UTC)).isoformat()
    page_content = f"[{stamp}] On {route or 'an unknown screen'}: {description}"
    document = Document(
        page_content=page_content,
        metadata={
            "user_id": str(user_id),
            "assistant_id": assistant_id,
            "session_id": session_id,
            "route": route,
            "screenshot_id": str(screenshot_id),
            "kind": "usage_analytics_screenshot",
        },
    )
    return {
        "document": document.to_json(),
        "screenshot_id": str(screenshot_id),
        "route": route,
        "assistant_id": assistant_id,
        "session_id": session_id,
        "occurred_at": stamp,
    }


async def describe_and_store(
    *,
    repository: Any,
    store: Any | None,
    screenshot_id: str,
    user_id: str,
    image_bytes: bytes,
    image_mime: str,
    recent_actions: str | None,
    route: str | None,
    assistant_id: str | None,
    session_id: str | None,
    occurred_at: datetime | None,
    describer: Any | None = None,
) -> dict[str, Any] | None:
    """Describe one capture, record the outcome, and mirror the text into the store.

    ``describer`` defaults to ``ImageDescriptionClass`` with the analytics
    prompt; tests pass a fake with the same ``describe(url, filename)`` shape.
    Every failure is recorded as ``failed`` on the row and swallowed: a vision
    outage must never surface as an error to the browser.
    """
    try:
        if describer is None:
            from src.anubis.utils.classes.ImageDescriptionClass import (
                ImageDescriptionClass,
            )

            describer = ImageDescriptionClass(
                system_prompt=describer_system_prompt(recent_actions)
            )
        meta = await describer.describe(
            image_data_url(image_bytes, image_mime), f"usage_capture_{screenshot_id}"
        )
        description = (meta.get("description") or "").strip()
        completed = await repository.complete_screenshot(
            screenshot_id,
            description=description,
            model_name=meta.get("model_name"),
            total_cost=float(meta.get("total_cost") or 0.0),
            status=STATUS_DESCRIBED,
        )
    except Exception as describe_error:  # noqa: BLE001 - recorded, never raised
        logger.warning(
            "Usage analytics capture %s could not be described: %s",
            screenshot_id,
            describe_error,
        )
        try:
            await repository.complete_screenshot(
                screenshot_id,
                description=None,
                model_name=None,
                total_cost=0.0,
                status=STATUS_FAILED,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failure status for %s was not stored", screenshot_id, exc_info=True
            )
        return None

    if store is not None and description:
        try:
            await store.aput(
                usage_analytics_namespace(user_id),
                key=str(screenshot_id),
                value=build_store_document(
                    user_id=user_id,
                    screenshot_id=screenshot_id,
                    description=description,
                    route=route,
                    assistant_id=assistant_id,
                    session_id=session_id,
                    occurred_at=occurred_at,
                ),
            )
        except Exception:  # noqa: BLE001 - the table row is the record of truth
            logger.debug(
                "Usage analytics store write failed for %s",
                screenshot_id,
                exc_info=True,
            )
    return completed


async def recall_usage_descriptions(
    store: Any | None, user_id: str, query: str, *, limit: int = 10
) -> list[dict[str, Any]]:
    """Recall stored capture descriptions by similarity to ``query``."""
    if store is None:
        return []
    try:
        items = await store.asearch(
            usage_analytics_namespace(user_id), query=query, limit=limit
        )
    except Exception:  # noqa: BLE001
        return []
    results: list[dict[str, Any]] = []
    for item in items:
        value = getattr(item, "value", None) or {}
        document = value.get("document") or {}
        kwargs = document.get("kwargs") if isinstance(document, dict) else {}
        results.append(
            {
                "screenshot_id": value.get("screenshot_id"),
                "route": value.get("route"),
                "occurred_at": value.get("occurred_at"),
                "text": (kwargs or {}).get("page_content"),
            }
        )
    return results
