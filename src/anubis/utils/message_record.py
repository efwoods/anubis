"""Fields that ride on a transcript row so a reload still shows them.

``created_at`` is the instant the turn was written. ``text_model`` /
``text_model_provider`` name the inference model that produced a reply;
those two are attached only in development so an operator can see which
catalog entry answered.
"""

from datetime import UTC, datetime
from typing import Any


def utc_now_isoformat() -> str:
    """Return the current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def additional_kwargs_with_created_at(
    additional_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy ``additional_kwargs`` and stamp ``created_at`` when it is missing."""
    stamped = dict(additional_kwargs or {})
    if not stamped.get("created_at"):
        stamped["created_at"] = utc_now_isoformat()
    return stamped


def attach_visible_reply_metadata(message: Any, context: Any | None = None) -> None:
    """Write ``created_at`` and, in development, the text model onto a reply.

    Mutates ``response_metadata`` and ``additional_kwargs`` so the live
    ``done`` frame and a reopened checkpoint both carry the same stamp
    (and, in DEV, the same model name).
    """
    from src.anubis.utils.model import (
        development_mode_enabled,
        text_inference_record,
    )

    metadata = dict(getattr(message, "response_metadata", None) or {})
    kwargs = dict(getattr(message, "additional_kwargs", None) or {})
    created_at = (
        metadata.get("created_at") or kwargs.get("created_at") or utc_now_isoformat()
    )
    metadata["created_at"] = created_at
    kwargs["created_at"] = created_at
    if development_mode_enabled(context):
        metadata.update(text_inference_record(context))
    message.response_metadata = metadata
    message.additional_kwargs = kwargs
