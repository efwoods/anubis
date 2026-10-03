"""The per-reply cost breakdown: the reply plus every vision and triage call before it.

The cost shown under an avatar reply used to cover only the reply's own model
calls (``response_metadata["total_cost"]``). The image descriptions and the
ambient triage calls a screen share or webcam makes were billed as their own
``api_metrics`` rows and appeared on no reply at all, and most observations are
judged ``ignore``, so most of those calls never had a reply to appear on.

Every such call now leaves one pending item in the ``pending_turn_cost_items``
state channel. The next reply absorbs the pending items into
``response_metadata["turn_cost"]`` — a total and a per-call breakdown — and the
pending list is emptied.

``response_metadata["total_cost"]`` and ``response_metadata["token_usage"]``
deliberately stay reply-only: the message endpoint bills those two fields into
Stripe and the ``message`` ``api_metrics`` row, while every image description
and triage call is already billed as its own row. ``turn_cost`` is for display
and reconciliation only and is never billed.
"""

from __future__ import annotations

from typing import Any

INFERENCE_TYPE_IMAGE_DESCRIPTION = "image_description"
INFERENCE_TYPE_AMBIENT_TRIAGE = "ambient_triage"

PENDING_TURN_COST_ITEMS_KEY = "pending_turn_cost_items"
TURN_COST_METADATA_KEY = "turn_cost"


def pending_turn_cost_item(
    inference_type: str,
    *,
    model_name: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    observation_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Return one pending cost item describing one billed model call."""
    return {
        "inference_type": inference_type,
        "model_name": model_name,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "cached_prompt_tokens": int(cached_prompt_tokens or 0),
        "cache_write_tokens": int(cache_write_tokens or 0),
        "cost_usd": float(cost_usd or 0.0),
        "latency_ms": float(latency_ms or 0.0),
        "observation_id": observation_id,
        "source": source,
    }


def _summarize_pending_items(
    pending_items: list[dict[str, Any]], inference_type: str
) -> dict[str, Any]:
    """Sum the pending items of one ``inference_type`` into a breakdown entry."""
    matching_items = [
        pending_item
        for pending_item in pending_items
        if pending_item.get("inference_type") == inference_type
    ]
    return {
        "count": len(matching_items),
        "prompt_tokens": sum(
            int(pending_item.get("prompt_tokens") or 0) for pending_item in matching_items
        ),
        "completion_tokens": sum(
            int(pending_item.get("completion_tokens") or 0)
            for pending_item in matching_items
        ),
        "cached_prompt_tokens": sum(
            int(pending_item.get("cached_prompt_tokens") or 0)
            for pending_item in matching_items
        ),
        "cost_usd": sum(
            float(pending_item.get("cost_usd") or 0.0) for pending_item in matching_items
        ),
    }


def build_turn_cost_breakdown(
    response_metadata: dict[str, Any] | None,
    pending_items: list[dict[str, Any]] | None,
    *,
    reply_model_name: str | None = None,
) -> dict[str, Any]:
    """Build the ``turn_cost`` record from the reply's metadata and the pending items.

    :param response_metadata: The reply's ``response_metadata``; the reply cost
        is ``total_cost`` and the reply tokens are ``token_usage``.
    :param pending_items: The ``pending_turn_cost_items`` channel's items.
    :param reply_model_name: The inference model the reply was priced at.
    :returns: ``{total_cost_usd, total_tokens, reply, image_descriptions,
        ambient_triage, items}``.
    """
    response_metadata = response_metadata or {}
    pending_items = [
        dict(pending_item)
        for pending_item in (pending_items or [])
        if isinstance(pending_item, dict)
    ]
    reply_token_usage = response_metadata.get("token_usage") or {}
    reply_prompt_tokens = int(reply_token_usage.get("prompt_tokens") or 0)
    reply_completion_tokens = int(reply_token_usage.get("completion_tokens") or 0)
    reply_total_tokens = int(
        reply_token_usage.get("total_tokens")
        or reply_prompt_tokens + reply_completion_tokens
    )
    reply_cost_usd = float(response_metadata.get("total_cost") or 0.0)
    image_descriptions = _summarize_pending_items(
        pending_items, INFERENCE_TYPE_IMAGE_DESCRIPTION
    )
    ambient_triage = _summarize_pending_items(
        pending_items, INFERENCE_TYPE_AMBIENT_TRIAGE
    )
    pending_items_cost_usd = sum(
        float(pending_item.get("cost_usd") or 0.0) for pending_item in pending_items
    )
    pending_items_tokens = sum(
        int(pending_item.get("prompt_tokens") or 0)
        + int(pending_item.get("completion_tokens") or 0)
        for pending_item in pending_items
    )
    return {
        "total_cost_usd": reply_cost_usd + pending_items_cost_usd,
        "total_tokens": reply_total_tokens + pending_items_tokens,
        "reply": {
            "model_name": reply_model_name,
            "prompt_tokens": reply_prompt_tokens,
            "completion_tokens": reply_completion_tokens,
            "cached_prompt_tokens": int(
                reply_token_usage.get("cached_prompt_tokens") or 0
            ),
            "cost_usd": reply_cost_usd,
        },
        "image_descriptions": image_descriptions,
        "ambient_triage": ambient_triage,
        "items": pending_items,
    }


def attach_turn_cost_breakdown(
    avatar_response: Any,
    pending_items: list[dict[str, Any]] | None,
    *,
    reply_model_name: str | None = None,
) -> None:
    """Write ``response_metadata["turn_cost"]`` onto the avatar's reply in place."""
    response_metadata = dict(getattr(avatar_response, "response_metadata", None) or {})
    response_metadata[TURN_COST_METADATA_KEY] = build_turn_cost_breakdown(
        response_metadata, pending_items, reply_model_name=reply_model_name
    )
    avatar_response.response_metadata = response_metadata
