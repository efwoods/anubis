"""The cheap first pass that runs inline on every chat turn.

OpenAI's moderation endpoint (``omni-moderation-latest``) is free and answers in
roughly a tenth of a second, so it can sit on the critical path of a reply where
the structured-output terms-of-service judge in
``src.anubis.utils.moderation.content_moderation`` cannot. It returns a score per
harm category; this module turns those scores into one of three outcomes:

``FAST_SCREEN_CLEAN``    nothing flagged - the turn proceeds and no deep judge runs.
``FAST_SCREEN_SUSPECT``  flagged below the hard threshold - the turn proceeds and the
                         deep judge reads the text afterwards, off the reply's path.
``FAST_SCREEN_BLOCK``    a category scored at or above ``fast_screen_threshold`` - the
                         turn is refused inline and the account is banned.

Like the deep judge this is fail-OPEN: a screening outage returns CLEAN and logs,
because refusing every person is a far worse outcome than missing one violation
that the background judge will catch on the next pass anyway.

The speech APIs set the precedent this follows: moderation always uses the OpenAI
client regardless of ``MODEL_PROVIDER``, reading ``openai_api_key`` and falling
back to ``llm_provider_api_key`` (see ``_openai_client_for_speech`` in
``src.anubis.utils.utility``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FAST_SCREEN_CLEAN = "clean"
FAST_SCREEN_SUSPECT = "suspect"
FAST_SCREEN_BLOCK = "block"

DEFAULT_FAST_SCREEN_MODEL = "omni-moderation-latest"
DEFAULT_FAST_SCREEN_THRESHOLD = 0.9
# The screen reads a bounded head of the text: the categories it detects are
# visible in the opening of a message, and an unbounded body would make the
# "cheap" pass expensive to serialize.
FAST_SCREEN_MAX_CHARACTERS = 4000


def clean_screen(outcome: str = FAST_SCREEN_CLEAN) -> dict[str, Any]:
    """The screen result shape: an outcome plus whatever the endpoint reported."""
    return {
        "outcome": outcome,
        "flagged": False,
        "categories": [],
        "scores": {},
        "highest_category": "",
        "highest_score": 0.0,
    }


def _openai_client_for_moderation(context: Any):
    """The OpenAI client the moderation endpoint is called with.

    Mirrors ``_openai_client_for_speech``: moderation is an OpenAI-only API, so it
    never routes through ``MODEL_PROVIDER``.
    """
    from openai import AsyncOpenAI

    api_key = getattr(context, "openai_api_key", None) or getattr(
        context, "llm_provider_api_key", None
    )
    if not api_key:
        message = (
            "Set `openai_api_key` / OPENAI_API_KEY or `llm_provider_api_key` for the "
            "content moderation screen."
        )
        raise ValueError(message)
    return AsyncOpenAI(api_key=api_key, max_retries=0)


async def invoke_fast_screen_model(text: str, context: Any) -> Any:
    """One moderation-endpoint call. Isolated so tests can replace the endpoint."""
    client = _openai_client_for_moderation(context)
    model_name = (
        getattr(context, "content_moderation_fast_screen_model", None)
        or DEFAULT_FAST_SCREEN_MODEL
    )
    return await client.moderations.create(model=model_name, input=text)


def _threshold_from_context(context: Any) -> float:
    try:
        threshold = float(
            getattr(context, "content_moderation_fast_screen_threshold", None)
            or DEFAULT_FAST_SCREEN_THRESHOLD
        )
    except (TypeError, ValueError):
        return DEFAULT_FAST_SCREEN_THRESHOLD
    # A threshold outside the endpoint's own 0-1 score range would make the block
    # outcome either unreachable or unconditional; clamp rather than surprise.
    return min(max(threshold, 0.0), 1.0)


def _result_to_screen(result: Any, threshold: float) -> dict[str, Any]:
    """Turn one moderation result into the screen shape, with the outcome decided."""
    categories = getattr(result, "categories", None)
    scores = getattr(result, "category_scores", None)
    category_map = (
        categories
        if isinstance(categories, dict)
        else getattr(categories, "__dict__", {})
    ) or {}
    score_map = (
        scores if isinstance(scores, dict) else getattr(scores, "__dict__", {})
    ) or {}

    flagged_categories = [
        str(name) for name, value in category_map.items() if bool(value)
    ]
    numeric_scores: dict[str, float] = {}
    for name, value in score_map.items():
        try:
            numeric_scores[str(name)] = float(value)
        except (TypeError, ValueError):
            continue

    highest_category = ""
    highest_score = 0.0
    if numeric_scores:
        highest_category, highest_score = max(
            numeric_scores.items(), key=lambda item: item[1]
        )

    flagged = bool(getattr(result, "flagged", False)) or bool(flagged_categories)
    if highest_score >= threshold:
        outcome = FAST_SCREEN_BLOCK
    elif flagged:
        outcome = FAST_SCREEN_SUSPECT
    else:
        outcome = FAST_SCREEN_CLEAN

    return {
        "outcome": outcome,
        "flagged": flagged,
        "categories": flagged_categories,
        "scores": numeric_scores,
        "highest_category": highest_category,
        "highest_score": highest_score,
    }


async def fast_screen_text(text: str, context: Any) -> dict[str, Any]:
    """Screen one text; returns the screen shape described in the module docstring."""
    text = (text or "").strip()
    if not text:
        return clean_screen()
    try:
        response = await invoke_fast_screen_model(
            text[:FAST_SCREEN_MAX_CHARACTERS], context
        )
    except Exception as screen_error:  # noqa: BLE001 - fail open, see module docstring
        logger.error(
            "Content moderation fast screen failed (treating as clean): %s",
            screen_error,
        )
        return {**clean_screen(), "screen_error": str(screen_error)}

    results = getattr(response, "results", None) or []
    if not results:
        return clean_screen()
    return _result_to_screen(results[0], _threshold_from_context(context))


def screen_to_verdict(screen: dict[str, Any]) -> dict[str, Any]:
    """Render a blocking screen result as the verdict shape the ban path consumes."""
    categories = ", ".join(screen.get("categories") or []) or screen.get(
        "highest_category", ""
    )
    return {
        "violation": True,
        "reasoning": (
            "The message was refused by automated content screening for "
            f"{categories or 'prohibited content'}."
        ),
        "violated_clauses": [],
        "screen": {
            "categories": screen.get("categories") or [],
            "highest_category": screen.get("highest_category", ""),
            "highest_score": screen.get("highest_score", 0.0),
        },
    }


__all__ = [
    "DEFAULT_FAST_SCREEN_MODEL",
    "DEFAULT_FAST_SCREEN_THRESHOLD",
    "FAST_SCREEN_BLOCK",
    "FAST_SCREEN_CLEAN",
    "FAST_SCREEN_MAX_CHARACTERS",
    "FAST_SCREEN_SUSPECT",
    "clean_screen",
    "fast_screen_text",
    "invoke_fast_screen_model",
    "screen_to_verdict",
]
