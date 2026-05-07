"""Build / refresh the per-avatar stylistic profile in the LangGraph store.

Threshold rules:

* Build only when ``len(quote_documents) >= context.min_quotes_for_profile``.
* Refresh only when there are
  ``context.profile_refresh_threshold`` more quote Documents than were used
  the last time the profile was built (recorded in the profile metadata as
  ``built_with_document_count``).

The profile is stored at namespace ``(user_id, assistant_id,
"stylistic_profile")`` under key ``"profile"``. Evaluators MUST load this
profile once per evaluation pass and never re-process the corpus.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langgraph.store.base import BaseStore

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.dataset.stylistic_profile import compute_profile_from_quotes

logger = logging.getLogger(__name__)


STYLISTIC_PROFILE_NAMESPACE_TAG = "stylistic_profile"
STYLISTIC_PROFILE_KEY = "profile"


async def _enumerate_quote_texts(
    user_id: str, assistant_id: str, store: BaseStore
) -> List[str]:
    """Return the raw page_content of every quote-namespace Document.

    ``BaseStore.asearch`` requires a ``query`` argument; we use a
    permissive wildcard string and rely on the store's vector backend to
    return all items. Stores that ignore unsupported queries simply return
    everything in the namespace, which is what we want here.
    """
    namespace = (user_id, assistant_id, "quote")
    try:
        items = await store.asearch(namespace, query="*", limit=10000)
    except Exception as exc:
        logger.warning("asearch over quotes failed (%s); profile not built", exc)
        return []
    texts: List[str] = []
    for item in items or []:
        value = getattr(item, "value", {}) or {}
        # Documents stored via index_docs typically expose page_content
        # either at the top level or under a nested ``document`` blob.
        content = value.get("page_content")
        if not content and isinstance(value.get("document"), dict):
            content = (value["document"] or {}).get("page_content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return texts


async def _load_existing_profile(
    user_id: str, assistant_id: str, store: BaseStore
) -> Optional[Dict[str, Any]]:
    namespace = (user_id, assistant_id, STYLISTIC_PROFILE_NAMESPACE_TAG)
    try:
        item = await store.aget(namespace, STYLISTIC_PROFILE_KEY)
    except Exception:
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if isinstance(value, dict):
        return value
    return None


async def maybe_build_stylistic_profile(
    *,
    user_id: str,
    assistant_id: str,
    store: BaseStore,
    context: GlobalContext,
) -> Optional[Dict[str, Any]]:
    """Build or refresh the stylistic profile if thresholds are met.

    Returns the (possibly newly written) profile dict, or ``None`` if a
    build was skipped. Callers can ignore the return value; this helper
    exists primarily for the LangGraph node ``build_stylistic_fingerprint``.
    """
    if not user_id or not assistant_id:
        logger.info("Skipping stylistic profile: missing user/assistant id")
        return None

    quotes = await _enumerate_quote_texts(user_id, assistant_id, store)
    if len(quotes) < int(context.min_quotes_for_profile or 0):
        logger.info(
            "Stylistic profile build skipped: %d quotes < min_quotes_for_profile=%s",
            len(quotes),
            context.min_quotes_for_profile,
        )
        return None

    existing = await _load_existing_profile(user_id, assistant_id, store)
    if existing is not None:
        last_count = int(existing.get("built_with_document_count", 0) or 0)
        delta = len(quotes) - last_count
        if delta < int(context.profile_refresh_threshold or 0):
            logger.info(
                "Stylistic profile refresh skipped: only %d new quotes since last build (%d).",
                delta,
                last_count,
            )
            return existing

    target_name = (
        (existing or {}).get("target_name")
        or context.audio_diarization_known_speaker_name
    )
    profile = compute_profile_from_quotes(quotes, target_name=target_name)
    profile["built_with_document_count"] = len(quotes)
    profile["built_at"] = datetime.now(tz=timezone.utc).isoformat()

    namespace = (user_id, assistant_id, STYLISTIC_PROFILE_NAMESPACE_TAG)
    await store.aput(namespace, key=STYLISTIC_PROFILE_KEY, value=profile)
    logger.info(
        "Stylistic profile written for %s/%s (n=%d quotes)",
        user_id,
        assistant_id,
        len(quotes),
    )
    return profile


async def load_stylistic_profile(
    *, user_id: str, assistant_id: str, store: BaseStore
) -> Optional[Dict[str, Any]]:
    """Read-only fetch used by the authenticity evaluator."""
    return await _load_existing_profile(user_id, assistant_id, store)
