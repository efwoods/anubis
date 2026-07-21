# src/anubis/utils/billing/system_prompt_estimate_cache.py

"""Process-wide cache of measured system-prompt token estimates.

The message allotment gate must know how large the REAL system prompt is
before the model runs — the dynamically built consciousness prompt (identity
documents, recalled memories, style profile) dominates a message turn's input
tokens, and a guessed constant underestimates by tens of thousands of tokens
for identity-rich avatars. Measuring means building the prompt, which costs a
round of store retrievals; this cache makes that cost once-per-conversation
instead of once-per-request.

Producer: ``_build_consciousness_system_message_update``
(``src/anubis/utils/nodes.py``) records the measured word-ratio token estimate
of every system prompt the moment the prompt is built — token usage is
estimated when token usage occurs. Consumer: the message endpoints'
pre-request estimator reads the cached measurement; on a miss the endpoint
builds the prompt through ``build_system_prompt_text_for_estimation`` (which
records here) and fails closed if the build fails.

The FastAPI webapp and every LangGraph graph run inside the same
``langgraph-api`` server process (the same property ``store_cache.py`` relies
on), so producer writes reach consumer reads in process memory. Staleness is
bounded by the caller-supplied maximum age (environment variable
``SYSTEM_PROMPT_TOKEN_ESTIMATE_CACHE_TTL_SECONDS``): identity uploads can grow
the prompt between turns, and every real turn's producer write self-heals the
estimate. Values are small integers, so the entry bound is generous.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from src.anubis.utils.billing.estimation import (
    count_words,
    estimate_text_tokens_from_words,
)

# Default staleness bound when the caller supplies no maximum age.
DEFAULT_SYSTEM_PROMPT_ESTIMATE_MAX_AGE_SECONDS: float = 300.0

# Upper bound on resident entries (one small integer per (user, avatar) pair).
SYSTEM_PROMPT_ESTIMATE_CACHE_MAX_ENTRIES: int = 4096

# Maps (user_id, assistant_id) -> (monotonic time recorded, estimated tokens).
# Ordered so the least-recently-used entry sits first for eviction.
_system_prompt_estimate_cache: OrderedDict[tuple[str, str], tuple[float, int]] = (
    OrderedDict()
)


def record_system_prompt_token_estimate(
    user_id: str, assistant_id: str, system_prompt_text: str
) -> int:
    """Measure and cache the token estimate of a just-built system prompt.

    Called at the single point where the system prompt text exists — the
    moment ``_build_consciousness_system_message_update`` finishes building
    the prompt — so the cached measurement always reflects the prompt the
    model will actually read. Returns the estimate for callers that need the
    value immediately.
    """
    estimated_tokens = estimate_text_tokens_from_words(
        count_words(system_prompt_text)
    )
    cache_key = (str(user_id), str(assistant_id))
    _system_prompt_estimate_cache[cache_key] = (time.monotonic(), estimated_tokens)
    _system_prompt_estimate_cache.move_to_end(cache_key)
    while (
        len(_system_prompt_estimate_cache)
        > SYSTEM_PROMPT_ESTIMATE_CACHE_MAX_ENTRIES
    ):
        _system_prompt_estimate_cache.popitem(last=False)
    return estimated_tokens


def fetch_system_prompt_token_estimate(
    user_id: str,
    assistant_id: str,
    max_age_seconds: float = DEFAULT_SYSTEM_PROMPT_ESTIMATE_MAX_AGE_SECONDS,
) -> int | None:
    """Return the cached system-prompt token estimate, or ``None`` when absent/stale.

    ``None`` tells the caller to build the prompt (recording a fresh
    measurement) before estimating — never to fall back to a guessed
    constant.
    """
    cache_key = (str(user_id), str(assistant_id))
    cached_entry = _system_prompt_estimate_cache.get(cache_key)
    if cached_entry is None:
        return None
    recorded_at, estimated_tokens = cached_entry
    if time.monotonic() - recorded_at >= max_age_seconds:
        return None
    _system_prompt_estimate_cache.move_to_end(cache_key)
    return estimated_tokens


def invalidate_system_prompt_token_estimate(
    user_id: str, assistant_id: str
) -> None:
    """Drop one cached measurement (identity writes may grow the prompt)."""
    _system_prompt_estimate_cache.pop((str(user_id), str(assistant_id)), None)
