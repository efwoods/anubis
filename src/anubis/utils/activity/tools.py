"""The ``review_my_activity`` tool: the owner asks what the owner did in a period.

Attached on the owner's personal avatar only. The tool resolves the period in
the owner's time zone, gathers every source (``sources.gather_activity``), and
returns the sections as structured data. The avatar's own turn writes the
answer; the tool makes no model call and writes nothing.
"""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from src.anubis.utils.activity.periods import resolve_activity_period
from src.anubis.utils.activity.sources import gather_activity


def activity_review_enabled(context: Any) -> bool:
    """Whether ``ACTIVITY_REVIEW_ENABLED`` is TRUE (the default)."""
    value = getattr(context, "activity_review_enabled", "TRUE")
    return str(value if value not in (None, "") else "TRUE").strip().upper() == "TRUE"


def build_activity_tools(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    timezone_name: str | None,
    connections: list[Any],
) -> list[Any]:
    """Build the per-turn activity review tool for the owner's personal avatar.

    ``connections`` are the live machines bound to the avatar for this turn;
    an empty list leaves the screen timeline working and reports the machine
    sources as unavailable.
    """

    @tool
    async def review_my_activity(
        period: str = "today",
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Review what the owner did in a period: screen activity, commits, coding sessions, browsing.

        Call this tool when the owner asks how the owner's time was spent, how
        productive the owner has been, what the owner worked on, or what the
        owner accomplished — today, yesterday, this week, last week, this
        month, or between two dates. The review is read from the screen
        observations already kept on the owner's conversations with this
        avatar and from the owner's connected machines; nothing is stored.

        The result holds one section per source under ``sources``:
        ``screen`` (segments of activity with local start and end times,
        seconds per activity, observed and not-observed seconds), ``development``
        (commits per repository and Claude Code sessions), and ``browser``
        (visits, top sites, searches, the busiest site per hour).

        When answering:
        - Describe the timeline and name the evidence: what was on screen,
          which commits landed, which coding sessions ran, which sites took
          the time. Do not invent a productivity score or a percentage the
          sections do not contain.
        - Report screen time from ``observed_seconds`` and say that the
          figures come from observation spacing (``duration_method``).
          Hours in ``not_observed_seconds`` were not watched: never describe
          that time as idle or as work.
        - Name every source listed in ``unavailable_sources`` and the reason
          in that section's ``detail``, so the owner knows what the answer
          could not see.
        - Give times in the owner's local time.

        Args:
            period: One of "today", "yesterday", "this_week", "last_week",
                "this_month". Weeks start on Monday in the owner's time zone.
            since: Optional ISO date or datetime for a custom start, for
                example "2026-11-10". A date or a time without a zone is read
                in the owner's time zone. Overrides ``period``.
            until: Optional ISO date or datetime for a custom end; a bare date
                includes that whole day. Defaults to now.
        """
        try:
            resolved_period = resolve_activity_period(
                period, since, until, timezone_name
            )
        except ValueError as period_error:
            return {"status": "error", "detail": str(period_error)}
        return await gather_activity(
            context,
            resolved_period,
            user_id=user_id,
            assistant_id=assistant_id,
            connections=connections,
        )

    return [review_my_activity]
