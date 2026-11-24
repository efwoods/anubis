"""Resolve a spoken period ("today", "this week") to exact bounds in the owner's zone.

"Today" means the owner's calendar day, not the server's and not the UTC day:
an owner in New York asking at 9 p.m. would otherwise lose the evening to the
next UTC day. The bounds are aware datetimes, so every source compares them
against timestamps without guessing a zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

PERIOD_TODAY = "today"
PERIOD_YESTERDAY = "yesterday"
PERIOD_THIS_WEEK = "this_week"
PERIOD_LAST_WEEK = "last_week"
PERIOD_THIS_MONTH = "this_month"
PERIOD_CUSTOM = "custom"

ACTIVITY_PERIODS = (
    PERIOD_TODAY,
    PERIOD_YESTERDAY,
    PERIOD_THIS_WEEK,
    PERIOD_LAST_WEEK,
    PERIOD_THIS_MONTH,
    PERIOD_CUSTOM,
)


@dataclass(frozen=True)
class ActivityPeriod:
    """One resolved period: aware bounds plus the zone the bounds were resolved in."""

    name: str
    period_start: datetime
    period_end: datetime
    timezone_name: str

    def describe(self) -> dict[str, str]:
        """Return the period as the review reports the period to the avatar."""
        return {
            "period": self.name,
            "timezone": self.timezone_name,
            "period_start_local": self.period_start.isoformat(),
            "period_end_local": self.period_end.isoformat(),
            "period_start_utc": self.period_start.astimezone(UTC).isoformat(),
            "period_end_utc": self.period_end.astimezone(UTC).isoformat(),
        }


def resolve_timezone(timezone_name: str | None) -> tuple[tzinfo, str]:
    """Return the owner's zone and the zone's name; UTC when the name is missing or unknown."""
    if not timezone_name:
        return UTC, "UTC"
    try:
        return ZoneInfo(timezone_name), timezone_name
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown owner timezone %r; activity review uses UTC", timezone_name
        )
        return UTC, "UTC"


def _local_midnight(day: datetime, zone: tzinfo) -> datetime:
    """Midnight at the start of ``day``'s calendar date in ``zone``."""
    return datetime.combine(day.date(), time.min, tzinfo=zone)


def _parse_bound(text: str, zone: tzinfo, *, end_of_day: bool) -> datetime:
    """Parse an ISO date or datetime; a bare date or naive time is read in the owner's zone."""
    stripped = text.strip()
    parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    if len(stripped) == 10:
        # A bare date: the start of that day, or the start of the following day
        # when the date is an inclusive upper bound ("until 2026-11-17").
        parsed = datetime.combine(parsed.date(), time.min, tzinfo=zone)
        return parsed + timedelta(days=1) if end_of_day else parsed
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def resolve_activity_period(
    period: str | None,
    since: str | None,
    until: str | None,
    timezone_name: str | None,
    *,
    now: datetime | None = None,
) -> ActivityPeriod:
    """Resolve a named period, or explicit ``since`` / ``until`` bounds, in the owner's zone.

    Weeks start on Monday. A period that has not finished ends at ``now``, so
    "today" never reaches into the future. Explicit bounds win over the period
    name: when ``since`` is given the period is ``custom``. Raises ``ValueError``
    on an unknown period name or an unparseable bound.
    """
    zone, resolved_zone_name = resolve_timezone(timezone_name)
    current = (now or datetime.now(UTC)).astimezone(zone)
    name = (period or PERIOD_TODAY).strip().lower().replace(" ", "_")

    if since:
        period_start = _parse_bound(since, zone, end_of_day=False)
        period_end = _parse_bound(until, zone, end_of_day=True) if until else current
        name = PERIOD_CUSTOM
    elif name == PERIOD_TODAY:
        period_start, period_end = _local_midnight(current, zone), current
    elif name == PERIOD_YESTERDAY:
        today_start = _local_midnight(current, zone)
        period_start = _local_midnight(today_start - timedelta(hours=12), zone)
        period_end = today_start
    elif name == PERIOD_THIS_WEEK:
        monday = current - timedelta(days=current.weekday())
        period_start, period_end = _local_midnight(monday, zone), current
    elif name == PERIOD_LAST_WEEK:
        this_monday = _local_midnight(current - timedelta(days=current.weekday()), zone)
        last_monday = _local_midnight(this_monday - timedelta(days=6, hours=12), zone)
        period_start, period_end = last_monday, this_monday
    elif name == PERIOD_THIS_MONTH:
        period_start = datetime.combine(
            current.date().replace(day=1), time.min, tzinfo=zone
        )
        period_end = current
    else:
        raise ValueError(
            f"Unknown period {period!r}; use one of {', '.join(ACTIVITY_PERIODS)} "
            "or pass since/until."
        )
    if period_end <= period_start:
        raise ValueError("The period ends before the period starts.")
    return ActivityPeriod(
        name=name,
        period_start=period_start,
        period_end=period_end,
        timezone_name=resolved_zone_name,
    )
