"""Gather every activity source for one period, each source independent of the others.

Every source returns one section shaped ``{"status", "detail", ...data}``:

- ``ok``          the source answered for this period;
- ``unavailable`` the source cannot answer right now (no machine online, the
                  screen history cannot be read in this process) and says why;
- ``error``       the source raised; the reason is reported and the other
                  sections are unaffected.

A source that fails never fails the review: the owner asking "what did I do
today" with the laptop asleep still gets the screen timeline, and an owner who
shared no screen today still gets the commits.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from src.anubis.utils.activity.periods import ActivityPeriod, resolve_timezone

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

MAXIMUM_REPOSITORIES_PER_REVIEW = 25
"""Repositories whose commit history is read in one review, most recently committed first."""


def _setting(context: Any, name: str, default: Any) -> Any:
    """Read a ``GlobalContext`` setting, falling back to ``default`` when empty."""
    value = getattr(context, name, None)
    return default if value in (None, "") else value


async def screen_section(
    context: Any,
    period: ActivityPeriod,
    *,
    user_id: str,
    assistant_id: str,
) -> dict[str, Any]:
    """Return the screen timeline for the period, read from the owner's checkpointed threads."""
    from src.anubis.utils.activity.screen_timeline import screen_timeline
    from src.anubis.utils.activity.thread_reader import (
        get_activity_graph,
        read_thread_messages,
        threads_touched_in_period,
    )
    from src.anubis.utils.runtime_handles import get_postgres_pool

    graph = get_activity_graph()
    pool = get_postgres_pool()
    if graph is None or pool is None:
        return {
            "status": STATUS_UNAVAILABLE,
            "detail": "The screen history cannot be read in this process (no API lifespan).",
        }
    thread_ids = await threads_touched_in_period(
        pool,
        user_id,
        assistant_id,
        period.period_start,
        period.period_end,
        int(_setting(context, "activity_review_maximum_threads", 200)),
    )
    messages_by_thread = await read_thread_messages(graph, thread_ids)
    owner_zone, _zone_name = resolve_timezone(period.timezone_name)
    timeline = screen_timeline(
        messages_by_thread,
        period.period_start,
        period.period_end,
        owner_zone=owner_zone,
        maximum_gap_seconds=float(
            _setting(context, "activity_review_maximum_gap_seconds", 660)
        ),
        maximum_blocks=int(_setting(context, "activity_review_maximum_blocks", 120)),
        maximum_description_characters=int(
            _setting(context, "activity_review_maximum_description_characters", 240)
        ),
    )
    return {"status": STATUS_OK, **timeline}


def _commit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the commit fields an activity answer uses; drop the per-file listing."""
    return {
        "device_label": row.get("device_label"),
        "authored_at": row.get("authored_at"),
        "subject": row.get("subject"),
        "files_changed": row.get("files_changed"),
        "insertions": row.get("insertions"),
        "deletions": row.get("deletions"),
    }


def _session_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the coding-session fields an activity answer uses."""
    return {
        "device_label": row.get("device_label"),
        "cwd": row.get("cwd"),
        "git_branch": row.get("git_branch"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "duration_minutes": row.get("duration_minutes"),
        "first_prompt": str(row.get("first_prompt") or "")[:200],
        "summary": row.get("summary"),
    }


def _committed_since(row: dict[str, Any], period_start: datetime) -> bool:
    """Whether a repository row's last commit is inside or after the period start."""
    stamp = str(row.get("last_commit_at") or "").strip()
    if not stamp:
        return False
    try:
        last_commit_at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if last_commit_at.tzinfo is None:
        last_commit_at = last_commit_at.replace(tzinfo=UTC)
    return last_commit_at >= period_start


async def development_section(
    context: Any, period: ActivityPeriod, connections: list[Any]
) -> dict[str, Any]:
    """Return commits and Claude Code sessions in the period from every live machine."""
    if not connections:
        return {
            "status": STATUS_UNAVAILABLE,
            "detail": "No machine running the Neural Nexus connector is online.",
        }
    from src.anubis.utils.tools.data_analysis.analysis_tools import (
        _device_fanout_timeout_seconds,
    )
    from src.anubis.utils.tools.data_analysis.development_tools import (
        DAEMON_GIT_LOG,
        DAEMON_LIST_CLAUDE_CODE_SESSIONS,
        DAEMON_LIST_GIT_REPOSITORIES,
        fan_out_development_tool,
    )

    timeout_seconds = _device_fanout_timeout_seconds(context)
    since_text = period.period_start.astimezone(UTC).isoformat()
    until_text = period.period_end.astimezone(UTC).isoformat()

    repositories_result, sessions_result = await asyncio.gather(
        fan_out_development_tool(
            connections,
            None,
            DAEMON_LIST_GIT_REPOSITORIES,
            {"root": "", "max_depth": 3},
            timeout_seconds,
            sort_key="last_commit_at",
        ),
        fan_out_development_tool(
            connections,
            None,
            DAEMON_LIST_CLAUDE_CODE_SESSIONS,
            {"since": since_text, "until": until_text, "max_sessions": 200},
            timeout_seconds,
            sort_key="started_at",
            sort_descending=False,
        ),
    )
    active_repositories = [
        row
        for row in repositories_result.get("rows") or []
        if row.get("path") and _committed_since(row, period.period_start)
    ][:MAXIMUM_REPOSITORIES_PER_REVIEW]

    async def commits_of(repository_row: dict[str, Any]) -> dict[str, Any]:
        log_result = await fan_out_development_tool(
            connections,
            repository_row.get("device_label"),
            DAEMON_GIT_LOG,
            {
                "repository": repository_row["path"],
                "since": since_text,
                "until": until_text,
                "max_commits": 200,
            },
            timeout_seconds,
            sort_key="authored_at",
            sort_descending=False,
        )
        commits = [_commit_row(row) for row in log_result.get("rows") or []]
        return {
            "repository": repository_row.get("name") or repository_row["path"],
            "path": repository_row["path"],
            "device_label": repository_row.get("device_label"),
            "commit_count": len(commits),
            "insertions": sum(int(commit.get("insertions") or 0) for commit in commits),
            "deletions": sum(int(commit.get("deletions") or 0) for commit in commits),
            "commits": commits,
        }

    repositories = [
        repository
        for repository in await asyncio.gather(
            *(commits_of(row) for row in active_repositories)
        )
        if repository["commit_count"]
    ]
    sessions = [_session_row(row) for row in sessions_result.get("rows") or []]
    session_minutes = sum(float(row.get("duration_minutes") or 0) for row in sessions)
    unreached = [
        *(repositories_result.get("unreached") or []),
        *(sessions_result.get("unreached") or []),
    ]
    visible_repository_count = len(
        [row for row in repositories_result.get("rows") or [] if row.get("path")]
    )
    development = {
        "status": STATUS_OK,
        "visible_repository_count": visible_repository_count,
        "commit_count": sum(repository["commit_count"] for repository in repositories),
        "repositories": repositories,
        "claude_code_session_count": len(sessions),
        "claude_code_session_minutes": round(session_minutes, 1),
        "claude_code_sessions": sessions,
        "unreached_machines": unreached,
    }
    if visible_repository_count == 0:
        # Zero commits from zero visible repositories is not "no commits": the
        # connector sees only its shared folders plus DEVELOPMENT_REPOSITORY_ROOTS,
        # and with neither holding a repository the commit history is unknown.
        development["commit_detail"] = (
            "No git repository is visible to the connector, so commits are unknown, "
            "not zero. Repositories become visible when the connector's "
            "DEVELOPMENT_REPOSITORY_ROOTS setting names the folder holding them."
        )
    return development


async def browser_section(
    context: Any,
    period: ActivityPeriod,
    connections: list[Any],
) -> dict[str, Any]:
    """Browsing in the period from every live machine, read without moving any watermark.

    The browsing-insights sweep keeps its own watermark per machine; this read
    passes the period start as ``since`` and never writes that watermark, so
    asking about today does not make the sweep skip anything.
    """
    if not connections:
        return {
            "status": STATUS_UNAVAILABLE,
            "detail": "No machine running the Neural Nexus connector is online.",
        }
    from src.anubis.utils.browsing.digest import summarize_visits
    from src.anubis.utils.browsing.history_client import read_new_visits

    maximum_visits = int(
        _setting(context, "activity_review_maximum_browser_visits", 5000)
    )
    since_text = period.period_start.astimezone(UTC).isoformat()
    results = await asyncio.gather(
        *(
            read_new_visits(
                connection, since_text, limit=maximum_visits, default_days=1
            )
            for connection in connections
        ),
        return_exceptions=True,
    )
    visits: list[dict[str, Any]] = []
    machines: list[dict[str, Any]] = []
    for connection, result in zip(connections, results):
        label = getattr(connection, "device_label", "")
        if isinstance(result, BaseException):
            machines.append(
                {"device_label": label, "status": STATUS_ERROR, "detail": str(result)}
            )
            continue
        status = result.get("status")
        machine_visits = [
            visit
            for visit in result.get("visits") or []
            if _visit_inside(visit, period)
        ]
        visits.extend(machine_visits)
        machines.append(
            {
                "device_label": label,
                "status": status,
                "detail": result.get("detail"),
                "visit_count": len(machine_visits),
                "truncated": int(result.get("visit_count") or 0) >= maximum_visits,
            }
        )
    if not any(machine.get("status") == STATUS_OK for machine in machines):
        return {
            "status": STATUS_UNAVAILABLE,
            "detail": "No machine answered for browser history.",
            "machines": machines,
        }
    owner_zone, _zone_name = resolve_timezone(period.timezone_name)
    summary = summarize_visits(visits, owner_zone=owner_zone)
    host_hours = _visits_by_hour_and_host(visits, owner_zone)
    return {
        "status": STATUS_OK,
        "machines": machines,
        "visit_count": summary["visit_count"],
        "distinct_hosts": summary["distinct_hosts"],
        "top_hosts": summary["top_hosts"],
        "searches": summary["searches"],
        "titles": summary["titles"],
        "visits_by_local_hour": summary["hours"],
        "top_host_by_local_hour": host_hours,
    }


def _visit_inside(visit: dict[str, Any], period: ActivityPeriod) -> bool:
    """Whether a visit falls inside the period; a visit with no readable time is kept."""
    stamp = str(visit.get("visited_at") or "").strip()
    if not stamp:
        return True
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return period.period_start <= moment < period.period_end


def _visits_by_hour_and_host(
    visits: list[dict[str, Any]], owner_zone: Any
) -> dict[str, str]:
    """Return the most visited host per local hour, keyed ``YYYY-MM-DD HH:00``."""
    from src.anubis.utils.browsing.digest import meaningful_visits

    counts: dict[str, Counter[str]] = {}
    for visit in meaningful_visits(visits):
        stamp = str(visit.get("visited_at") or "").strip()
        host = str(visit.get("host") or "")
        if not stamp or not host:
            continue
        try:
            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        hour_key = moment.astimezone(owner_zone).strftime("%Y-%m-%d %H:00")
        counts.setdefault(hour_key, Counter())[host] += 1
    return {
        hour_key: host_counts.most_common(1)[0][0]
        for hour_key, host_counts in sorted(counts.items())
    }


SourceCoroutineFactory = Callable[[], Awaitable[dict[str, Any]]]


async def gather_activity(
    context: Any,
    period: ActivityPeriod,
    *,
    user_id: str,
    assistant_id: str,
    connections: list[Any],
) -> dict[str, Any]:
    """Run every source concurrently and return one section per source.

    A new source is one more entry in ``source_factories``; nothing else in
    the review changes.
    """
    source_factories: dict[str, SourceCoroutineFactory] = {
        "screen": lambda: screen_section(
            context, period, user_id=user_id, assistant_id=assistant_id
        ),
        "development": lambda: development_section(context, period, connections),
        "browser": lambda: browser_section(context, period, connections),
    }
    names = list(source_factories)
    outcomes = await asyncio.gather(
        *(source_factories[name]() for name in names), return_exceptions=True
    )
    sections: dict[str, Any] = {}
    for name, outcome in zip(names, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("Activity source %s failed: %s", name, outcome)
            sections[name] = {
                "status": STATUS_ERROR,
                "detail": str(outcome) or type(outcome).__name__,
            }
        else:
            sections[name] = outcome
    return {
        **period.describe(),
        "sources": sections,
        "unavailable_sources": [
            name
            for name, section in sections.items()
            if section.get("status") != STATUS_OK
        ],
    }
