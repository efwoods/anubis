"""Development analytics: what shipped, what is in progress, and how long features take.

Inputs are the plain rows the development tools return from the owner's
machines (``src/anubis/utils/tools/data_analysis/development_tools.py``):
commits from ``git_log``, sessions from ``list_claude_code_sessions``, status
rows from ``git_status``, and repository rows from ``list_git_repositories``.
Every function here is pure except :func:`label_features`, which makes one
structured model call to group commits and sessions into named features.

Timestamps are ISO-8601 strings on the rows (``authored_at``, ``started_at``,
``ended_at``); each function parses them with :func:`parse_moment` and treats
a naive timestamp as UTC.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.anubis.utils.analytics.forecast import forecast_series

# Runnable tag every structured-output model call must carry so the streaming
# layer (``_stream_deep_agent`` in ``src/anubis/graph.py``) never forwards the
# JSON tokens to the user as a reply. The string value is copied from
# ``STRUCTURED_OUTPUT_STREAM_TAG`` in ``src/anubis/utils/model.py`` (re-exported
# by ``src/anubis/graph.py``) so this module does not import the graph.
STRUCTURED_OUTPUT_STREAM_TAG = "structured_output_no_user_stream"

# A commit counts as part of a session when the commit lands between the
# session start minus this margin and the session end plus this margin: a
# person often commits a little before the session opens (finishing earlier
# work) and well after the session closes (review, then commit).
SESSION_MATCH_LEAD = timedelta(minutes=30)
SESSION_MATCH_LAG = timedelta(hours=2)

# Upper bound on the compact listing handed to the labeling model.
LABELING_LISTING_LIMIT = 200

# Plan-document line markers that denote work not yet done.
_PLAN_ITEM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("checkbox", re.compile(r"^\s*[-*+]\s+\[\s\]\s+(?P<text>.+?)\s*$")),
    ("todo", re.compile(r"^\s*(?:[-*+]\s+)?(?:#+\s*)?TODO\b[:\s-]*(?P<text>.*?)\s*$")),
    ("planned", re.compile(r"^\s*(?:[-*+]\s+)?(?:#+\s*)?.*?🔲\s*(?P<text>.*?)\s*$")),
)

FeatureStatus = Literal["done", "in_progress", "planned"]


class FeatureLabel(BaseModel):
    """One feature the model recognized across commits, sessions, and plan items."""

    feature: str = Field(description="Short human-readable feature name.")
    commit_shas: list[str] = Field(
        default_factory=list,
        description="Commit shas (full or the listed prefix) belonging to the feature.",
    )
    session_ids: list[str] = Field(
        default_factory=list,
        description="Claude Code session identifiers belonging to the feature.",
    )
    status: FeatureStatus = Field(
        description=(
            "done when the commits complete the feature, in_progress when "
            "sessions or uncommitted work continue the feature, planned when "
            "only plan items mention the feature."
        )
    )


class FeatureLabeling(BaseModel):
    """The full grouping the labeling model returns."""

    features: list[FeatureLabel] = Field(default_factory=list)


def parse_moment(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` suffix accepted) into an aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _repository_path(repository: Any) -> str | None:
    """Normalize one repository entry (a string or a row with ``path``) to a path."""
    if isinstance(repository, dict):
        repository = repository.get("path")
    if not isinstance(repository, str) or not repository.strip():
        return None
    return repository.rstrip("/") or "/"


def _path_is_inside(path: str | None, repository_path: str) -> bool:
    """Return whether ``path`` equals ``repository_path`` or lies beneath the path."""
    if not path:
        return False
    normalized = path.rstrip("/") or "/"
    return normalized == repository_path or normalized.startswith(repository_path + "/")


def _repository_of_session(
    session: dict[str, Any], repository_paths: list[str]
) -> str | None:
    """Find the deepest repository path that contains the session's working directory."""
    working_directory = session.get("cwd")
    candidates = [
        repository_path
        for repository_path in repository_paths
        if _path_is_inside(working_directory, repository_path)
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def match_sessions_to_commits(
    sessions: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    repositories: list[Any],
) -> dict[str, Any]:
    """Pair coding sessions with the commits made during (or right after) them.

    A session belongs to the repository whose path contains the session
    ``cwd``. A commit belongs to a session when the commit is in that
    repository (the commit's ``repository`` field when present, otherwise any
    repository) and the commit's ``authored_at`` lies within
    ``[started_at - 30 minutes, ended_at + 2 hours]``.

    Returns ``{"matches": [{"session_id", "repository", "commit_shas"}],
    "sessions_by_commit": {sha: [session_id, ...]},
    "unmatched_commit_shas": [...]}``. Sessions outside every repository are
    omitted from ``matches``.
    """
    repository_paths = [
        path for path in (_repository_path(entry) for entry in repositories) if path
    ]
    matches: list[dict[str, Any]] = []
    sessions_by_commit: dict[str, list[str]] = {}
    matched_shas: set[str] = set()

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        repository_path = _repository_of_session(session, repository_paths)
        started_at = parse_moment(session.get("started_at"))
        ended_at = parse_moment(session.get("ended_at")) or started_at
        if not session_id or repository_path is None or started_at is None:
            continue
        window_start = started_at - SESSION_MATCH_LEAD
        window_end = ended_at + SESSION_MATCH_LAG
        commit_shas: list[str] = []
        for commit in commits:
            sha = str(commit.get("sha") or "")
            authored_at = parse_moment(commit.get("authored_at"))
            if not sha or authored_at is None:
                continue
            commit_repository = _repository_path(commit.get("repository"))
            if commit_repository is not None and commit_repository != repository_path:
                continue
            if window_start <= authored_at <= window_end:
                commit_shas.append(sha)
                matched_shas.add(sha)
                sessions_by_commit.setdefault(sha, []).append(session_id)
        matches.append(
            {
                "session_id": session_id,
                "repository": repository_path,
                "commit_shas": commit_shas,
            }
        )

    unmatched = [
        str(commit.get("sha"))
        for commit in commits
        if commit.get("sha") and str(commit.get("sha")) not in matched_shas
    ]
    return {
        "matches": matches,
        "sessions_by_commit": sessions_by_commit,
        "unmatched_commit_shas": unmatched,
    }


def _short(text: Any, limit: int) -> str:
    """One-line prefix of ``text`` no longer than ``limit`` characters."""
    single_line = " ".join(str(text or "").split())
    return single_line[:limit]


def build_labeling_listing(
    commits: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    plan_items: list[Any],
    *,
    limit: int = LABELING_LISTING_LIMIT,
) -> str:
    """Compact text listing of commits, sessions, and plan items for the labeling model.

    The three groups share one ``limit``; commits are listed first because
    they are the strongest evidence, then sessions, then plan items.
    """
    lines: list[str] = []
    remaining = max(0, int(limit))

    commit_lines = [
        f"commit {str(commit.get('sha') or '')[:12]} "
        f"{_short(commit.get('authored_at'), 25)} "
        f"{_short(commit.get('subject'), 120)}"
        for commit in commits
    ]
    session_lines = [
        f"session {session.get('session_id')} "
        f"{_short(session.get('started_at'), 25)} "
        f"{_short(session.get('duration_minutes'), 8)}min "
        f"branch={_short(session.get('git_branch'), 40)} "
        f"prompt={_short(session.get('first_prompt') or session.get('summary'), 160)}"
        for session in sessions
    ]
    plan_lines = [
        f"plan {_short(item.get('text') if isinstance(item, dict) else item, 160)}"
        for item in plan_items
    ]
    for group_name, group_lines in (
        ("Commits", commit_lines),
        ("Sessions", session_lines),
        ("Plan items", plan_lines),
    ):
        if remaining <= 0:
            break
        taken = group_lines[:remaining]
        remaining -= len(taken)
        lines.append(f"{group_name} ({len(taken)} of {len(group_lines)}):")
        lines.extend(taken)
    return "\n".join(lines)


LABELING_INSTRUCTIONS = """You group software development activity into named features.

Input: a listing of git commits (sha prefix, time, subject), Claude Code coding sessions (session identifier, start time, duration, branch, first prompt), and plan items (unfinished lines from planning documents).

Task: return every distinct feature you can recognize. For each feature give a short name, the commit sha prefixes that belong to the feature, the session identifiers that belong to the feature, and a status:
- "done" when commits complete the feature and no session or plan item continues the feature,
- "in_progress" when sessions or plan items continue a feature that also has commits,
- "planned" when only plan items mention the feature.

Rules: use each commit sha and each session identifier at most once; copy identifiers exactly as listed; do not invent identifiers; group by intent (the same feature across several commits and sessions), not by file name."""


async def label_features(
    context: Any,
    commits: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    plan_items: list[Any],
    *,
    model: Any = None,
) -> FeatureLabeling:
    """Group commits, sessions, and plan items into named features with one model call.

    ``model`` is injectable for tests: any object whose
    ``with_structured_output(FeatureLabeling)`` returns a runnable with
    ``ainvoke``. When ``model`` is omitted the classification model from
    ``init_model`` is used, which already applies ``with_structured_output``
    and the ``STRUCTURED_OUTPUT_STREAM_TAG`` tag.
    """
    listing = build_labeling_listing(commits, sessions, plan_items)
    if model is None:
        from src.anubis.utils.model import init_model

        structured_model = init_model(context, response_format=FeatureLabeling)
    else:
        structured_model = model.with_structured_output(FeatureLabeling).with_config(
            tags=[STRUCTURED_OUTPUT_STREAM_TAG]
        )
    result = await structured_model.ainvoke(
        [
            {"role": "system", "content": LABELING_INSTRUCTIONS},
            {"role": "user", "content": listing or "No activity was listed."},
        ]
    )
    if isinstance(result, FeatureLabeling):
        return result
    if isinstance(result, dict):
        return FeatureLabeling.model_validate(result)
    return FeatureLabeling.model_validate(result, from_attributes=True)


def merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping or touching ``(start, end)`` intervals, sorted by start."""
    ordered = sorted(
        ((start, end) for start, end in intervals if start <= end),
        key=lambda interval: interval[0],
    )
    merged: list[tuple[datetime, datetime]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _session_interval(session: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """Read the ``(started_at, ended_at)`` pair of one session row, when both parse."""
    started_at = parse_moment(session.get("started_at"))
    ended_at = parse_moment(session.get("ended_at")) or started_at
    if started_at is None or ended_at is None:
        return None
    return started_at, ended_at


def session_hours(sessions: list[dict[str, Any]]) -> float:
    """Hours covered by the sessions, counting overlapping sessions once."""
    intervals = [
        interval
        for interval in (_session_interval(session) for session in sessions)
        if interval is not None
    ]
    total_seconds = sum(
        (end - start).total_seconds() for start, end in merge_intervals(intervals)
    )
    return round(total_seconds / 3600.0, 2)


def _commit_matches_prefixes(sha: str, prefixes: list[str]) -> bool:
    """Return whether ``sha`` starts with any listed prefix (the model may shorten shas)."""
    return any(prefix and sha.startswith(prefix) for prefix in prefixes)


def feature_effort(
    labeling: FeatureLabeling,
    sessions: list[dict[str, Any]],
    commits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Effort per labeled feature: session hours, calendar span, commits, and line counts.

    Returns one row per feature: ``{feature, status, session_count, hours,
    commit_count, insertions, deletions, first_activity_at, last_activity_at,
    calendar_days}``. Hours count overlapping sessions once; the calendar span
    runs from the earliest session start or commit to the latest session end
    or commit.
    """
    sessions_by_id = {
        str(session.get("session_id")): session
        for session in sessions
        if session.get("session_id")
    }
    rows: list[dict[str, Any]] = []
    for feature in labeling.features:
        feature_sessions = [
            sessions_by_id[session_id]
            for session_id in feature.session_ids
            if session_id in sessions_by_id
        ]
        feature_commits = [
            commit
            for commit in commits
            if _commit_matches_prefixes(
                str(commit.get("sha") or ""), feature.commit_shas
            )
        ]
        moments: list[datetime] = []
        for session in feature_sessions:
            interval = _session_interval(session)
            if interval is not None:
                moments.extend(interval)
        for commit in feature_commits:
            authored_at = parse_moment(commit.get("authored_at"))
            if authored_at is not None:
                moments.append(authored_at)
        first_activity = min(moments) if moments else None
        last_activity = max(moments) if moments else None
        calendar_days = (
            round((last_activity - first_activity).total_seconds() / 86400.0, 2)
            if first_activity is not None and last_activity is not None
            else 0.0
        )
        rows.append(
            {
                "feature": feature.feature,
                "status": feature.status,
                "session_count": len(feature_sessions),
                "hours": session_hours(feature_sessions),
                "commit_count": len(feature_commits),
                "insertions": sum(
                    int(commit.get("insertions") or 0) for commit in feature_commits
                ),
                "deletions": sum(
                    int(commit.get("deletions") or 0) for commit in feature_commits
                ),
                "first_activity_at": first_activity.isoformat()
                if first_activity
                else None,
                "last_activity_at": last_activity.isoformat()
                if last_activity
                else None,
                "calendar_days": calendar_days,
            }
        )
    return rows


def _within(
    moment: datetime | None, since: datetime | None, until: datetime | None
) -> bool:
    """Return whether ``moment`` parses and lies inside the optional bounds."""
    if moment is None:
        return False
    if since is not None and moment < since:
        return False
    if until is not None and moment > until:
        return False
    return True


def sprint_summary(
    commits: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    since: Any,
    until: Any,
) -> dict[str, Any]:
    """Totals for one window: commits, lines, files, sessions, hours, active days.

    ``since`` and ``until`` are ISO-8601 strings or datetimes; either may be
    ``None``. Commits are filtered by ``authored_at`` and sessions by
    ``started_at``. Per-repository and per-author breakdowns use the optional
    ``repository`` and ``author`` fields on the commit rows; the most recent
    commit subjects are listed as ``highlights``.
    """
    since_moment = parse_moment(since)
    until_moment = parse_moment(until)
    window_commits = [
        commit
        for commit in commits
        if _within(parse_moment(commit.get("authored_at")), since_moment, until_moment)
    ]
    window_sessions = [
        session
        for session in sessions
        if _within(parse_moment(session.get("started_at")), since_moment, until_moment)
    ]

    active_days: set[str] = set()
    for commit in window_commits:
        authored_at = parse_moment(commit.get("authored_at"))
        if authored_at is not None:
            active_days.add(authored_at.date().isoformat())
    for session in window_sessions:
        started_at = parse_moment(session.get("started_at"))
        if started_at is not None:
            active_days.add(started_at.date().isoformat())

    by_repository: dict[str, dict[str, int]] = {}
    by_author: dict[str, int] = {}
    for commit in window_commits:
        repository_key = _repository_path(commit.get("repository")) or "unknown"
        bucket = by_repository.setdefault(
            repository_key, {"commits": 0, "insertions": 0, "deletions": 0}
        )
        bucket["commits"] += 1
        bucket["insertions"] += int(commit.get("insertions") or 0)
        bucket["deletions"] += int(commit.get("deletions") or 0)
        author = str(commit.get("author") or "unknown")
        by_author[author] = by_author.get(author, 0) + 1

    ordered_commits = sorted(
        window_commits,
        key=lambda commit: str(commit.get("authored_at") or ""),
        reverse=True,
    )
    return {
        "since": since_moment.isoformat() if since_moment else None,
        "until": until_moment.isoformat() if until_moment else None,
        "commit_count": len(window_commits),
        "insertions": sum(
            int(commit.get("insertions") or 0) for commit in window_commits
        ),
        "deletions": sum(
            int(commit.get("deletions") or 0) for commit in window_commits
        ),
        "files_changed": sum(
            int(commit.get("files_changed") or 0) for commit in window_commits
        ),
        "session_count": len(window_sessions),
        "session_hours": session_hours(window_sessions),
        "active_days": len(active_days),
        "by_repository": by_repository,
        "by_author": by_author,
        "highlights": [
            str(commit.get("subject") or "") for commit in ordered_commits[:10]
        ],
    }


def work_in_progress(
    git_status_rows: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    active_days: int = 3,
    now: Any = None,
) -> dict[str, Any]:
    """Report what is unfinished right now: dirty repositories and recently active sessions.

    A repository is dirty when the status row has staged, modified, or
    untracked paths, or commits ahead of the upstream. A session is recent
    when the session ended within ``active_days`` of ``now`` (default: the
    current time).
    """
    now_moment = parse_moment(now) or datetime.now(UTC)
    cutoff = now_moment - timedelta(days=max(0, int(active_days)))

    dirty_repositories: list[dict[str, Any]] = []
    for row in git_status_rows:
        staged = list(row.get("staged") or [])
        modified = list(row.get("modified") or [])
        untracked = list(row.get("untracked") or [])
        ahead = int(row.get("ahead") or 0)
        if not (staged or modified or untracked or ahead > 0):
            continue
        dirty_repositories.append(
            {
                "repository": row.get("repository"),
                "device_label": row.get("device_label"),
                "branch": row.get("branch"),
                "upstream": row.get("upstream"),
                "ahead": ahead,
                "behind": int(row.get("behind") or 0),
                "staged_count": len(staged),
                "modified_count": len(modified),
                "untracked_count": len(untracked),
                "paths": sorted(set(staged) | set(modified) | set(untracked))[:20],
            }
        )

    recent_sessions: list[dict[str, Any]] = []
    for session in sessions:
        ended_at = parse_moment(session.get("ended_at")) or parse_moment(
            session.get("started_at")
        )
        if ended_at is None or ended_at < cutoff:
            continue
        recent_sessions.append(
            {
                "session_id": session.get("session_id"),
                "device_label": session.get("device_label"),
                "cwd": session.get("cwd"),
                "git_branch": session.get("git_branch"),
                "ended_at": ended_at.isoformat(),
                "first_prompt": session.get("first_prompt"),
                "summary": session.get("summary"),
            }
        )
    recent_sessions.sort(key=lambda entry: entry["ended_at"], reverse=True)
    return {
        "as_of": now_moment.isoformat(),
        "active_days": int(active_days),
        "dirty_repositories": dirty_repositories,
        "recent_sessions": recent_sessions,
    }


def upcoming_from_plans(plan_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unfinished items from planning documents.

    Each document is ``{"path", "text"}``. A line counts when the line is an
    unchecked checkbox (``- [ ] ...``), carries a ``TODO`` marker, or carries
    the ``🔲`` planned marker. Returns ``[{path, line_number, marker, text}]``
    in document order.
    """
    items: list[dict[str, Any]] = []
    for document in plan_documents:
        path = str(document.get("path") or "")
        text = str(document.get("text") or "")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for marker, pattern in _PLAN_ITEM_PATTERNS:
                match = pattern.match(line)
                if match is None:
                    continue
                item_text = match.group("text").strip() or line.strip()
                items.append(
                    {
                        "path": path,
                        "line_number": line_number,
                        "marker": marker,
                        "text": item_text,
                    }
                )
                break
    return items


def _history_hours(history: list[Any]) -> list[float]:
    """Hours per historical feature, from effort rows or plain numbers, in order."""
    hours: list[float] = []
    for entry in history:
        value = entry.get("hours") if isinstance(entry, dict) else entry
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            hours.append(number)
    return hours


def forecast_feature_time(history: list[Any], planned: list[Any]) -> dict[str, Any]:
    """Estimate hours for each planned feature from the hours past features took.

    ``history`` is the chronological list of :func:`feature_effort` rows (or
    plain hour values); ``planned`` lists the upcoming features (strings, plan
    items, or ``FeatureLabel`` rows). The band comes from
    :func:`forecast_series` over the historical hours; with fewer than three
    historical points the estimate falls back to the mean and no band.
    """
    hours = _history_hours(history)
    planned_names = [
        (
            entry.get("feature") or entry.get("text")
            if isinstance(entry, dict)
            else getattr(entry, "feature", None) or str(entry)
        )
        for entry in planned
    ]
    if not planned_names:
        return {
            "planned": [],
            "history_points": len(hours),
            "method": "none",
            "total_hours": 0.0,
        }

    forecast = forecast_series(hours, horizon=len(planned_names))
    if "error" in forecast:
        mean_hours = round(sum(hours) / len(hours), 2) if hours else None
        planned_rows = [
            {
                "feature": name,
                "point_hours": mean_hours,
                "lower_hours": None,
                "upper_hours": None,
            }
            for name in planned_names
        ]
        return {
            "planned": planned_rows,
            "history_points": len(hours),
            "method": "mean" if hours else "none",
            "total_hours": round((mean_hours or 0.0) * len(planned_names), 2),
        }

    planned_rows = [
        {
            "feature": name,
            "point_hours": round(max(0.0, forecast["point"][index]), 2),
            "lower_hours": round(max(0.0, forecast["lower"][index]), 2),
            "upper_hours": round(max(0.0, forecast["upper"][index]), 2),
        }
        for index, name in enumerate(planned_names)
    ]
    return {
        "planned": planned_rows,
        "history_points": len(hours),
        "method": forecast["method"],
        "slope_hours_per_feature": forecast["slope_per_step"],
        "total_hours": round(sum(row["point_hours"] for row in planned_rows), 2),
    }
