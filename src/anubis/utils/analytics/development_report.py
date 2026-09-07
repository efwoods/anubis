"""The development-report tool: git, Claude Code sessions, GitHub → features, hours, forecasts.

The desktop daemon answers "which commits and which coding sessions happened";
the GitHub connection answers "which issues and pull requests"; this tool
turns those rows into what the owner asks for: features and how long each
took, what happened in a period, what is in progress, what is upcoming, and
how long planned work is expected to take. The avatar charts the result
with ``make_chart`` and saves the report with ``save_report``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

DEVELOPMENT_REPORT_TOOL_NAMES: tuple[str, ...] = ("development_report",)


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_development_report_tools(
    context: Any,
    *,
    live_connections: list[Any],
    connected_accounts: list[dict[str, Any]],
    store: Any = None,
) -> list[Any]:
    """Build the development-report tool when a machine or GitHub is connected."""
    has_devices = bool(live_connections)
    github_accounts = [record for record in connected_accounts if record.get("provider") == "github"]
    if not has_devices and not github_accounts:
        return []

    @tool
    async def development_report(
        since: str | None = None,
        until: str | None = None,
        repositories: str | None = None,
        github_repository: str | None = None,
        planned_features: str | None = None,
        hourly_rate_usd: float | None = None,
    ) -> dict[str, Any]:
        """Report on development in a period: features, hours per feature, sprint summary, in progress, upcoming, forecast.

        Use for "what happened in the last sprint", "what is in progress",
        "how long did <feature> take", "what is upcoming", "cost to develop a
        feature", and "next quarter's forecast". ``since``/``until`` are ISO
        dates (default the last 14 days); ``repositories`` is a comma-separated
        list of repository paths on the connected machine (default: every
        repository found); ``github_repository`` is owner/name for issues and
        pull requests; ``planned_features`` is a comma-separated list of
        features to forecast; ``hourly_rate_usd`` prices the hours. Chart the
        result with make_chart and save the report with save_report (kind
        "development").
        """
        from src.anubis.utils.analytics import development

        now = datetime.now(UTC)
        end = _parse(until) or now
        start = _parse(since) or (end - timedelta(days=14))
        commits: list[dict[str, Any]] = []
        sessions: list[dict[str, Any]] = []
        repository_rows: list[dict[str, Any]] = []
        status_rows: list[dict[str, Any]] = []
        plan_documents: list[dict[str, Any]] = []
        notes: list[str] = []

        if has_devices:
            from src.anubis.utils.tools.data_analysis.development_tools import (
                build_development_tools,
            )

            tools = {entry.name: entry for entry in build_development_tools(context, live_connections)}
            try:
                found = await tools["list_git_repositories"].coroutine()
                repository_rows = list(found.get("rows") or [])
            except Exception as list_error:
                notes.append(f"Repositories could not be listed: {list_error}")
            wanted = [entry.strip() for entry in str(repositories or "").split(",") if entry.strip()]
            targets = wanted or [str(row.get("path") or "") for row in repository_rows]
            for path in targets[:12]:
                if not path:
                    continue
                try:
                    log = await tools["git_log"].coroutine(
                        repository=path, since=start.isoformat(), until=end.isoformat(), max_commits=300
                    )
                    for row in log.get("rows") or []:
                        commits.append({**row, "repository": path})
                except Exception as log_error:
                    notes.append(f"{path}: {log_error}")
                try:
                    status = await tools["git_status"].coroutine(repository=path)
                    for row in status.get("rows") or []:
                        status_rows.append({**row, "repository": path})
                except Exception:
                    pass
            try:
                listed = await tools["list_claude_code_sessions"].coroutine(
                    since=start.isoformat(), until=end.isoformat(), max_sessions=300
                )
                sessions = list(listed.get("rows") or [])
            except Exception as session_error:
                notes.append(f"Coding sessions could not be listed: {session_error}")

        github_issues: list[dict[str, Any]] = []
        github_pulls: list[dict[str, Any]] = []
        if github_accounts and github_repository:
            from src.anubis.utils.connected_accounts.vendor_api_tools import (
                build_vendor_api_tools,
            )

            github_tools = {
                entry.name: entry
                for entry in build_vendor_api_tools(context, github_accounts, store=store)
            }
            try:
                issues = await github_tools["github_issues"].coroutine(repository=github_repository, state="open")
                github_issues = list(issues.get("issues") or [])
                pulls = await github_tools["github_pull_requests"].coroutine(repository=github_repository, state="all")
                github_pulls = list(pulls.get("pull_requests") or [])
                if not commits:
                    activity = await github_tools["github_activity"].coroutine(
                        repository=github_repository, since=start.isoformat(), until=end.isoformat()
                    )
                    commits = [{**row, "repository": github_repository} for row in activity.get("commits") or []]
            except Exception as github_error:
                notes.append(f"GitHub: {github_error}")
            plan_documents.extend(
                {"path": f"github:{github_repository}#{issue.get('number')}", "text": f"- [ ] {issue.get('title')}"}
                for issue in github_issues
            )

        plan_items = development.upcoming_from_plans(plan_documents) if plan_documents else []
        matched = development.match_sessions_to_commits(sessions, commits, repository_rows) if (sessions and commits) else []
        labeling = None
        effort: list[dict[str, Any]] = []
        if commits or sessions:
            try:
                labeling = await development.label_features(context, commits, sessions, plan_items)
                effort = development.feature_effort(labeling, sessions, commits)
            except Exception as label_error:
                notes.append(f"Features could not be labelled: {label_error}")
        summary = development.sprint_summary(commits, sessions, start, end)
        in_progress = development.work_in_progress(status_rows, sessions)
        planned = [
            {"feature": entry.strip()}
            for entry in str(planned_features or "").split(",")
            if entry.strip()
        ] or [{"feature": item.get("text") or item.get("feature")} for item in plan_items[:10]]
        forecast = (
            development.forecast_feature_time(effort, planned) if effort and planned else {}
        )
        if hourly_rate_usd:
            for entry in effort:
                entry["cost_usd"] = round(float(entry.get("hours") or 0.0) * float(hourly_rate_usd), 2)
            for entry in forecast.get("planned") or []:
                if isinstance(entry, dict):
                    entry["expected_cost_usd"] = round(
                        float(entry.get("expected_hours") or 0.0) * float(hourly_rate_usd), 2
                    )
        return {
            "status": "ok",
            "since": start.isoformat(),
            "until": end.isoformat(),
            "sprint_summary": summary,
            "features": effort,
            "matched_sessions": len(matched),
            "in_progress": in_progress,
            "upcoming": plan_items[:50],
            "github_open_issues": len(github_issues),
            "github_pull_requests": github_pulls[:50],
            "forecast": forecast,
            "notes": notes,
        }

    return [development_report]
