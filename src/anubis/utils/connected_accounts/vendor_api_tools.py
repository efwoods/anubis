"""Tools over vendor APIs the owner authorised through OAuth.

GitHub, X, Vercel, Google Calendar, Google Analytics, and YouTube each have an
official API; the popup sign-in stored a token bundle and this module presents
a fresh access token (``get_fresh_access_token``) to that API. Every call is a
small ``httpx`` request; a lapsed token reports ``needs_reconnect`` so the
avatar re-raises the card instead of failing silently.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

VENDOR_API_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "github": ("github_activity", "github_issues", "github_pull_requests"),
    "twitter": ("x_recent_posts", "x_post_reply"),
    "vercel": ("vercel_deployments", "vercel_usage"),
    "google_calendar": ("calendar_events",),
    "google_analytics": ("analytics_traffic_report",),
    "youtube": ("youtube_channel_stats",),
    "coinbase": ("coinbase_accounts", "coinbase_transactions"),
}

_COINBASE_HEADERS = {"CB-VERSION": "2024-10-01"}


def _period(since: str | None, until: str | None, default_days: int = 30) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    end = _parse(until) or now
    start = _parse(since) or (end - timedelta(days=default_days))
    return start, end


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def _bearer(context: Any, store: Any, record: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    from src.anubis.utils.connected_accounts.oauth_flow import (
        OAuthFlowError,
        OAuthReconnectRequired,
        get_fresh_access_token,
    )

    try:
        return await get_fresh_access_token(context, store, str(record.get("user_id") or ""), record), None
    except OAuthReconnectRequired:
        return None, {
            "status": "needs_reconnect",
            "connection": record.get("display_label"),
            "error": f"{record.get('display_label')} needs to be signed in again (connect_account with provider {record.get('provider')}).",
        }
    except OAuthFlowError as flow_error:
        return None, {"status": "error", "error": flow_error.detail}


async def _get_json(url: str, token: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: float = 20.0) -> tuple[int, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json", **(headers or {})},
        )
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {"text": response.text[:2000]}


async def _post_json(url: str, token: str, body: dict[str, Any], *, headers: dict[str, str] | None = None, timeout: float = 20.0) -> tuple[int, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json", **(headers or {})},
        )
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {"text": response.text[:2000]}


def _selector(records: list[dict[str, Any]], provider_name: str):
    candidates = [record for record in records if record.get("provider") == provider_name]

    def _select(connection: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not candidates:
            return None, {"status": "not_connected", "error": f"No {provider_name} account is connected."}
        if connection is None or not str(connection).strip():
            return candidates[0], None
        wanted = str(connection).strip().lower()
        for record in candidates:
            if wanted in (str(record.get("display_label") or "").lower(), str(record.get("account_address") or "").lower()):
                return record, None
        return None, {"status": "unknown_connection", "error": f"No {provider_name} account named {connection!r}."}

    return _select


def _github_headers() -> dict[str, str]:
    return {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


async def _github_paginate(url: str, token: str, params: dict[str, Any], *, max_pages: int = 5) -> list[Any]:
    import httpx

    rows: list[Any] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        next_url: str | None = url
        next_params: dict[str, Any] | None = {**params, "per_page": 100}
        for _ in range(max_pages):
            if not next_url:
                break
            response = await client.get(next_url, params=next_params, headers={"Authorization": f"Bearer {token}", **_github_headers()})
            if response.status_code >= 400:
                break
            document = response.json()
            if isinstance(document, list):
                rows.extend(document)
            else:
                rows.append(document)
                break
            next_url = response.links.get("next", {}).get("url") if hasattr(response, "links") else None
            next_params = None
    return rows


def build_vendor_api_tools(context: Any, accounts: list[dict[str, Any]], *, store: Any = None, pool: Any = None) -> list[Any]:
    """Build the API-backed tools for every OAuth vendor account present."""
    providers_present = {str(record.get("provider") or "") for record in accounts}
    tools: list[Any] = []

    if "github" in providers_present:
        select_github = _selector(accounts, "github")

        @tool
        async def github_activity(repository: str, since: str | None = None, until: str | None = None, connection: str | None = None) -> dict[str, Any]:
            """List commits of a GitHub repository (owner/name) in a period, newest first.

            Use for "what happened in <repo> since <date>", sprint summaries, and
            per-feature timelines. Defaults to the last 30 days.
            """
            record, error = select_github(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            start, end = _period(since, until)
            rows = await _github_paginate(
                f"https://api.github.com/repos/{repository}/commits", token,
                {"since": start.isoformat(), "until": end.isoformat()},
            )
            commits = [
                {
                    "sha": entry.get("sha"),
                    "author": ((entry.get("commit") or {}).get("author") or {}).get("name"),
                    "authored_at": ((entry.get("commit") or {}).get("author") or {}).get("date"),
                    "subject": str((entry.get("commit") or {}).get("message") or "").split("\n", 1)[0][:200],
                    "url": entry.get("html_url"),
                }
                for entry in rows
                if isinstance(entry, dict)
            ]
            return {"status": "ok", "repository": repository, "since": start.isoformat(), "until": end.isoformat(), "commit_count": len(commits), "commits": commits[:300]}

        @tool
        async def github_issues(repository: str, state: str = "open", since: str | None = None, labels: str | None = None, connection: str | None = None) -> dict[str, Any]:
            """List issues of a GitHub repository (feature requests, bugs) by state.

            Use for "are there feature requests", "what do users ask for", and
            "what is upcoming". ``state`` is open, closed, or all.
            """
            record, error = select_github(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            params: dict[str, Any] = {"state": state or "open"}
            if since:
                params["since"] = since
            if labels:
                params["labels"] = labels
            rows = await _github_paginate(f"https://api.github.com/repos/{repository}/issues", token, params, max_pages=3)
            issues = [
                {"number": entry.get("number"), "title": entry.get("title"), "state": entry.get("state"), "labels": [label.get("name") for label in entry.get("labels") or [] if isinstance(label, dict)], "created_at": entry.get("created_at"), "closed_at": entry.get("closed_at"), "url": entry.get("html_url"), "is_pull_request": "pull_request" in entry}
                for entry in rows
                if isinstance(entry, dict)
            ]
            return {"status": "ok", "repository": repository, "issue_count": len([issue for issue in issues if not issue["is_pull_request"]]), "issues": [issue for issue in issues if not issue["is_pull_request"]][:200]}

        @tool
        async def github_pull_requests(repository: str, state: str = "open", connection: str | None = None) -> dict[str, Any]:
            """List pull requests of a GitHub repository by state (open, closed, all)."""
            record, error = select_github(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            rows = await _github_paginate(f"https://api.github.com/repos/{repository}/pulls", token, {"state": state or "open", "sort": "updated", "direction": "desc"}, max_pages=3)
            pulls = [
                {"number": entry.get("number"), "title": entry.get("title"), "state": entry.get("state"), "draft": entry.get("draft"), "created_at": entry.get("created_at"), "merged_at": entry.get("merged_at"), "closed_at": entry.get("closed_at"), "author": (entry.get("user") or {}).get("login"), "url": entry.get("html_url")}
                for entry in rows
                if isinstance(entry, dict)
            ]
            return {"status": "ok", "repository": repository, "pull_request_count": len(pulls), "pull_requests": pulls[:200]}

        tools.extend([github_activity, github_issues, github_pull_requests])

    if "twitter" in providers_present:
        select_x = _selector(accounts, "twitter")

        @tool
        async def x_recent_posts(max_results: int = 20, connection: str | None = None) -> dict[str, Any]:
            """List the owner's recent posts on X (the connected account's own timeline)."""
            record, error = select_x(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            status_code, me = await _get_json("https://api.x.com/2/users/me", token)
            user_id = ((me or {}).get("data") or {}).get("id") if isinstance(me, dict) else None
            if not user_id:
                return {"status": "error", "status_code": status_code, "error": "X did not identify the signed-in account."}
            status_code, document = await _get_json(
                f"https://api.x.com/2/users/{user_id}/tweets", token,
                params={"max_results": max(5, min(int(max_results or 20), 100)), "tweet.fields": "created_at,public_metrics"},
            )
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            return {"status": "ok", "posts": (document or {}).get("data") or []}

        @tool
        async def x_post_reply(text: str, in_reply_to_post_id: str | None = None, connection: str | None = None) -> dict[str, Any]:
            """Post on X as the owner (a new post, or a reply when in_reply_to_post_id is given).

            Only when the owner asked for exactly this post in this conversation.
            """
            record, error = select_x(connection)
            if error:
                return error
            if not str(text or "").strip():
                return {"status": "error", "error": "Post text is required."}
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            body: dict[str, Any] = {"text": str(text)[:280]}
            if in_reply_to_post_id:
                body["reply"] = {"in_reply_to_tweet_id": str(in_reply_to_post_id)}
            status_code, document = await _post_json("https://api.x.com/2/tweets", token, body)
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            return {"status": "ok", "post": (document or {}).get("data")}

        tools.extend([x_recent_posts, x_post_reply])

    if "vercel" in providers_present:
        select_vercel = _selector(accounts, "vercel")

        @tool
        async def vercel_deployments(project: str | None = None, limit: int = 20, connection: str | None = None) -> dict[str, Any]:
            """List recent Vercel deployments (optionally for one project)."""
            record, error = select_vercel(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            params: dict[str, Any] = {"limit": max(1, min(int(limit or 20), 100))}
            if project:
                params["projectId"] = project
            status_code, document = await _get_json("https://api.vercel.com/v6/deployments", token, params=params)
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            deployments = [
                {"uid": entry.get("uid"), "name": entry.get("name"), "url": entry.get("url"), "state": entry.get("state") or entry.get("readyState"), "created_at": entry.get("created"), "target": entry.get("target")}
                for entry in (document or {}).get("deployments") or []
            ]
            return {"status": "ok", "deployments": deployments}

        @tool
        async def vercel_usage(connection: str | None = None) -> dict[str, Any]:
            """Read the Vercel account's projects and team as a usage overview."""
            record, error = select_vercel(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            status_code, projects = await _get_json("https://api.vercel.com/v9/projects", token, params={"limit": 50})
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(projects)[:500]}
            return {"status": "ok", "projects": [{"id": entry.get("id"), "name": entry.get("name"), "framework": entry.get("framework"), "updated_at": entry.get("updatedAt")} for entry in (projects or {}).get("projects") or []]}

        tools.extend([vercel_deployments, vercel_usage])

    if "google_calendar" in providers_present:
        select_calendar = _selector(accounts, "google_calendar")

        @tool
        async def calendar_events(since: str | None = None, until: str | None = None, max_results: int = 50, connection: str | None = None) -> dict[str, Any]:
            """List the owner's Google Calendar events in a period (default the next 7 days)."""
            record, error = select_calendar(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            now = datetime.now(UTC)
            start = _parse(since) or now
            end = _parse(until) or (start + timedelta(days=7))
            status_code, document = await _get_json(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events", token,
                params={"timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true", "orderBy": "startTime", "maxResults": max(1, min(int(max_results or 50), 250))},
            )
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            events = [
                {"summary": entry.get("summary"), "start": (entry.get("start") or {}).get("dateTime") or (entry.get("start") or {}).get("date"), "end": (entry.get("end") or {}).get("dateTime") or (entry.get("end") or {}).get("date"), "location": entry.get("location"), "attendees": len(entry.get("attendees") or [])}
                for entry in (document or {}).get("items") or []
            ]
            return {"status": "ok", "events": events}

        tools.append(calendar_events)

    if "google_analytics" in providers_present:
        select_analytics = _selector(accounts, "google_analytics")

        @tool
        async def analytics_traffic_report(property_id: str | None = None, since: str | None = None, until: str | None = None, connection: str | None = None) -> dict[str, Any]:
            """Report sessions, users, and top pages from Google Analytics for a period.

            ``property_id`` is the GA4 property (numeric); omitted → the first
            property the account can see. Default period the last 30 days.
            """
            record, error = select_analytics(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            return await analytics_report(token, property_id=property_id, since=since, until=until)

        tools.append(analytics_traffic_report)

    if "coinbase" in providers_present:
        select_coinbase = _selector(accounts, "coinbase")

        @tool
        async def coinbase_accounts(connection: str | None = None) -> dict[str, Any]:
            """List the owner's Coinbase accounts with their balances (read-only).

            Use for "what is my crypto balance" or "what do I hold on Coinbase".
            """
            record, error = select_coinbase(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            status_code, document = await _get_json(
                "https://api.coinbase.com/v2/accounts", token, headers=_COINBASE_HEADERS, params={"limit": 100}
            )
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            accounts_out = []
            for entry in (document or {}).get("data") or []:
                if not isinstance(entry, dict):
                    continue
                currency = entry.get("currency")
                accounts_out.append({
                    "id": entry.get("id"),
                    "name": entry.get("name"),
                    "currency": currency.get("code") if isinstance(currency, dict) else currency,
                    "balance": (entry.get("balance") or {}).get("amount"),
                    "native_balance": (entry.get("native_balance") or {}).get("amount"),
                    "type": entry.get("type"),
                })
            return {"status": "ok", "accounts": accounts_out}

        @tool
        async def coinbase_transactions(account_id: str, limit: int = 25, connection: str | None = None) -> dict[str, Any]:
            """List recent transactions for one Coinbase account (read-only).

            ``account_id`` comes from coinbase_accounts. Use for "my recent
            crypto transactions".
            """
            record, error = select_coinbase(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            status_code, document = await _get_json(
                f"https://api.coinbase.com/v2/accounts/{account_id}/transactions", token,
                headers=_COINBASE_HEADERS, params={"limit": max(1, min(int(limit or 25), 100))},
            )
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            transactions = []
            for entry in (document or {}).get("data") or []:
                if not isinstance(entry, dict):
                    continue
                amount = entry.get("amount") or {}
                transactions.append({
                    "type": entry.get("type"),
                    "status": entry.get("status"),
                    "amount": amount.get("amount"),
                    "currency": amount.get("currency"),
                    "native_amount": (entry.get("native_amount") or {}).get("amount"),
                    "description": entry.get("description") or (entry.get("details") or {}).get("title"),
                    "created_at": entry.get("created_at"),
                })
            return {"status": "ok", "account_id": account_id, "transactions": transactions}

        tools.extend([coinbase_accounts, coinbase_transactions])

    if "youtube" in providers_present:
        select_youtube = _selector(accounts, "youtube")

        @tool
        async def youtube_channel_stats(connection: str | None = None) -> dict[str, Any]:
            """Read the owner's YouTube channel statistics (subscribers, views, videos)."""
            record, error = select_youtube(connection)
            if error:
                return error
            token, failure = await _bearer(context, store, record)
            if failure:
                return failure
            status_code, document = await _get_json("https://www.googleapis.com/youtube/v3/channels", token, params={"part": "snippet,statistics", "mine": "true"})
            if status_code >= 400:
                return {"status": "error", "status_code": status_code, "error": str(document)[:500]}
            channels = [
                {"title": (entry.get("snippet") or {}).get("title"), "statistics": entry.get("statistics")}
                for entry in (document or {}).get("items") or []
            ]
            return {"status": "ok", "channels": channels}

        tools.append(youtube_channel_stats)

    return tools


async def analytics_report(token: str, *, property_id: str | None, since: str | None, until: str | None, hostname: str | None = None) -> dict[str, Any]:
    """Run a GA4 sessions/users/pages report; pick the first property when none is given."""
    start, end = _period(since, until)
    chosen = property_id
    if not chosen:
        status_code, summaries = await _get_json("https://analyticsadmin.googleapis.com/v1beta/accountSummaries", token)
        if status_code >= 400:
            return {"status": "error", "status_code": status_code, "error": str(summaries)[:500]}
        for account in (summaries or {}).get("accountSummaries") or []:
            for property_summary in account.get("propertySummaries") or []:
                chosen = str(property_summary.get("property") or "").replace("properties/", "")
                if chosen:
                    break
            if chosen:
                break
    if not chosen:
        return {"status": "error", "error": "The Google Analytics account has no properties."}
    body: dict[str, Any] = {
        "dateRanges": [{"startDate": start.date().isoformat(), "endDate": end.date().isoformat()}],
        "dimensions": [{"name": "date"}],
        "metrics": [{"name": "sessions"}, {"name": "activeUsers"}, {"name": "screenPageViews"}],
    }
    if hostname:
        body["dimensionFilter"] = {"filter": {"fieldName": "hostName", "stringFilter": {"matchType": "CONTAINS", "value": hostname}}}
    status_code, daily = await _post_json(f"https://analyticsdata.googleapis.com/v1beta/properties/{chosen}:runReport", token, body)
    if status_code >= 400:
        return {"status": "error", "status_code": status_code, "error": str(daily)[:500]}
    pages_body = {**body, "dimensions": [{"name": "pagePath"}], "metrics": [{"name": "screenPageViews"}], "limit": 20, "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}]}
    _status, pages = await _post_json(f"https://analyticsdata.googleapis.com/v1beta/properties/{chosen}:runReport", token, pages_body)

    def _rows(document: Any) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for row in (document or {}).get("rows") or []:
            rows.append([value.get("value") for value in row.get("dimensionValues") or []] + [value.get("value") for value in row.get("metricValues") or []])
        return rows

    return {
        "status": "ok",
        "property_id": chosen,
        "since": start.date().isoformat(),
        "until": end.date().isoformat(),
        "daily": {"columns": ["date", "sessions", "active_users", "page_views"], "rows": sorted(_rows(daily))},
        "top_pages": {"columns": ["page", "page_views"], "rows": _rows(pages)},
    }


async def traffic_report_for_site(context: Any, store: Any, accounts: list[dict[str, Any]], site_url: str, *, since: str | None, until: str | None) -> dict[str, Any]:
    """Traffic for one site from the first Google Analytics (or Vercel) account."""
    from urllib.parse import urlparse

    hostname = (urlparse(site_url).hostname or "").lower()
    for record in accounts:
        if record.get("provider") != "google_analytics":
            continue
        token, failure = await _bearer(context, store, record)
        if failure:
            return failure
        return await analytics_report(token, property_id=None, since=since, until=until, hostname=hostname)
    return {"status": "no_traffic_source", "message": "Only Google Analytics traffic is supported today; connect Google Analytics for this site."}
