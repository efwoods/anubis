"""The chat tools that let the owner ask business, finance, and vendor questions.

Built per turn for the personal avatar only. Every tool docstring names the
owner question the tool answers, because the docstring is what the model
reads when choosing a tool. Dates are ISO strings; the period defaults to the
last thirty days; a missing pool makes every tool answer
``{"status": "unavailable"}`` rather than raise, so a dev server without
Postgres still runs the conversation.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

ANALYTICS_TOOL_NAMES: tuple[str, ...] = (
    "query_platform_metrics",
    "query_finances",
    "query_vendor_usage",
    "make_chart",
    "save_report",
    "list_reports",
    "schedule_report",
    "cancel_report_schedule",
    "list_report_schedules",
    "forecast_metric",
)

PLATFORM_METRIC_NAMES: tuple[str, ...] = (
    "messages_per_day",
    "messages_per_user_per_day",
    "average_conversation_length",
    "avatars_by_conversation_count",
    "feature_usage_per_avatar",
    "first_seen_users_per_week",
    "active_users",
    "spend_by_period",
    "feedback_summary",
    "revenue_estimate",
)

FINANCE_METRIC_NAMES: tuple[str, ...] = (
    "spend",
    "by_category",
    "by_merchant",
    "by_day",
    "cac",
    "accounts",
)

STATUS_UNAVAILABLE = "unavailable"
STATUS_FORBIDDEN = "forbidden"
STATUS_ERROR = "error"
STATUS_OK = "ok"

DEFAULT_PERIOD_DAYS = 30
BANK_KIND = "bank"
BANK_PROVIDER = "plaid"


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO date or datetime string into an aware UTC datetime."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def period_from(since: str | None, until: str | None) -> tuple[datetime, datetime]:
    """Return the period as datetimes, defaulting to the last thirty days."""
    end = parse_iso_datetime(until) or datetime.now(UTC)
    start = parse_iso_datetime(since) or end - timedelta(days=DEFAULT_PERIOD_DAYS)
    return start, end


def _unavailable(what: str) -> dict[str, Any]:
    """Return the answer every tool gives when the database is not reachable."""
    return {
        "status": STATUS_UNAVAILABLE,
        "message": f"{what} is unavailable: the analytics database is not connected.",
    }


def _error(error: Exception) -> dict[str, Any]:
    """Return an error answer the model can relay plainly."""
    return {"status": STATUS_ERROR, "message": str(error)[:500]}


def build_analytics_tools(
    context: Any,
    *,
    store: Any,
    pool: Any,
    user_id: str,
    assistant_id: str,
    connected_accounts: list[dict[str, Any]] | None,
    analysis_bundle: Any = None,
    thread_id: str | None = None,
    timezone_name: str | None = None,
) -> list[Any]:
    """Build the analytics tools bound to the owner's personal avatar."""
    from src.anubis.utils.analytics.charts import (
        ChartSpec,
        TurnChartCollector,
        chart_artifact_name,
        render_chart_png,
    )
    from src.anubis.utils.analytics.forecast import forecast_series
    from src.anubis.utils.analytics.reports import (
        get_report_repository,
        normalise_report_kind,
        public_report_view,
    )
    from src.anubis.utils.analytics.schedules import (
        first_run_time,
        get_schedule_repository,
        normalise_interval,
        public_schedule_view,
    )

    accounts = list(connected_accounts or [])

    def _bank_records(connection_label: str | None) -> list[dict[str, Any]]:
        """Return the finance connections, narrowed to one label when given."""
        bank_records = [
            record
            for record in accounts
            if str(record.get("kind") or "") == BANK_KIND
            or str(record.get("provider") or "") == BANK_PROVIDER
        ]
        if connection_label:
            needle = str(connection_label).strip().lower()
            bank_records = [
                record
                for record in bank_records
                if needle
                in str(
                    record.get("display_label") or record.get("account_key") or ""
                ).lower()
            ]
        return bank_records

    @tool
    async def query_platform_metrics(
        metric: str,
        since: str | None = None,
        until: str | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Answer the owner's questions about the Neural Nexus platform's users, conversations, features, feedback, spend, and revenue.

        Owner questions and the metric that answers each:
        - how often users send messages on average: "messages_per_user_per_day"
          (total traffic per day: "messages_per_day")
        - average conversation length: "average_conversation_length"
        - which avatars users speak to most: "avatars_by_conversation_count"
        - features used most and least by each user's personal avatar:
          "feature_usage_per_avatar"
        - how many users are active and how fast the user base grows:
          "active_users", "first_seen_users_per_week"
        - platform model spend in a period: "spend_by_period"
          (group_by "day", "model", or "inference_type")
        - what users love, hate, dislike, and request: "feedback_summary"
        - projected revenue: "revenue_estimate" (monthly recurring revenue by tier)

        Platform-wide numbers are reserved for the platform administrator with
        the Neural Nexus business account connected; anyone else receives
        status "forbidden". Administrator traffic is excluded from the
        platform's metrics by design.

        Args:
            metric: One of the metric names above.
            since: ISO date or datetime the period starts at (default: thirty days ago).
            until: ISO date or datetime the period ends at (default: now).
            group_by: For "spend_by_period": "day", "model", or "inference_type".
        """
        from src.anubis.utils.analytics import platform_metrics

        metric_name = str(metric or "").strip().lower()
        if metric_name not in PLATFORM_METRIC_NAMES:
            return {
                "status": STATUS_ERROR,
                "message": f"Unknown metric '{metric}'.",
                "metrics": list(PLATFORM_METRIC_NAMES),
            }
        if not platform_metrics.is_platform_admin(context, accounts, user_id):
            return {
                "status": STATUS_FORBIDDEN,
                "message": (
                    "Platform-wide metrics are reserved for the platform administrator "
                    "with the Neural Nexus business account connected."
                ),
            }
        try:
            start, end = period_from(since, until)
        except ValueError as parse_error:
            return _error(parse_error)
        try:
            if metric_name == "revenue_estimate":
                result = await platform_metrics.revenue_estimate(
                    context, SimpleNamespace(context=context)
                )
                return {"status": STATUS_OK, "metric": metric_name, **result}
            if metric_name == "feedback_summary":
                if pool is None:
                    if store is None:
                        return _unavailable("Feedback")
                    result = await platform_metrics.feedback_summary(
                        store, user_id=user_id, assistant_id=assistant_id
                    )
                else:
                    result = await platform_metrics.feedback_summary_all(pool)
                return {"status": STATUS_OK, "metric": metric_name, **result}
            if pool is None:
                return _unavailable("Platform metrics")
            if metric_name == "spend_by_period":
                result = await platform_metrics.spend_by_period(
                    pool, start, end, group_by=group_by or "day"
                )
            else:
                query = getattr(platform_metrics, metric_name)
                result = await query(pool, start, end)
            return {
                "status": STATUS_OK,
                "metric": metric_name,
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                **result,
            }
        except Exception as query_error:  # noqa: BLE001 - relay, never raise
            logger.exception("Platform metric %s failed", metric_name)
            return _error(query_error)

    @tool
    async def query_finances(
        metric: str,
        since: str | None = None,
        until: str | None = None,
        connection_label: str | None = None,
    ) -> dict[str, Any]:
        """Answer the owner's spending questions from the connected bank and card accounts.

        Owner questions and the metric that answers each:
        - how much was spent in a period: "spend" (total plus breakdown by category)
        - spend by category: "by_category"; by merchant: "by_merchant"; per day
          (the series to chart and to forecast burn rate): "by_day"
        - what a new user costs to acquire: "cac" (advertising spend divided by
          the platform's new users in the period)
        - which accounts are linked: "accounts"

        Transactions are synced from the bank first when the stored copy is
        older than the configured minimum interval. When no bank is connected
        the answer says so; offer connect_account with provider "plaid".

        Args:
            metric: One of spend, by_category, by_merchant, by_day, cac, accounts.
            since: ISO date the period starts at (default: thirty days ago).
            until: ISO date the period ends at (default: today).
            connection_label: Narrow to one connected institution by label.
        """
        from src.anubis.utils.analytics import finance

        metric_name = str(metric or "").strip().lower()
        if metric_name not in FINANCE_METRIC_NAMES:
            return {
                "status": STATUS_ERROR,
                "message": f"Unknown metric '{metric}'.",
                "metrics": list(FINANCE_METRIC_NAMES),
            }
        bank_records = _bank_records(connection_label)
        if not bank_records:
            return {
                "status": STATUS_UNAVAILABLE,
                "message": (
                    "No bank or card account is connected. Offer connect_account "
                    "with provider 'plaid'."
                ),
                "missing_connection": BANK_PROVIDER,
            }
        if metric_name == "accounts":
            listed = []
            for record in bank_records:
                listed.extend(await finance.accounts_for_record(record))
            return {"status": STATUS_OK, "accounts": listed}
        if pool is None:
            return _unavailable("Finance")
        try:
            start, end = period_from(since, until)
        except ValueError as parse_error:
            return _error(parse_error)
        minimum_interval = int(
            getattr(context, "finance_sync_min_interval_minutes", None) or 360
        )
        sync_notes: list[dict[str, Any]] = []
        for record in bank_records:
            try:
                cursor_row = await finance.read_sync_cursor(pool, user_id, record)
                if finance.should_sync(cursor_row, minimum_interval):
                    sync_result = await finance.sync_transactions(
                        context, pool, user_id, record
                    )
                    sync_notes.append(
                        {"connection": record.get("display_label"), **sync_result}
                    )
            except Exception as sync_error:  # noqa: BLE001 - stored rows still answer
                logger.warning("Finance sync failed: %s", sync_error)
                sync_notes.append(
                    {
                        "connection": record.get("display_label"),
                        "error": str(sync_error)[:300],
                    }
                )
        try:
            if metric_name == "cac":
                advertising = await finance.advertising_spend(pool, user_id, start, end)
                new_users = 0
                try:
                    from src.anubis.utils.analytics import platform_metrics

                    growth = await platform_metrics.first_seen_users_per_week(
                        pool, start, end
                    )
                    new_users = int(sum(int(row[1] or 0) for row in growth["rows"]))
                except Exception:  # noqa: BLE001 - the platform table may be absent
                    logger.debug("New-user count unavailable for CAC", exc_info=True)
                result: dict[str, Any] = {
                    **finance.customer_acquisition_cost(
                        advertising["advertising_spend_usd"], new_users
                    ),
                    "advertising": advertising,
                }
            else:
                grouping = {
                    "spend": "category",
                    "by_category": "category",
                    "by_merchant": "merchant",
                    "by_day": "day",
                }[metric_name]
                result = await finance.spend_by_period(
                    pool, user_id, start, end, group_by=grouping
                )
            return {
                "status": STATUS_OK,
                "metric": metric_name,
                "synced": sync_notes,
                **result,
            }
        except Exception as query_error:  # noqa: BLE001 - relay, never raise
            logger.exception("Finance metric %s failed", metric_name)
            return _error(query_error)

    @tool
    async def query_vendor_usage(
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Answer "how much did we use and spend on our vendors" (LangSmith, OpenAI, Anthropic, and other connected usage sources).

        Returns the daily rows for one provider when ``provider`` is given, and
        the totals by provider, metric, and unit otherwise. When no vendor
        is connected the answer says so; offer connect_account with the
        vendor's provider name.

        Args:
            provider: A provider name such as "openai", "anthropic", or "langsmith".
            since: ISO date the period starts at (default: thirty days ago).
            until: ISO date the period ends at (default: today).
        """
        from src.anubis.utils.analytics import vendor_usage

        if pool is None:
            return _unavailable("Vendor usage")
        try:
            start, end = period_from(since, until)
        except ValueError as parse_error:
            return _error(parse_error)
        try:
            totals = await vendor_usage.usage_totals(pool, user_id, start, end)
            answer: dict[str, Any] = {"status": STATUS_OK, "totals": totals}
            if provider:
                answer["daily"] = await vendor_usage.usage_by_period(
                    pool, user_id, str(provider).strip().lower(), start, end
                )
            if not totals["rows"]:
                answer["message"] = (
                    "No vendor usage is recorded for the period. Connect a vendor "
                    "(langsmith, openai, anthropic) with connect_account, or the "
                    "connected vendor has not been read yet."
                )
            return answer
        except Exception as query_error:  # noqa: BLE001 - relay, never raise
            logger.exception("Vendor usage query failed")
            return _error(query_error)

    @tool
    async def make_chart(spec: dict[str, Any]) -> dict[str, Any]:
        """Draw one chart from numbers a tool returned, so the owner sees the trend rather than a table.

        Use this for every time series and every ranked breakdown; never draw
        a chart with code. ``spec`` holds: "type" (line, bar, area, pie,
        stacked_bar), "title", "x" ({"label", "values": [...]}), "series"
        ([{"name", "values": [...], "unit"}]), and optional "unit" and
        "notes". Every series must hold one value per x value. The chart is
        rendered to a PNG, saved as a created artifact, and attached to the
        reply; the returned chart_id can be passed to save_report.

        Args:
            spec: The chart specification described above.
        """
        try:
            chart_spec = ChartSpec.model_validate(spec or {})
        except Exception as validation_error:  # noqa: BLE001 - relay the message
            return _error(validation_error)
        try:
            png_bytes = render_chart_png(chart_spec)
        except Exception as render_error:  # noqa: BLE001
            logger.exception("Chart rendering failed")
            return _error(render_error)
        png_artifact_name = chart_artifact_name(chart_spec)
        persisted_name = png_artifact_name
        try:
            persisted_name = await _persist_chart_png(
                png_bytes,
                png_artifact_name,
                store=store,
                analysis_bundle=analysis_bundle,
                user_id=user_id,
                assistant_id=assistant_id,
            )
        except Exception:  # noqa: BLE001 - the chart still reaches the reply
            logger.exception("Could not persist chart %s", png_artifact_name)
        spec_dict = chart_spec.model_dump()
        TurnChartCollector.add(spec_dict, persisted_name)
        return {
            "status": STATUS_OK,
            "chart_id": chart_spec.chart_id,
            "png_artifact_name": persisted_name,
        }

    @tool
    async def save_report(
        kind: str,
        title: str,
        summary_markdown: str,
        chart_ids: list[str] | None = None,
        sources: list[str] | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> dict[str, Any]:
        """Save an answer that covers a period as a report the owner can find again.

        Call this whenever an answer covers a period (a week, a month, a
        quarter, "since a date"): a sprint digest, a spend digest, a usage
        summary, a forecast. The charts made this turn are attached (those
        whose chart_id is listed, or all of them when chart_ids is omitted).

        Args:
            kind: One of platform_usage, finance, vendor_usage, development,
                website, sprint_digest, spend_digest, custom.
            title: A short title naming the subject and the period.
            summary_markdown: The report body in Markdown, with the numbers.
            chart_ids: chart_id values from make_chart to attach (default: all).
            sources: Where the numbers came from (tool names, connections).
            period_start: ISO date the report's period starts at.
            period_end: ISO date the report's period ends at.
        """
        repository = get_report_repository()
        if repository is None:
            return _unavailable("Saving reports")
        collected = TurnChartCollector.collect()
        if chart_ids:
            wanted = {str(chart_id) for chart_id in chart_ids}
            charts = [
                chart for chart in collected if str(chart.get("chart_id")) in wanted
            ]
        else:
            charts = collected
        try:
            row = await repository.create(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "kind": normalise_report_kind(kind),
                    "title": str(title or "").strip() or "Untitled report",
                    "summary_markdown": str(summary_markdown or ""),
                    "charts": charts,
                    "sources": [str(source) for source in (sources or [])],
                    "period_start": parse_iso_datetime(period_start),
                    "period_end": parse_iso_datetime(period_end),
                    "thread_id": thread_id,
                }
            )
        except Exception as save_error:  # noqa: BLE001
            logger.exception("Could not save report")
            return _error(save_error)
        view = public_report_view(row)
        return {
            "status": STATUS_OK,
            "report_id": view["report_id"],
            "title": view["title"],
            "kind": view["kind"],
            "chart_count": len(charts),
            "created_at": view["created_at"],
        }

    @tool
    async def list_reports(
        query: str | None = None,
        kind: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find reports saved earlier, answering "what did we say about X last month" and "show me the last sprint digest".

        Args:
            query: Words to search for in report titles and summaries.
            kind: Narrow to one report kind (for example sprint_digest).
            since: ISO date; only reports created at or after this moment.
            until: ISO date; only reports created at or before this moment.
            limit: Maximum reports to return.
        """
        repository = get_report_repository()
        if repository is None:
            return _unavailable("Listing reports")
        try:
            rows = await repository.list(
                user_id,
                assistant_id=assistant_id,
                query=query,
                kind=str(kind).strip().lower() if kind else None,
                since=parse_iso_datetime(since),
                until=parse_iso_datetime(until),
                limit=max(1, min(int(limit or 10), 50)),
            )
        except Exception as list_error:  # noqa: BLE001
            return _error(list_error)
        reports = []
        for row in rows:
            view = public_report_view(row)
            view["summary_markdown"] = view["summary_markdown"][:1200]
            reports.append(view)
        return {"status": STATUS_OK, "reports": reports}

    @tool
    async def schedule_report(
        kind: str, title: str, question: str, interval: str
    ) -> dict[str, Any]:
        """Run a question on a cadence and deliver the report to the owner's inbox, for recurring insight such as a weekly sprint digest or a monthly spend digest.

        The first run is the next Monday at 09:00 in the owner's time zone for
        weekly, the first of next month at 09:00 for monthly, and tomorrow at
        09:00 for daily.

        Args:
            kind: The report kind (sprint_digest, spend_digest, finance,
                platform_usage, vendor_usage, development, website, custom).
            title: A short title for the recurring report.
            question: The full question the avatar should answer each time,
                including "chart" and "save the report" instructions.
            interval: daily, weekly, or monthly.
        """
        repository = get_schedule_repository()
        if repository is None:
            return _unavailable("Scheduling reports")
        cleaned_question = str(question or "").strip()
        if not cleaned_question:
            return {"status": STATUS_ERROR, "message": "A question is required."}
        normalised_interval = normalise_interval(interval)
        try:
            row = await repository.create(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "kind": normalise_report_kind(kind),
                    "title": str(title or "").strip() or "Scheduled report",
                    "question": cleaned_question,
                    "interval": normalised_interval,
                    "next_run_at": first_run_time(
                        normalised_interval, datetime.now(UTC), timezone_name
                    ),
                }
            )
        except Exception as create_error:  # noqa: BLE001
            return _error(create_error)
        return {"status": STATUS_OK, **public_schedule_view(row)}

    @tool
    async def cancel_report_schedule(schedule_id: str) -> dict[str, Any]:
        """Stop a recurring report the owner no longer wants.

        Args:
            schedule_id: The schedule's id from list_report_schedules.
        """
        repository = get_schedule_repository()
        if repository is None:
            return _unavailable("Cancelling schedules")
        try:
            disabled = await repository.disable(user_id, str(schedule_id))
        except Exception as disable_error:  # noqa: BLE001
            return _error(disable_error)
        return {
            "status": STATUS_OK if disabled else STATUS_ERROR,
            "schedule_id": str(schedule_id),
            "message": (
                "Schedule cancelled." if disabled else "No enabled schedule with that id."
            ),
        }

    @tool
    async def list_report_schedules() -> dict[str, Any]:
        """List the recurring reports that run for the owner, answering "what reports do I get automatically".

        Returns every schedule with the next run time and whether the
        schedule is enabled.
        """
        repository = get_schedule_repository()
        if repository is None:
            return _unavailable("Listing schedules")
        try:
            rows = await repository.list_for_avatar(user_id, assistant_id)
        except Exception as list_error:  # noqa: BLE001
            return _error(list_error)
        return {
            "status": STATUS_OK,
            "schedules": [public_schedule_view(row) for row in rows],
        }

    @tool
    async def forecast_metric(
        values: list[float], horizon: int, season_length: int | None = None
    ) -> dict[str, Any]:
        """Project a series forward, answering "what does next month or next quarter look like" for revenue, spend, burn rate, or users.

        Pass the observed series (one number per period, oldest first) and how
        many periods ahead to project; pass season_length (7 for weekly
        patterns in daily data, 12 for yearly patterns in monthly data) when
        the series repeats. The answer names the method; say so when
        relaying the projection, and chart the observed and projected
        values together with make_chart.

        Args:
            values: The observed numbers, oldest first (at least three).
            horizon: How many periods ahead to project.
            season_length: The length of a repeating season, when known.
        """
        try:
            result = forecast_series(
                [float(value) for value in values or []],
                int(horizon or 1),
                int(season_length) if season_length else None,
            )
        except Exception as forecast_error:  # noqa: BLE001
            return _error(forecast_error)
        if "error" in result:
            return {"status": STATUS_ERROR, "message": result["error"]}
        return {"status": STATUS_OK, **result}

    return [
        query_platform_metrics,
        query_finances,
        query_vendor_usage,
        make_chart,
        save_report,
        list_reports,
        schedule_report,
        cancel_report_schedule,
        list_report_schedules,
        forecast_metric,
    ]


async def _persist_chart_png(
    png_bytes: bytes,
    png_artifact_name: str,
    *,
    store: Any,
    analysis_bundle: Any,
    user_id: str,
    assistant_id: str,
) -> str:
    """Save the rendered chart as a created artifact; return the saved name.

    With an analysis bundle the PNG is written into the turn's workspace and
    persisted through ``persist_workspace_file`` (so the artifact shows in the
    reply alongside other created files); without one the same record shape
    is written straight into the created-artifact store namespace.
    """
    if analysis_bundle is not None and getattr(analysis_bundle, "workspace_path", None):
        from src.anubis.utils.tools.data_analysis.analysis_tools import (
            persist_workspace_file,
        )

        workspace_path = analysis_bundle.workspace_path
        workspace_path.mkdir(parents=True, exist_ok=True)
        candidate_path = workspace_path / png_artifact_name
        candidate_path.write_bytes(png_bytes)
        record = await persist_workspace_file(
            analysis_bundle, store or analysis_bundle.store, candidate_path
        )
        return str(record.get("name") or png_artifact_name)
    if store is None:
        return png_artifact_name
    from src.anubis.utils.tools.data_analysis.backend import created_namespace

    saved_at = datetime.now(UTC).isoformat()
    await store.aput(
        created_namespace(user_id, assistant_id),
        key=f"/{png_artifact_name}",
        value={
            "content": base64.standard_b64encode(png_bytes).decode("ascii"),
            "encoding": "base64",
            "created_at": saved_at,
            "modified_at": saved_at,
            "size_bytes": len(png_bytes),
        },
    )
    return png_artifact_name
