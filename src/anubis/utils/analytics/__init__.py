"""Analytics engine: platform metrics, finance, vendor usage, charts, reports, schedules.

Two boot-time entry points are published here for the FastAPI lifespan:
``ensure_analytics_tables`` runs every table script in the package and
``publish_analytics_repositories`` binds the repositories to the pool. Both
import their siblings lazily so importing this package on the request path
stays cheap, and neither touches ``development.py`` (owned elsewhere).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_analytics_pool: Any | None = None


def set_analytics_pool(pool: Any | None) -> None:
    """Publish the psycopg pool the analytics tools query through."""
    global _analytics_pool
    _analytics_pool = pool


def get_analytics_pool() -> Any | None:
    """Return the published analytics pool, or ``None``."""
    return _analytics_pool


async def ensure_analytics_tables(pool: Any) -> None:
    """Create every analytics table if absent: tool calls, reports, schedules, finance, vendor usage."""
    from src.anubis.utils.analytics.finance import ensure_finance_tables
    from src.anubis.utils.analytics.reports import ensure_reports_table
    from src.anubis.utils.analytics.schedules import ensure_schedules_table
    from src.anubis.utils.analytics.tool_calls import ensure_tool_calls_table
    from src.anubis.utils.analytics.vendor_usage import ensure_vendor_usage_table

    await ensure_tool_calls_table(pool)
    await ensure_reports_table(pool)
    await ensure_schedules_table(pool)
    await ensure_finance_tables(pool)
    await ensure_vendor_usage_table(pool)


def publish_analytics_repositories(pool: Any) -> None:
    """Bind the tool-call recorder, report and schedule repositories, and the pool."""
    from src.anubis.utils.analytics.reports import (
        PostgresReportRepository,
        set_report_repository,
    )
    from src.anubis.utils.analytics.schedules import (
        PostgresScheduleRepository,
        set_schedule_repository,
    )
    from src.anubis.utils.analytics.tool_calls import set_tool_call_pool

    set_tool_call_pool(pool)
    set_report_repository(PostgresReportRepository(pool))
    set_schedule_repository(PostgresScheduleRepository(pool))
    set_analytics_pool(pool)
