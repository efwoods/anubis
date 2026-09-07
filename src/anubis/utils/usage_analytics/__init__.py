"""Opt-in usage analytics: consent, action events, and described page captures.

A person who opts in (at signup, in the personal avatar's settings, or in
account settings) has the Neural Nexus browser record every action taken in
the application and capture the page itself on an interval and on each route
change; the API describes each capture with the image-description model and
keeps the events, the descriptions, and a thumbnail per user in Postgres.
Nothing is recorded for an account that has not opted in, and nothing is
recorded for anonymous visitors.

Boot-time entry points for the FastAPI lifespan: ``ensure_usage_analytics_tables``
creates the tables and ``publish_usage_analytics_repository`` binds the
repository to the pool. Siblings are imported lazily so importing this
package on the request path stays cheap.
"""

from __future__ import annotations

from typing import Any


async def ensure_usage_analytics_tables(pool: Any) -> None:
    """Create the consent, events, and screenshots tables when absent."""
    from src.anubis.utils.usage_analytics.repository import (
        ensure_usage_analytics_tables as _ensure,
    )

    await _ensure(pool)


def publish_usage_analytics_repository(pool: Any) -> Any:
    """Bind a Postgres repository to ``pool`` and publish the repository."""
    from src.anubis.utils.usage_analytics.repository import (
        PostgresUsageAnalyticsRepository,
        set_usage_analytics_repository,
    )

    repository = PostgresUsageAnalyticsRepository(pool)
    set_usage_analytics_repository(repository)
    return repository


def get_usage_analytics_repository() -> Any | None:
    """Return the published repository, or ``None`` before the lifespan ran."""
    from src.anubis.utils.usage_analytics.repository import (
        get_usage_analytics_repository as _get,
    )

    return _get()
