"""LangSmith trace tally: a hard cap on billable traces per LangSmith billing period.

LangSmith bills every trace past the plan's included traces in a billing
period. The trace tally counts traces as the langsmith client sends them, so the
cap does not depend on LangSmith's usage reporting, which lags real traffic.

How the trace tally works:

- Every run the langsmith client sends (a new run or an update to a run) passes
  through ``Client._filter_for_sampling``, keyed by the run's trace id. The
  trace tally wraps that filter. The first time a trace id is seen, the trace
  takes one slot from the tally; every later run of the same trace follows the
  first decision. A trace that gets no slot is dropped before the trace leaves
  the process, together with every run of the dropped trace. Conversations are
  not affected; only the trace is dropped.
- The tally lives in Postgres (table ``langsmith_trace_tally``), keyed by the
  LangSmith organization id and the end of the LangSmith billing period, so
  every process sending traces for one organization (the production API, the
  development API, the phone workers) shares one count. Each process reserves
  slots in blocks of ``LANGSMITH_TRACE_TALLY_BLOCK_SIZE`` with one atomic
  statement, and a reservation never takes the tally past
  ``LANGSMITH_MONTHLY_TRACE_LIMIT``.
- LangSmith is asked about the billing period only at period boundaries:
  ``GET /api/v1/orgs/current/billing`` returns ``end_of_billing_period``; the
  period sync sleeps until that moment, then reads the next period. A new
  period's tally row is seeded once from LangSmith's own usage for the period,
  so traces LangSmith counted before the trace tally existed are not forgotten.

Failsafes (each one drops traces rather than risk a billed trace):

- No billing period known yet, a billing period already over, the Postgres
  tally unreachable, or a reservation error: new traces are dropped until the
  tally answers again.
- A langsmith release without ``Client._filter_for_sampling``: tracing is
  turned off for the whole process, because the trace tally cannot see traces.
- A payment method on the LangSmith organization is logged as a warning at
  every period sync, because only a card lets LangSmith bill an overage.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_BILLING_PATH = "/api/v1/orgs/current/billing"
LANGSMITH_GRANULAR_USAGE_PATH = "/api/v1/orgs/current/billing/granular-usage"
# Trace ids remembered per process to route later runs of a trace to the first
# decision. An evicted trace id that reappears takes a second slot, which can
# only over-count.
REMEMBERED_TRACE_IDS = 200_000
# Seconds to wait after LangSmith's end of billing period before reading the
# next period, and between retries when LangSmith cannot be read.
PERIOD_BOUNDARY_GRACE_SECONDS = 60.0
PERIOD_SYNC_RETRY_SECONDS = 60.0
# Seconds a process stops asking Postgres after a failed reservation, so an
# unreachable tally costs conversations no connect timeouts; traces are dropped
# in the meantime.
RESERVATION_RETRY_SECONDS = 30.0

CREATE_TALLY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS langsmith_trace_tally (
    organization_id TEXT NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    trace_limit BIGINT NOT NULL,
    seeded_traces BIGINT NOT NULL,
    reserved_traces BIGINT NOT NULL,
    sent_traces BIGINT NOT NULL DEFAULT 0,
    dropped_traces BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, period_end)
)
"""

# Which LangSmith organization and billing period each LangSmith API key
# belongs to, so a restarted process resumes the trace tally from Postgres
# without asking LangSmith (a process that has to ask LangSmith drops every
# trace until LangSmith answers). The key is stored as a SHA-256 fingerprint.
CREATE_KEY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS langsmith_trace_tally_keys (
    api_key_fingerprint TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

UPSERT_KEY_SQL = """
INSERT INTO langsmith_trace_tally_keys (api_key_fingerprint, organization_id, period_end)
VALUES (%s, %s, %s)
ON CONFLICT (api_key_fingerprint) DO UPDATE
SET organization_id = EXCLUDED.organization_id,
    period_end = EXCLUDED.period_end,
    updated_at = now()
"""

SELECT_KNOWN_PERIOD_SQL = """
SELECT tally_keys.organization_id, tally_keys.period_end
FROM langsmith_trace_tally_keys AS tally_keys
JOIN langsmith_trace_tally AS tally
  ON tally.organization_id = tally_keys.organization_id
 AND tally.period_end = tally_keys.period_end
WHERE tally_keys.api_key_fingerprint = %s AND tally_keys.period_end > now()
"""

SELECT_PERIOD_SQL = """
SELECT 1 FROM langsmith_trace_tally WHERE organization_id = %s AND period_end = %s
"""

INSERT_PERIOD_SQL = """
INSERT INTO langsmith_trace_tally
    (organization_id, period_end, trace_limit, seeded_traces, reserved_traces)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (organization_id, period_end) DO NOTHING
"""

# Reserve up to the requested slots without passing the limit, record the
# traces sent and dropped since the last reservation, and return the number of
# slots granted (zero once the limit is reached).
RESERVE_SLOTS_SQL = """
WITH current_period AS (
    SELECT reserved_traces
    FROM langsmith_trace_tally
    WHERE organization_id = %(organization_id)s AND period_end = %(period_end)s
    FOR UPDATE
), granted AS (
    SELECT GREATEST(0, LEAST(%(requested)s, %(limit)s - reserved_traces)) AS granted_slots
    FROM current_period
)
UPDATE langsmith_trace_tally
SET reserved_traces = langsmith_trace_tally.reserved_traces + granted.granted_slots,
    trace_limit = %(limit)s,
    sent_traces = langsmith_trace_tally.sent_traces + %(sent)s,
    dropped_traces = langsmith_trace_tally.dropped_traces + %(dropped)s,
    updated_at = now()
FROM granted
WHERE organization_id = %(organization_id)s AND period_end = %(period_end)s
RETURNING granted.granted_slots
"""


class TraceTally:
    """Process-local view of the shared Postgres trace tally for one LangSmith organization."""

    def __init__(self, postgres_uri: str, trace_limit: int, block_size: int) -> None:
        """Hold the Postgres address, the trace limit, and the reservation block size."""
        self.postgres_uri = postgres_uri
        self.trace_limit = trace_limit
        self.block_size = max(1, block_size)
        self.organization_id: str | None = None
        self.period_end: datetime | None = None
        self.available_slots = 0
        self.unreported_sent_traces = 0
        self.unreported_dropped_traces = 0
        self.trace_decisions: OrderedDict[str, bool] = OrderedDict()
        self.lock = threading.Lock()
        self.connection: Any = None
        self.limit_reached_logged_for: datetime | None = None
        self.reservation_retry_after: datetime | None = None

    def _connection(self) -> Any:
        import psycopg

        if self.connection is None or self.connection.closed:
            self.connection = psycopg.connect(
                self.postgres_uri,
                autocommit=True,
                connect_timeout=3,
                options="-c statement_timeout=2000",
            )
        return self.connection

    def ensure_table(self) -> None:
        """Create the langsmith_trace_tally table when the table does not exist."""
        with self.lock:
            self._connection().execute(CREATE_TALLY_TABLE_SQL)
            self._connection().execute(CREATE_KEY_TABLE_SQL)

    def remember_key_period(
        self, api_key_fingerprint: str, organization_id: str, period_end: datetime
    ) -> None:
        """Record which organization and billing period the LangSmith API key belongs to."""
        with self.lock:
            self._connection().execute(
                UPSERT_KEY_SQL, (api_key_fingerprint, organization_id, period_end)
            )

    def known_period(self, api_key_fingerprint: str) -> tuple[str, datetime] | None:
        """Return the stored organization and open billing period for the LangSmith API key."""
        with self.lock:
            known_period_row = (
                self._connection()
                .execute(SELECT_KNOWN_PERIOD_SQL, (api_key_fingerprint,))
                .fetchone()
            )
        if known_period_row is None:
            return None
        return str(known_period_row[0]), known_period_row[1]

    def tally_row_exists(self, organization_id: str, period_end: datetime) -> bool:
        """Return True when the billing period already has a tally row."""
        with self.lock:
            tally_row = (
                self._connection()
                .execute(SELECT_PERIOD_SQL, (organization_id, period_end))
                .fetchone()
            )
        return tally_row is not None

    def start_period(
        self, organization_id: str, period_end: datetime, seeded_traces: int | None
    ) -> None:
        """Point the trace tally at a billing period; seeded_traces creates the period's row."""
        with self.lock:
            if seeded_traces is not None:
                self._connection().execute(
                    INSERT_PERIOD_SQL,
                    (
                        organization_id,
                        period_end,
                        self.trace_limit,
                        seeded_traces,
                        seeded_traces,
                    ),
                )
            if (self.organization_id, self.period_end) != (organization_id, period_end):
                # Slots reserved in an earlier period do not carry over.
                self.available_slots = 0
                self.trace_decisions.clear()
            self.organization_id = organization_id
            self.period_end = period_end

    def _reserve_block(self) -> int:
        """Reserve up to block_size slots in Postgres; the caller holds self.lock."""
        reservation_row = (
            self._connection()
            .execute(
                RESERVE_SLOTS_SQL,
                {
                    "requested": self.block_size,
                    "limit": self.trace_limit,
                    "sent": self.unreported_sent_traces,
                    "dropped": self.unreported_dropped_traces,
                    "organization_id": self.organization_id,
                    "period_end": self.period_end,
                },
            )
            .fetchone()
        )
        if reservation_row is None:
            raise RuntimeError(
                "No trace tally row exists for the current billing period"
            )
        self.unreported_sent_traces = 0
        self.unreported_dropped_traces = 0
        return int(reservation_row[0])

    def admit_trace(self, trace_id: str) -> bool:
        """Return True when the trace may be sent; the first call per trace id takes a slot."""
        with self.lock:
            trace_decision = self.trace_decisions.get(trace_id)
            if trace_decision is not None:
                self.trace_decisions.move_to_end(trace_id)
                return trace_decision
            trace_decision = self._take_slot()
            self.trace_decisions[trace_id] = trace_decision
            if len(self.trace_decisions) > REMEMBERED_TRACE_IDS:
                self.trace_decisions.popitem(last=False)
            return trace_decision

    def _take_slot(self) -> bool:
        if self.period_end is None or datetime.now(UTC) >= self.period_end:
            # Billing period unknown or already over: fail closed until the
            # period sync reads the next period.
            self.unreported_dropped_traces += 1
            return False
        if self.available_slots == 0:
            now = datetime.now(UTC)
            if self.reservation_retry_after and now < self.reservation_retry_after:
                self.unreported_dropped_traces += 1
                return False
            try:
                self.available_slots = self._reserve_block()
                self.reservation_retry_after = None
            except Exception as reservation_error:  # noqa: BLE001 - fail closed
                self.connection = None
                self.reservation_retry_after = now + timedelta(
                    seconds=RESERVATION_RETRY_SECONDS
                )
                self.unreported_dropped_traces += 1
                logger.warning(
                    "LangSmith trace tally unreachable (%s); the trace is dropped.",
                    reservation_error,
                )
                return False
        if self.available_slots == 0:
            self.unreported_dropped_traces += 1
            if self.limit_reached_logged_for != self.period_end:
                self.limit_reached_logged_for = self.period_end
                logger.warning(
                    "LangSmith trace limit of %s reached for the billing period ending %s; "
                    "traces are dropped until the period resets. Conversations continue.",
                    self.trace_limit,
                    self.period_end.isoformat(),
                )
            return False
        self.available_slots -= 1
        self.unreported_sent_traces += 1
        return True


def _is_true(setting_value: Any) -> bool:
    return str(setting_value or "").strip().lower() in ("1", "true", "yes", "on")


def tracing_requested(context: Any) -> bool:
    """Return True when the operator asked for LangSmith tracing (LANGSMITH_TRACING)."""
    return _is_true(getattr(context, "langsmith_tracing", None))


def trace_limit_of(context: Any) -> int | None:
    """Return LANGSMITH_MONTHLY_TRACE_LIMIT, or None when no limit is configured."""
    configured_limit = getattr(context, "langsmith_monthly_trace_limit", None)
    if configured_limit in (None, ""):
        return None
    return int(configured_limit)


def set_process_tracing(enabled: bool) -> None:
    """Turn LangSmith tracing on or off for every task in this process."""
    from langsmith.run_trees import configure as configure_langsmith

    configure_langsmith(enabled=enabled)


def trace_id_of(run: Any) -> str | None:
    """Return the trace id of a run payload the langsmith client is about to send."""

    def run_value(key: str) -> Any:
        try:
            return run[key]
        except (KeyError, TypeError):
            return getattr(run, key, None)

    trace_id = run_value("trace_id")
    if trace_id is not None:
        return str(trace_id)
    dotted_order = run_value("dotted_order")
    if dotted_order and "Z" in dotted_order:
        return dotted_order.split(".", 1)[0].split("Z", 1)[1]
    run_id = run_value("id")
    return str(run_id) if run_id is not None else None


def install_trace_tally_filter(trace_tally: TraceTally) -> bool:
    """Wrap ``langsmith.Client._filter_for_sampling`` so every run passes the trace tally.

    Returns False when the langsmith release has no such method; the caller
    then turns tracing off, because the trace tally cannot see traces.
    """
    from langsmith import Client

    original_filter = getattr(Client, "_filter_for_sampling", None)
    if original_filter is None:
        return False
    original_filter = getattr(original_filter, "_original_filter", original_filter)

    def filter_through_trace_tally(client: Any, runs: Any) -> list:
        sampled_runs = original_filter(client, runs)
        return [
            run
            for run in sampled_runs
            if (trace_id := trace_id_of(run)) is None
            or trace_tally.admit_trace(trace_id)
        ]

    filter_through_trace_tally._original_filter = original_filter  # type: ignore[attr-defined]
    Client._filter_for_sampling = filter_through_trace_tally  # type: ignore[method-assign]
    return True


async def _langsmith_get(
    context: Any, path: str, query_parameters: dict | None = None
) -> Any:
    import httpx

    endpoint = str(
        getattr(context, "langsmith_endpoint", None) or DEFAULT_LANGSMITH_ENDPOINT
    ).rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        langsmith_response = await http_client.get(
            endpoint + path,
            params=query_parameters,
            headers={
                "x-api-key": str(getattr(context, "langsmith_api_key", None) or "")
            },
        )
    if langsmith_response.status_code != 200:
        raise RuntimeError(
            f"LangSmith {path} answered {langsmith_response.status_code}: "
            f"{langsmith_response.text[:300]}"
        )
    return langsmith_response.json()


async def fetch_billing_period(context: Any) -> dict[str, Any]:
    """Return the LangSmith organization id, end of billing period, and payment method."""
    billing_payload = await _langsmith_get(context, LANGSMITH_BILLING_PATH)
    period_end_text = billing_payload.get("end_of_billing_period")
    if not period_end_text:
        raise RuntimeError("LangSmith reported no end_of_billing_period")
    return {
        "organization_id": str(billing_payload["id"]),
        "period_end": datetime.fromisoformat(period_end_text).astimezone(UTC),
        "payment_method": billing_payload.get("payment_method"),
    }


def period_start_for(period_end: datetime) -> datetime:
    """Return the start of the monthly billing period that ends at period_end."""
    if period_end.month == 1:
        return period_end.replace(year=period_end.year - 1, month=12)
    return period_end.replace(month=period_end.month - 1)


async def fetch_period_traces(context: Any, period_end: datetime) -> int:
    """Return LangSmith's own trace count for the billing period that ends at period_end."""
    usage_payload = await _langsmith_get(
        context,
        LANGSMITH_GRANULAR_USAGE_PATH,
        {
            "start_time": period_start_for(period_end).isoformat(),
            "end_time": min(
                period_end, datetime.now(UTC) + timedelta(hours=1)
            ).isoformat(),
            "group_by": "workspace",
            "kind": "traces",
        },
    )
    return sum(
        int(usage_record.get("traces") or 0)
        for usage_record in usage_payload.get("usage", [])
    )


def api_key_fingerprint_of(context: Any) -> str:
    """Return a SHA-256 fingerprint of LANGSMITH_API_KEY; the key itself is never stored."""
    import hashlib

    langsmith_api_key = str(getattr(context, "langsmith_api_key", None) or "")
    return hashlib.sha256(langsmith_api_key.encode()).hexdigest()[:16]


async def sync_billing_period(context: Any, trace_tally: TraceTally) -> datetime:
    """Read LangSmith's billing period once and point the trace tally at the period."""
    billing_period = await fetch_billing_period(context)
    organization_id = billing_period["organization_id"]
    period_end = billing_period["period_end"]
    if billing_period["payment_method"]:
        logger.warning(
            "The LangSmith organization %s has a payment method on file; LangSmith bills "
            "traces past the plan's included traces, and LANGSMITH_MONTHLY_TRACE_LIMIT=%s "
            "is the only cap on that bill.",
            organization_id,
            trace_tally.trace_limit,
        )
    seeded_traces = None
    if not await asyncio.to_thread(
        trace_tally.tally_row_exists, organization_id, period_end
    ):
        seeded_traces = await fetch_period_traces(context, period_end)
    await asyncio.to_thread(
        trace_tally.start_period, organization_id, period_end, seeded_traces
    )
    await asyncio.to_thread(
        trace_tally.remember_key_period,
        api_key_fingerprint_of(context),
        organization_id,
        period_end,
    )
    logger.info(
        "LangSmith trace tally: organization %s, billing period ends %s, limit %s%s.",
        organization_id,
        period_end.isoformat(),
        trace_tally.trace_limit,
        f", seeded with {seeded_traces} traces LangSmith already counted"
        if seeded_traces is not None
        else "",
    )
    return period_end


async def run_trace_tally(context: Any) -> None:
    """Install the trace tally, then re-read the billing period at every period reset."""
    if not tracing_requested(context):
        return
    trace_limit = trace_limit_of(context)
    if trace_limit is None:
        return
    trace_tally = TraceTally(
        postgres_uri=str(getattr(context, "async_postgres_store_uri", None) or ""),
        trace_limit=trace_limit,
        block_size=int(
            getattr(context, "langsmith_trace_tally_block_size", None) or 10
        ),
    )
    # The filter is installed first: until the billing period is known, the
    # filter drops every trace (fail closed).
    if not install_trace_tally_filter(trace_tally):
        logger.error(
            "langsmith.Client._filter_for_sampling is missing in this langsmith release; "
            "the trace tally cannot see traces, so tracing is off for this process."
        )
        set_process_tracing(False)
        return
    period_end: datetime | None = None
    try:
        # Resume from Postgres when another start of this process (or another
        # process with the same key) already learned the billing period, so
        # the first traces after a restart are counted instead of dropped.
        await asyncio.to_thread(trace_tally.ensure_table)
        known_period = await asyncio.to_thread(
            trace_tally.known_period, api_key_fingerprint_of(context)
        )
        if known_period is not None:
            await asyncio.to_thread(
                trace_tally.start_period, known_period[0], known_period[1], None
            )
            period_end = known_period[1]
            logger.info(
                "LangSmith trace tally resumed from Postgres: organization %s, "
                "billing period ends %s, limit %s.",
                known_period[0],
                period_end.isoformat(),
                trace_limit,
            )
    except Exception as resume_error:  # noqa: BLE001 - fall back to LangSmith
        logger.warning("LangSmith trace tally could not resume (%s).", resume_error)
    while True:
        try:
            if period_end is not None:
                # LangSmith is asked only when the billing period resets.
                seconds_until_reset = (period_end - datetime.now(UTC)).total_seconds()
                await asyncio.sleep(
                    max(0.0, seconds_until_reset) + PERIOD_BOUNDARY_GRACE_SECONDS
                )
            await asyncio.to_thread(trace_tally.ensure_table)
            period_end = await sync_billing_period(context, trace_tally)
        except asyncio.CancelledError:
            return
        except Exception as period_sync_error:  # noqa: BLE001 - retry; traces stay capped
            logger.warning(
                "LangSmith trace tally could not read the billing period (%s); "
                "retrying in %s seconds.",
                period_sync_error,
                PERIOD_SYNC_RETRY_SECONDS,
            )
            period_end = None
            await asyncio.sleep(PERIOD_SYNC_RETRY_SECONDS)
