"""Keeping the avatar current with the owner's browsing, without paying for silence.

The owner asked for this to be continuous: the avatar should know what the
owner has been reading within minutes of the owner reading it, not once a week.
Continuous and cheap are in tension, and the tension is resolved by asking the
cheap question first.

ONE PASS, PER MACHINE
    1. Read the machine's watermark from the store — what has already been
       analysed. No watermark means this machine has never been read, and the
       pass becomes a bounded backfill of the last
       ``browsing_insights_backfill_days`` days, which is the "learn about me
       when I first connect" case.
    2. Ask the machine how many visits are newer than the watermark. This
       costs one indexed count per browser profile; no rows travel and no
       model is called.
    3. Stop unless the count clears ``browsing_insights_minimum_new_visits``
       AND enough time has passed since this machine's last analysis
       (``browsing_insights_minimum_seconds_between_analyses``). A person
       reading one page has not become a different person; a person who has
       read forty has.
    4. Read only the new visits, analyse them, and write the findings.
    5. Advance the watermark, so nothing is ever read or paid for twice.

The loop runs every ``browsing_insights_poll_seconds`` over every machine that
currently holds a live relay socket, which is why a machine that is asleep
costs nothing at all: the machine is not in the list.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.browsing.history_client import (
    new_visit_count,
    online_connections,
    read_new_visits,
    read_watermark,
    write_watermark,
)
from src.anubis.utils.browsing.insights import analyze_visits, apply_insights

logger = logging.getLogger(__name__)


def _flag_is_true(context: Any, field_name: str, default: bool = False) -> bool:
    """Read one of the browsing switches off the global context."""
    value = getattr(context, field_name, None)
    if value is None or not str(value).strip():
        return default
    return str(value).strip().upper() in {"TRUE", "1", "YES", "ON"}


def _number(context: Any, field_name: str, default: float) -> float:
    """Read one of the browsing numbers off the global context."""
    try:
        value = getattr(context, field_name, None)
        if value is None or not str(value).strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _seconds_since(timestamp: str | None) -> float:
    """How long ago a recorded moment was, in seconds; a huge number when unknown."""
    if not timestamp:
        return float("inf")
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (datetime.now(UTC) - moment).total_seconds()


async def analyse_machine(
    store: Any,
    connection: Any,
    *,
    context: Any,
    user_id: str,
    assistant_id: str,
    target_name: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run one pass for one machine. Returns what the pass did, and why not when not."""
    device_id = getattr(connection, "device_id", "") or ""
    device_label = getattr(connection, "device_label", "") or "this machine"
    watermark_record = await read_watermark(store, user_id, assistant_id, device_id)
    watermark = str(watermark_record.get("watermark") or "")
    first_pass = not watermark
    backfill_days = int(_number(context, "browsing_insights_backfill_days", 30))
    minimum_new_visits = int(_number(context, "browsing_insights_minimum_new_visits", 25))
    minimum_gap_seconds = _number(
        context, "browsing_insights_minimum_seconds_between_analyses", 900
    )
    maximum_visits = int(_number(context, "browsing_insights_max_visits_per_pass", 2000))

    since = watermark or str(backfill_days)
    counted = await new_visit_count(connection, since)
    if counted.get("status") != "ok":
        return {"analysed": False, "reason": counted.get("detail") or counted.get("status")}
    new_visits = int(counted.get("visit_count") or 0)
    if not new_visits:
        return {"analysed": False, "reason": "no new browsing", "device_label": device_label}
    if not force:
        # A first pass runs on any amount of history — that is the point of a
        # backfill. Later passes wait until enough has happened to be worth a
        # model call.
        if not first_pass and new_visits < minimum_new_visits:
            return {
                "analysed": False,
                "reason": f"only {new_visits} new visits",
                "device_label": device_label,
                "new_visits": new_visits,
            }
        since_last = _seconds_since(watermark_record.get("analysed_at"))
        if since_last < minimum_gap_seconds:
            return {
                "analysed": False,
                "reason": "analysed too recently",
                "device_label": device_label,
                "new_visits": new_visits,
            }

    read = await read_new_visits(
        connection, since, limit=maximum_visits, default_days=backfill_days
    )
    if read.get("status") != "ok" or not read.get("visits"):
        return {
            "analysed": False,
            "reason": read.get("detail") or "the machine returned no visits",
            "device_label": device_label,
        }
    visits = read["visits"]
    known_hosts = list(watermark_record.get("hosts") or [])
    insights = await analyze_visits(
        visits,
        target_name=target_name or "the person",
        known_hosts=known_hosts,
        max_digest_characters=int(
            _number(context, "browsing_insights_max_digest_characters", 24000)
        ),
    )
    if insights is None:
        return {
            "analysed": False,
            "reason": "the browsing said nothing worth recording",
            "device_label": device_label,
        }
    applied = await apply_insights(
        store,
        insights,
        user_id=user_id,
        assistant_id=assistant_id,
        visits=visits,
        device_label=device_label,
        write_report=_flag_is_true(context, "browsing_insights_report_enabled", True),
    )
    # The watermark advances to the newest visit READ, never to the count's
    # newest visit: a limit that cut the batch short must leave the remainder
    # for the next pass rather than skipping over it.
    newest_read = visits[-1].get("visited_at") or read.get("watermark") or since
    merged_hosts = sorted(set(known_hosts) | set(applied.get("hosts") or []))[:2000]
    await write_watermark(
        store,
        user_id,
        assistant_id,
        device_id,
        {
            "watermark": newest_read,
            "analysed_at": datetime.now(UTC).isoformat(),
            "device_label": device_label,
            "platform": read.get("platform") or getattr(connection, "platform", ""),
            "hosts": merged_hosts,
            "passes": int(watermark_record.get("passes") or 0) + 1,
            "visits_analysed": int(watermark_record.get("visits_analysed") or 0)
            + len(visits),
        },
    )
    return {
        "analysed": True,
        "device_label": device_label,
        "first_pass": first_pass,
        "watermark": newest_read,
        **applied,
    }


async def analyse_account(
    store: Any,
    *,
    context: Any,
    user_id: str,
    assistant_id: str,
    target_name: str = "",
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run a pass for every machine of one avatar's owner that is reachable."""
    connections = await online_connections(store, user_id, assistant_id)
    if not connections:
        return []
    results = await asyncio.gather(
        *(
            analyse_machine(
                store,
                connection,
                context=context,
                user_id=user_id,
                assistant_id=assistant_id,
                target_name=target_name,
                force=force,
            )
            for connection in connections
        ),
        return_exceptions=True,
    )
    outcomes: list[dict[str, Any]] = []
    for connection, result in zip(connections, results):
        if isinstance(result, BaseException):
            logger.warning(
                "Browsing pass failed for %s: %s", connection.device_label, result
            )
            outcomes.append(
                {
                    "analysed": False,
                    "device_label": connection.device_label,
                    "reason": str(result),
                }
            )
            continue
        outcomes.append(result)
    return outcomes


async def accounts_with_machines_online(store: Any) -> list[tuple[str, str]]:
    """Every ``(owner, avatar)`` pair that has a machine connected right now.

    Read off the live relay registry rather than off the database, so the loop
    only ever touches accounts whose machines can actually answer.
    """
    from src.anubis.utils.tools.data_analysis import relay
    from src.anubis.utils.tools.data_analysis.discovery import read_user_connections

    user_ids = sorted({session.user_id for session in relay.all_sessions()})
    pairs: set[tuple[str, str]] = set()
    for user_id in user_ids:
        try:
            records = await read_user_connections(store, user_id)
        except Exception as read_error:  # noqa: BLE001 - one account must not stop the loop
            logger.warning("Could not read connections for %s: %s", user_id, read_error)
            continue
        for record in records:
            assistant_id = record.get("assistant_id")
            if record.get("status") == "connected" and assistant_id:
                pairs.add((user_id, str(assistant_id)))
    return sorted(pairs)


async def run_browsing_sweep_once(store: Any, context: Any) -> dict[str, int]:
    """One turn of the loop over every account with a machine online."""
    totals = {"accounts": 0, "machines": 0, "analysed": 0, "facts": 0, "traits": 0}
    for user_id, assistant_id in await accounts_with_machines_online(store):
        outcomes = await analyse_account(
            store, context=context, user_id=user_id, assistant_id=assistant_id
        )
        if not outcomes:
            continue
        totals["accounts"] += 1
        totals["machines"] += len(outcomes)
        for outcome in outcomes:
            if outcome.get("analysed"):
                totals["analysed"] += 1
                totals["facts"] += int(outcome.get("fact_count") or 0)
                totals["traits"] += int(outcome.get("trait_count") or 0)
    return totals


async def run_browsing_sweeper(app: Any) -> None:
    """Lifespan task: keep every connected machine's browsing analysed as it happens."""
    context = app.state.context
    interval_seconds = _number(context, "browsing_insights_poll_seconds", 300)
    logger.info(
        "Browsing insights sweeper started (every %.0fs, at least %s new visits per pass)",
        interval_seconds,
        int(_number(context, "browsing_insights_minimum_new_visits", 25)),
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            totals = await run_browsing_sweep_once(app.state.store, context)
            if totals["analysed"]:
                logger.info("Browsing sweep %s", totals)
        except asyncio.CancelledError:
            logger.info("Browsing insights sweeper stopped")
            raise
        except Exception as sweep_error:  # noqa: BLE001 - the loop must survive
            logger.exception("Browsing sweep failed: %s", sweep_error)
