"""Fetching browsing history from the owner's machines, cheaply and incrementally.

Every machine the owner runs the connector on exposes the same three tools,
whatever platform the machine is (see the daemon's ``history_tools``). This
module is the API's side of them, and it enforces the rule that makes the
whole feature affordable:

    ask what is new before asking for anything.

``browsing_activity_summary`` costs one indexed count per browser profile and
returns no rows. Only when that count clears the threshold does anything get
read, and only visits newer than the watermark are read. A person who has not
opened a browser since the last pass costs one count and no model call at all.

Watermarks live per ``(user, avatar, machine)`` in the store, so two machines
advance independently and a machine that was offline for a week is caught up
from where the machine left off rather than re-read from the beginning.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.anubis.utils.tools.data_analysis.discovery import McpConnection

logger = logging.getLogger(__name__)

# The daemon tools, named exactly as every platform's connector registers them.
TOOL_BROWSING_ACTIVITY = "browsing_activity_summary"
TOOL_READ_HISTORY = "read_browser_history"
TOOL_LIST_HISTORY_PROFILES = "list_browser_history_profiles"

WATERMARK_KIND = "browsing_watermark"

DEFAULT_DEVICE_TIMEOUT_SECONDS = 45.0


def watermark_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Where each machine's browsing watermark lives for one avatar."""
    return (user_id, assistant_id, WATERMARK_KIND)


async def read_watermark(
    store: Any, user_id: str, assistant_id: str, device_id: str
) -> dict[str, Any]:
    """Return the record of what has already been analysed from one machine.

    An empty record means this machine has never been read, which is what
    starts the first backfill.
    """
    if store is None or not device_id:
        return {}
    try:
        item = await store.aget(watermark_namespace(user_id, assistant_id), device_id)
    except Exception as read_error:  # noqa: BLE001 - a pass must survive a store hiccup
        logger.warning("Could not read the browsing watermark: %s", read_error)
        return {}
    value = getattr(item, "value", None)
    return dict(value) if isinstance(value, dict) else {}


async def write_watermark(
    store: Any, user_id: str, assistant_id: str, device_id: str, record: dict[str, Any]
) -> bool:
    """Record how far this machine's browsing has been analysed."""
    if store is None or not device_id:
        return False
    try:
        await store.aput(
            watermark_namespace(user_id, assistant_id), key=device_id, value=dict(record)
        )
    except Exception as write_error:  # noqa: BLE001 - findings are already stored
        logger.error("Could not write the browsing watermark: %s", write_error)
        return False
    return True


async def _call_device(
    connection: McpConnection,
    tool_name: str,
    tool_arguments: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    """Call one tool on one machine, converting every failure into a value.

    A machine that is asleep, or a connector too old to carry these tools, must
    not fail the pass for the machines that did answer.
    """
    from src.anubis.utils.tools.data_analysis.mcp_client import call_mcp_filesystem_tool

    try:
        return await asyncio.wait_for(
            call_mcp_filesystem_tool(connection, tool_name, tool_arguments),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return {
            "status": "offline",
            "detail": f"{connection.device_label} did not answer before the deadline.",
        }
    except Exception as call_error:  # noqa: BLE001 - reported, never raised
        message = str(call_error)
        if "does not expose" in message:
            return {
                "status": "unsupported",
                "detail": (
                    f"The connector on {connection.device_label} is too old to share "
                    "browsing history. Update the connector."
                ),
            }
        logger.info("%s could not be reached: %s", connection.device_label, message)
        return {"status": "unreachable", "detail": message}


def _failed(result: Any) -> dict[str, Any] | None:
    """Return the failure a machine reported, or ``None`` when there was none."""
    if isinstance(result, dict):
        if result.get("status") in {"offline", "unreachable", "unsupported"}:
            return result
        if result.get("disabled"):
            return {
                "status": "disabled",
                "detail": str(result.get("reason") or "Browsing history is not shared."),
            }
        return None
    if isinstance(result, str) and result.strip():
        return {"status": "error", "detail": result.strip()}
    return None


async def new_visit_count(
    connection: McpConnection, since: str, *, timeout_seconds: float = DEFAULT_DEVICE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """How much browsing one machine has seen since a watermark.

    The cheap question: counts only, no rows, no model call behind it.
    """
    result = await _call_device(
        connection, TOOL_BROWSING_ACTIVITY, {"since": since or ""}, timeout_seconds
    )
    failure = _failed(result)
    if failure is not None:
        return {"visit_count": 0, "device_label": connection.device_label, **failure}
    if not isinstance(result, dict):
        return {
            "visit_count": 0,
            "device_label": connection.device_label,
            "status": "error",
            "detail": "The machine answered with something that was not a summary.",
        }
    return {
        "visit_count": int(result.get("visit_count") or 0),
        "latest_visit": result.get("latest_visit"),
        "watermark": result.get("watermark") or since,
        "device_label": connection.device_label,
        "device_id": connection.device_id,
        "platform": result.get("platform") or connection.platform,
        "status": "ok",
    }


async def read_new_visits(
    connection: McpConnection,
    since: str,
    *,
    limit: int,
    default_days: int = 30,
    timeout_seconds: float = DEFAULT_DEVICE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return the visits one machine has recorded since a watermark, oldest first."""
    result = await _call_device(
        connection,
        TOOL_READ_HISTORY,
        {"since": since or "", "limit": int(limit), "default_days": int(default_days)},
        timeout_seconds,
    )
    failure = _failed(result)
    if failure is not None:
        return {
            "visits": [],
            "visit_count": 0,
            "device_label": connection.device_label,
            "device_id": connection.device_id,
            **failure,
        }
    if not isinstance(result, dict) or not isinstance(result.get("visits"), list):
        return {
            "visits": [],
            "visit_count": 0,
            "device_label": connection.device_label,
            "device_id": connection.device_id,
            "status": "error",
            "detail": "The machine answered with something that was not a history.",
        }
    visits = [visit for visit in result["visits"] if isinstance(visit, dict)]
    for visit in visits:
        # Every downstream reader can say which machine a visit came from,
        # which matters when a person browses on a work machine and a personal
        # one and the two lives are different.
        visit.setdefault("device_label", connection.device_label)
    return {
        "visits": visits,
        "visit_count": len(visits),
        "watermark": result.get("watermark") or since,
        "profiles_read": result.get("profiles_read") or [],
        "profiles_unreadable": result.get("profiles_unreadable") or [],
        "platform": result.get("platform") or connection.platform,
        "device_label": connection.device_label,
        "device_id": connection.device_id,
        "status": "ok",
    }


async def online_connections(
    store: Any, user_id: str, assistant_id: str
) -> list[McpConnection]:
    """Return the avatar's machines that are reachable right now."""
    from src.anubis.utils.tools.data_analysis.discovery import bound_connections_for

    connections = await bound_connections_for(store, user_id, assistant_id)
    return [connection for connection in connections if getattr(connection, "online", True)]
