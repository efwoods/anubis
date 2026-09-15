"""Last-known owner location from the web app, never from a mobile MCP tool.

Written when the browser already has a geolocation grant (the same permission
``POST /geo/checkin`` uses) and optionally on each chat turn as ``owner_location``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

OWNER_LOCATION_NAMESPACE_KIND = "owner_location"
OWNER_LOCATION_KEY = "current"


def owner_location_namespace(user_id: str) -> tuple[str, str]:
    """Store namespace holding the owner's last web-reported coordinates."""
    return (str(user_id), OWNER_LOCATION_NAMESPACE_KIND)


async def write_owner_location(
    store: Any,
    user_id: str,
    *,
    latitude: float,
    longitude: float,
    source: str = "geo_checkin",
    accuracy_meters: float | None = None,
) -> dict[str, Any]:
    """Persist one owner location. No-op when ``store`` is missing."""
    recorded_at = datetime.now(UTC).isoformat()
    value = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "accuracy_meters": accuracy_meters,
        "source": source,
        "recorded_at": recorded_at,
    }
    if store is None:
        return value
    await store.aput(
        owner_location_namespace(user_id),
        key=OWNER_LOCATION_KEY,
        value={"value": value},
    )
    return value


async def read_owner_location(store: Any, user_id: str) -> dict[str, Any] | None:
    """Return the stored owner location, or None."""
    if store is None:
        return None
    try:
        item = await store.aget(owner_location_namespace(user_id), OWNER_LOCATION_KEY)
    except Exception:
        return None
    if item is None:
        return None
    payload = item.value if hasattr(item, "value") else item
    if isinstance(payload, dict) and "value" in payload:
        payload = payload["value"]
    if not isinstance(payload, dict):
        return None
    return payload


async def newest_geo_visit(
    store: Any, user_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Return the newest geo_visit row for this owner and avatar, if any."""
    if store is None:
        return None
    try:
        namespace = (str(user_id), str(assistant_id), "geo_visit")
        items = await store.asearch(namespace, query=None, limit=20)
    except Exception:
        return None
    newest: dict[str, Any] | None = None
    newest_key = ""
    for item in items or []:
        key = str(getattr(item, "key", "") or "")
        payload = item.value if hasattr(item, "value") else item
        if isinstance(payload, dict) and "value" in payload:
            payload = payload["value"]
        if not isinstance(payload, dict):
            continue
        if key >= newest_key:
            newest_key = key
            newest = payload
    return newest
