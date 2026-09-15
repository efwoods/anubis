"""Driving time from the owner to a looked-up place.

OpenRouteService is the router. Origin is, in order: a stated origin, the
web-stored owner location, a fresh geo visit, or ``origin_needed``. This
module never calls iOS ``get_location`` or any other mobile MCP tool.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.phone.owner_location import newest_geo_visit, read_owner_location

logger = logging.getLogger(__name__)

DEFAULT_ORIGIN_MAX_AGE_SECONDS = 1800
TRAVEL_CACHE_TTL_SECONDS = 300

_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_travel_cache() -> None:
    """Drop cached routes (tests)."""
    _cache.clear()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_seconds(recorded_at: str | None) -> float | None:
    instant = _parse_iso(recorded_at)
    if instant is None:
        return None
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - instant).total_seconds())


def _parse_lat_lon(value: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


async def _geocode_address(address: str, context: Any) -> tuple[float, float] | None:
    from src.anubis.utils.phone.places import _get_json

    base = str(
        getattr(context, "nominatim_base_url", None)
        or "https://nominatim.openstreetmap.org"
    ).rstrip("/")
    try:
        results = await _get_json(
            f"{base}/search",
            context,
            params={"q": address, "format": "json", "limit": 1},
        )
    except Exception as geocode_error:
        logger.info("Travel geocode failed: %s", geocode_error)
        return None
    if not isinstance(results, list) or not results:
        return None
    row = results[0]
    try:
        return float(row["lat"]), float(row["lon"])
    except (KeyError, TypeError, ValueError):
        return None


async def resolve_origin(
    origin: str | None,
    *,
    store: Any,
    user_id: str,
    assistant_id: str,
    context: Any,
) -> dict[str, Any]:
    """Resolve the travel origin without touching a mobile MCP tool."""
    stated = str(origin or "").strip()
    if stated:
        pair = _parse_lat_lon(stated)
        if pair is None:
            pair = await _geocode_address(stated, context)
        if pair is not None:
            return {
                "latitude": pair[0],
                "longitude": pair[1],
                "origin_source": "stated",
                "origin_needed": False,
            }
        return {"origin_needed": True, "origin_source": None}

    stored = await read_owner_location(store, user_id)
    max_age = int(
        getattr(context, "travel_origin_max_age_seconds", None)
        or DEFAULT_ORIGIN_MAX_AGE_SECONDS
    )
    if stored and stored.get("latitude") is not None:
        age = _age_seconds(stored.get("recorded_at"))
        if age is None or age <= max_age:
            return {
                "latitude": float(stored["latitude"]),
                "longitude": float(stored["longitude"]),
                "origin_source": "owner_location",
                "origin_needed": False,
            }

    visit = await newest_geo_visit(store, user_id, assistant_id)
    if visit and visit.get("latitude") is not None:
        age = _age_seconds(visit.get("visited_at") or visit.get("recorded_at"))
        if age is None or age <= max_age:
            return {
                "latitude": float(visit["latitude"]),
                "longitude": float(visit["longitude"]),
                "origin_source": "geo_visit",
                "origin_needed": False,
            }

    return {"origin_needed": True, "origin_source": None}


async def _openroute_drive(
    origin: tuple[float, float],
    destination: tuple[float, float],
    context: Any,
) -> dict[str, Any] | None:
    api_key = str(getattr(context, "openroute_api_key", None) or "").strip()
    if not api_key:
        return None
    import httpx

    timeout = float(getattr(context, "phone_lookup_http_timeout_seconds", None) or 15.0)
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "coordinates": [
                        [origin[1], origin[0]],
                        [destination[1], destination[0]],
                    ]
                },
            )
            response.raise_for_status()
            body = response.json()
    except Exception as route_error:
        logger.info("OpenRouteService request failed: %s", route_error)
        return None
    features = body.get("features") if isinstance(body, dict) else None
    if not features:
        routes = body.get("routes") if isinstance(body, dict) else None
        if not routes:
            return None
        summary = (routes[0] or {}).get("summary") or {}
        duration = summary.get("duration")
        distance = summary.get("distance")
    else:
        summary = ((features[0] or {}).get("properties") or {}).get("summary") or {}
        duration = summary.get("duration")
        distance = summary.get("distance")
    if duration is None or distance is None:
        return None
    return {
        "duration_seconds": float(duration),
        "distance_meters": float(distance),
    }


def _format_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} min"
    hours, remain = divmod(minutes, 60)
    return f"{hours} hr {remain} min"


def _format_distance(meters: float) -> str:
    miles = meters / 1609.344
    if miles < 0.1:
        return f"{int(round(meters))} m"
    return f"{miles:.1f} mi"


async def estimate_travel(
    destination: str,
    context: Any,
    *,
    origin: str | None = None,
    store: Any = None,
    user_id: str = "",
    assistant_id: str = "",
    destination_latitude: float | None = None,
    destination_longitude: float | None = None,
) -> dict[str, Any]:
    """Return driving duration and distance, or ``origin_needed``."""
    dest_pair: tuple[float, float] | None = None
    if destination_latitude is not None and destination_longitude is not None:
        dest_pair = (float(destination_latitude), float(destination_longitude))
    else:
        stated_dest = _parse_lat_lon(destination)
        dest_pair = stated_dest or await _geocode_address(destination, context)
    if dest_pair is None:
        return {
            "origin_needed": False,
            "error": "destination_not_found",
            "destination": destination,
        }

    resolved = await resolve_origin(
        origin,
        store=store,
        user_id=user_id,
        assistant_id=assistant_id,
        context=context,
    )
    if resolved.get("origin_needed"):
        return {
            "origin_needed": True,
            "origin_source": None,
            "destination": destination,
            "destination_latitude": dest_pair[0],
            "destination_longitude": dest_pair[1],
        }

    origin_pair = (float(resolved["latitude"]), float(resolved["longitude"]))
    cache_key = (
        f"{round(origin_pair[0], 4)},{round(origin_pair[1], 4)}|"
        f"{round(dest_pair[0], 4)},{round(dest_pair[1], 4)}"
    )
    cached = _cache.get(cache_key)
    if cached is not None and time.time() <= cached[0]:
        result = dict(cached[1])
        result["cached"] = True
        return result

    routed = await _openroute_drive(origin_pair, dest_pair, context)
    result: dict[str, Any] = {
        "origin_needed": False,
        "origin_source": resolved.get("origin_source"),
        "origin_latitude": origin_pair[0],
        "origin_longitude": origin_pair[1],
        "destination": destination,
        "destination_latitude": dest_pair[0],
        "destination_longitude": dest_pair[1],
        "cached": False,
    }
    if routed is None:
        result["duration_seconds"] = None
        result["distance_meters"] = None
        result["duration_text"] = "unknown"
        result["distance_text"] = "unknown"
        result["routing"] = "unavailable"
    else:
        result.update(routed)
        result["duration_text"] = _format_duration(routed["duration_seconds"])
        result["distance_text"] = _format_distance(routed["distance_meters"])
        result["routing"] = "openrouteservice"
    _cache[cache_key] = (time.time() + TRAVEL_CACHE_TTL_SECONDS, dict(result))
    return result


def travel_as_json(result: dict[str, Any]) -> str:
    """Serialize a travel estimate for a tool result."""
    return json.dumps(result, default=str)
