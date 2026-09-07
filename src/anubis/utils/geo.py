"""Geo-located avatars: coordinates on avatars, distance, geofences, check-ins.

An avatar may be pinned to a real-world place (a memorial, a museum exhibit, a
grave, a gym machine, a shop) at creation or through ``/modify_avatar``. The
pin lives in the avatar's LangGraph assistant metadata as ``geo_location``::

    {"latitude": 44.9, "longitude": -93.2, "location_name": "...",
     "geofence_radius_meters": 50, "geo_located_at": "<iso>"}

``avatars_near`` ranks avatars by great-circle distance from a point and says
which geofences contain the point; the web application polls that through
``GET /avatars/nearby`` and ``POST /geo/checkin`` as the device moves, and
opens the avatar in camera-backed voice mode when the visitor is inside a
geofence. Check-ins are throttled per visitor so a moving phone cannot flood
the visit records. Both throttles live in the process that serves the request,
so a deployment that runs several worker processes records and notifies at most
once per worker rather than once per fleet.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from typing import Any, Iterable

GEO_LOCATION_METADATA_KEY = "geo_location"
DEFAULT_GEOFENCE_RADIUS_METERS = 50
MIN_GEOFENCE_RADIUS_METERS = 5
MAX_GEOFENCE_RADIUS_METERS = 5000
EARTH_RADIUS_METERS = 6_371_000.0


class GeoLocationError(ValueError):
    """A coordinate or radius outside the valid range."""


def validate_coordinates(latitude: Any, longitude: Any) -> tuple[float, float]:
    """Return the coordinate pair as floats, refusing anything off the globe."""
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        raise GeoLocationError("latitude and longitude must be numbers.")
    if not (-90.0 <= latitude_value <= 90.0):
        raise GeoLocationError("latitude must be between -90 and 90.")
    if not (-180.0 <= longitude_value <= 180.0):
        raise GeoLocationError("longitude must be between -180 and 180.")
    if math.isnan(latitude_value) or math.isnan(longitude_value):
        raise GeoLocationError("latitude and longitude must be numbers.")
    return latitude_value, longitude_value


def validate_geofence_radius(radius_meters: Any) -> int:
    """Return the geofence radius in whole meters, defaulting when unset."""
    if radius_meters is None:
        return DEFAULT_GEOFENCE_RADIUS_METERS
    try:
        radius_value = int(radius_meters)
    except (TypeError, ValueError):
        raise GeoLocationError("geofence_radius_meters must be a whole number of meters.")
    if not (MIN_GEOFENCE_RADIUS_METERS <= radius_value <= MAX_GEOFENCE_RADIUS_METERS):
        raise GeoLocationError(
            f"geofence_radius_meters must be between {MIN_GEOFENCE_RADIUS_METERS} "
            f"and {MAX_GEOFENCE_RADIUS_METERS}."
        )
    return radius_value


def build_geo_location(
    latitude: Any,
    longitude: Any,
    *,
    location_name: str | None = None,
    geofence_radius_meters: Any = None,
) -> dict[str, Any]:
    """Build the validated ``geo_location`` metadata block for an avatar."""
    latitude_value, longitude_value = validate_coordinates(latitude, longitude)
    return {
        "latitude": latitude_value,
        "longitude": longitude_value,
        "location_name": (location_name or "").strip() or None,
        "geofence_radius_meters": validate_geofence_radius(geofence_radius_meters),
        "geo_located_at": datetime.now(tz=UTC).isoformat(),
    }


def geo_location_of(assistant: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read an avatar's ``geo_location`` block from metadata or a public listing."""
    if not assistant:
        return None
    metadata = assistant.get("metadata")
    block = metadata.get(GEO_LOCATION_METADATA_KEY) if isinstance(metadata, dict) else None
    if not isinstance(block, dict):
        block = assistant.get(GEO_LOCATION_METADATA_KEY)
    if not isinstance(block, dict):
        return None
    try:
        validate_coordinates(block.get("latitude"), block.get("longitude"))
    except GeoLocationError:
        return None
    return block


def haversine_distance_meters(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    """Great-circle distance between two coordinates, in meters."""
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    chord = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(min(1.0, chord)))


def avatars_near(
    latitude: float,
    longitude: float,
    assistants: Iterable[dict[str, Any]],
    *,
    radius_meters: float,
    accuracy_meters: float | None = None,
) -> list[dict[str, Any]]:
    """Geo-located avatars within ``radius_meters`` of the point, nearest first.

    Each entry carries the avatar's public fields, the ``geo_location`` block,
    ``distance_meters``, and ``inside_geofence`` (the point, widened by the
    device's reported accuracy, falls inside the avatar's geofence).
    """
    results: list[dict[str, Any]] = []
    slack = max(0.0, float(accuracy_meters or 0.0))
    for assistant in assistants:
        block = geo_location_of(assistant)
        if block is None:
            continue
        distance = haversine_distance_meters(
            latitude, longitude, float(block["latitude"]), float(block["longitude"])
        )
        if distance > radius_meters:
            continue
        fence = float(block.get("geofence_radius_meters") or DEFAULT_GEOFENCE_RADIUS_METERS)
        results.append(
            {
                "assistant_id": assistant.get("assistant_id"),
                "name": assistant.get("name"),
                "description": assistant.get("description"),
                "is_public": bool(
                    (assistant.get("metadata") or {}).get("is_public")
                    if isinstance(assistant.get("metadata"), dict)
                    else assistant.get("is_public")
                ),
                "geo_location": block,
                "distance_meters": round(distance, 1),
                "inside_geofence": distance <= fence + slack,
            }
        )
    results.sort(key=lambda entry: entry["distance_meters"])
    return results


def within_bounds(
    block: dict[str, Any] | None,
    *,
    min_latitude: float | None,
    min_longitude: float | None,
    max_latitude: float | None,
    max_longitude: float | None,
) -> bool:
    """Whether a ``geo_location`` block lies inside the (optional) bounding box.

    A missing bound is open on that side. A box that crosses the antimeridian
    (``min_longitude > max_longitude``) wraps, as map viewports do.
    """
    if block is None:
        return False
    latitude = float(block["latitude"])
    longitude = float(block["longitude"])
    if min_latitude is not None and latitude < min_latitude:
        return False
    if max_latitude is not None and latitude > max_latitude:
        return False
    if min_longitude is not None and max_longitude is not None and min_longitude > max_longitude:
        return longitude >= min_longitude or longitude <= max_longitude
    if min_longitude is not None and longitude < min_longitude:
        return False
    if max_longitude is not None and longitude > max_longitude:
        return False
    return True


def render_avatar_place_section(
    block: dict[str, Any] | None, *, visitor_present: bool = False
) -> str:
    """Prose for the ``=== YOUR PLACE ===`` prompt section (empty when unpinned)."""
    if not block:
        return ""
    place_name = (block.get("location_name") or "").strip()
    latitude = float(block["latitude"])
    longitude = float(block["longitude"])
    where = place_name or "a specific real-world place"
    lines = [
        f"You belong to {where}, at latitude {latitude:.5f}, longitude {longitude:.5f}."
    ]
    if visitor_present:
        lines.append(
            "The person you are speaking with is standing at that place right now, "
            "looking at the place through a camera while talking to you. Greet them "
            "as someone who has come to visit you here, and speak about the place as "
            "the place is around you both."
        )
    else:
        lines.append(
            "The person you are speaking with is not at that place right now."
        )
    return "\n".join(lines)


class CheckinThrottle:
    """Per-visitor, per-avatar throttle on recorded visits (process-local)."""

    def __init__(self, min_interval_seconds: float):
        """Throttle each visitor and avatar pair to one event per interval."""
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_recorded: dict[tuple[str, str], float] = {}

    def allow(self, visitor_id: str, assistant_id: str, now: float | None = None) -> bool:
        """Say whether this visitor and avatar pair may record an event now."""
        now = time.monotonic() if now is None else now
        key = (visitor_id, assistant_id)
        last = self._last_recorded.get(key)
        if last is not None and now - last < self.min_interval_seconds:
            return False
        self._last_recorded[key] = now
        if len(self._last_recorded) > 10_000:
            oldest = sorted(self._last_recorded.items(), key=lambda item: item[1])[:5_000]
            for stale_key, _ in oldest:
                self._last_recorded.pop(stale_key, None)
        return True


__all__ = [
    "DEFAULT_GEOFENCE_RADIUS_METERS",
    "GEO_LOCATION_METADATA_KEY",
    "CheckinThrottle",
    "GeoLocationError",
    "avatars_near",
    "build_geo_location",
    "geo_location_of",
    "haversine_distance_meters",
    "render_avatar_place_section",
    "validate_coordinates",
    "validate_geofence_radius",
    "within_bounds",
]
