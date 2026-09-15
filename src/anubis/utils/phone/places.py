"""Look up a local place: phone, address, hours.

Nominatim finds the shop. Overpass fills phone, hours, and address. A website
``tel:`` or JSON-LD telephone is a last free fill. Google Places runs only when
``GOOGLE_PLACES_API_KEY`` is set and the number is still missing.

This is a personal-avatar capability. It never calls a mobile MCP tool.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import quote

from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164

logger = logging.getLogger(__name__)

DEFAULT_NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
DEFAULT_OVERPASS_BASE_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_CACHE_TTL_SECONDS = 604800
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_last_nominatim_at = 0.0
_TEL_HREF = re.compile(r"""href=["']tel:([^"']+)["']""", re.IGNORECASE)
_JSON_LD_TELEPHONE = re.compile(
    r'"telephone"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


def cache_key_for(name: str, city: str) -> str:
    """Stable cache key for one (name, city) pair."""
    return f"{str(name or '').strip().casefold()}|{str(city or '').strip().casefold()}"


def read_place_cache(name: str, city: str) -> dict[str, Any] | None:
    """Return a still-fresh cached place, or None."""
    key = cache_key_for(name, city)
    entry = _cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _cache.pop(key, None)
        return None
    return dict(value)


def write_place_cache(
    name: str, city: str, value: dict[str, Any], ttl_seconds: int
) -> None:
    """Store ``value`` under (name, city) for ``ttl_seconds``."""
    _cache[cache_key_for(name, city)] = (time.time() + max(1, int(ttl_seconds)), dict(value))


def clear_place_cache() -> None:
    """Drop every cached place (tests)."""
    _cache.clear()


def _http_timeout(context: Any) -> float:
    return float(getattr(context, "phone_lookup_http_timeout_seconds", None) or 15.0)


def _user_agent() -> str:
    return "NeuralNexus/1.0 (personal-avatar place lookup; +https://neuralnexus.site)"


async def _get_json(
    url: str, context: Any, *, params: dict[str, Any] | None = None
) -> Any:
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout(context)) as client:
        response = await client.get(
            url,
            params=params,
            headers={"User-Agent": _user_agent(), "Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()


async def _post_text(url: str, body: str, context: Any) -> Any:
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout(context)) as client:
        response = await client.post(
            url,
            content=body.encode("utf-8"),
            headers={
                "User-Agent": _user_agent(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        return response.json()


async def _nominatim_search(name: str, city: str, context: Any) -> dict[str, Any] | None:
    """Return the first amenity/shop Nominatim names for ``name`` in ``city``."""
    global _last_nominatim_at
    wait = NOMINATIM_MIN_INTERVAL_SECONDS - (time.time() - _last_nominatim_at)
    if wait > 0:
        import asyncio

        await asyncio.sleep(wait)
    base = str(
        getattr(context, "nominatim_base_url", None) or DEFAULT_NOMINATIM_BASE_URL
    ).rstrip("/")
    query = f"{name} {city}".strip()
    results = await _get_json(
        f"{base}/search",
        context,
        params={
            "q": query,
            "format": "json",
            "addressdetails": 1,
            "limit": 5,
        },
    )
    _last_nominatim_at = time.time()
    if not isinstance(results, list):
        return None
    for row in results:
        if not isinstance(row, dict):
            continue
        return row
    return None


def _osm_id_from_nominatim(row: dict[str, Any]) -> tuple[str, int] | None:
    osm_type = str(row.get("osm_type") or "").strip().lower()
    raw_id = row.get("osm_id")
    try:
        osm_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    type_letter = {"node": "node", "way": "way", "relation": "relation"}.get(osm_type)
    if type_letter is None:
        return None
    return type_letter, osm_id


async def _overpass_tags(
    osm_kind: str, osm_id: int, context: Any
) -> dict[str, str]:
    base = str(
        getattr(context, "overpass_base_url", None) or DEFAULT_OVERPASS_BASE_URL
    ).rstrip("/")
    query = f"[out:json][timeout:15];{osm_kind}({osm_id});out tags;"
    payload = f"data={quote(query)}"
    body = await _post_text(base, payload, context)
    elements = body.get("elements") if isinstance(body, dict) else None
    if not isinstance(elements, list) or not elements:
        return {}
    tags = elements[0].get("tags") if isinstance(elements[0], dict) else None
    if not isinstance(tags, dict):
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def _phone_from_tags(tags: dict[str, str]) -> str | None:
    for key in ("phone", "contact:phone", "telephone"):
        raw = tags.get(key)
        if not raw:
            continue
        try:
            return normalize_e164(raw.split(";")[0])
        except PhoneNumberError:
            continue
    return None


def _address_from(row: dict[str, Any], tags: dict[str, str]) -> str:
    display = str(row.get("display_name") or "").strip()
    house = tags.get("addr:housenumber", "")
    street = tags.get("addr:street", "")
    city = tags.get("addr:city") or tags.get("addr:town") or ""
    if house and street:
        parts = [f"{house} {street}".strip()]
        if city:
            parts.append(city)
        return ", ".join(parts)
    return display


async def _phone_from_website(website: str, context: Any) -> str | None:
    if not website:
        return None
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=_http_timeout(context), follow_redirects=True
        ) as client:
            response = await client.get(
                website, headers={"User-Agent": _user_agent()}
            )
        text = response.text or ""
    except Exception as fetch_error:
        logger.info("Place website fetch failed for %s: %s", website, fetch_error)
        return None
    match = _TEL_HREF.search(text)
    if match:
        try:
            return normalize_e164(match.group(1))
        except PhoneNumberError:
            pass
    match = _JSON_LD_TELEPHONE.search(text)
    if match:
        try:
            return normalize_e164(match.group(1))
        except PhoneNumberError:
            pass
    return None


async def _google_places_phone(
    name: str, city: str, context: Any
) -> dict[str, Any] | None:
    api_key = str(getattr(context, "google_places_api_key", None) or "").strip()
    if not api_key:
        return None
    try:
        found = await _get_json(
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            context,
            params={
                "input": f"{name} {city}",
                "inputtype": "textquery",
                "fields": "place_id",
                "key": api_key,
            },
        )
        candidates = found.get("candidates") if isinstance(found, dict) else None
        if not candidates:
            return None
        place_id = candidates[0].get("place_id")
        details = await _get_json(
            "https://maps.googleapis.com/maps/api/place/details/json",
            context,
            params={
                "place_id": place_id,
                "fields": "formatted_phone_number,international_phone_number,formatted_address,opening_hours,website,geometry",
                "key": api_key,
            },
        )
        result = details.get("result") if isinstance(details, dict) else None
        if not isinstance(result, dict):
            return None
        raw_phone = result.get("international_phone_number") or result.get(
            "formatted_phone_number"
        )
        phone = None
        if raw_phone:
            try:
                phone = normalize_e164(str(raw_phone))
            except PhoneNumberError:
                phone = None
        location = (result.get("geometry") or {}).get("location") or {}
        hours = ""
        opening = result.get("opening_hours") or {}
        weekday = opening.get("weekday_text")
        if isinstance(weekday, list):
            hours = "; ".join(str(line) for line in weekday)
        return {
            "phone_e164": phone,
            "address": str(result.get("formatted_address") or ""),
            "website": str(result.get("website") or ""),
            "hours": hours,
            "latitude": location.get("lat"),
            "longitude": location.get("lng"),
            "source": "google_places",
        }
    except Exception as places_error:
        logger.info("Google Places fallback failed: %s", places_error)
        return None


def _empty_place(name: str, city: str) -> dict[str, Any]:
    return {
        "name": name,
        "city": city,
        "phone_e164": None,
        "address": "",
        "latitude": None,
        "longitude": None,
        "hours": "",
        "website": "",
        "source": "not_found",
    }


async def lookup_local_place(
    name: str,
    city: str,
    context: Any,
) -> dict[str, Any]:
    """Return phone, address, hours for ``name`` in ``city``.

    A cache hit returns immediately. A place with no dialable number is still
    returned; the outbound-call tool refuses to dial it.
    """
    place_name = str(name or "").strip()
    place_city = str(city or "").strip()
    if not place_name:
        return _empty_place(place_name, place_city)
    ttl = int(
        getattr(context, "place_lookup_cache_ttl_seconds", None)
        or DEFAULT_CACHE_TTL_SECONDS
    )
    cached = read_place_cache(place_name, place_city)
    if cached is not None:
        cached["cached"] = True
        return cached

    place = _empty_place(place_name, place_city)
    try:
        nominatim = await _nominatim_search(place_name, place_city, context)
    except Exception as search_error:
        logger.info("Nominatim search failed: %s", search_error)
        nominatim = None
    tags: dict[str, str] = {}
    if nominatim is not None:
        place["address"] = str(nominatim.get("display_name") or "")
        try:
            place["latitude"] = float(nominatim.get("lat"))
            place["longitude"] = float(nominatim.get("lon"))
        except (TypeError, ValueError):
            pass
        osm = _osm_id_from_nominatim(nominatim)
        if osm is not None:
            try:
                tags = await _overpass_tags(osm[0], osm[1], context)
            except Exception as overpass_error:
                logger.info("Overpass lookup failed: %s", overpass_error)
        place["hours"] = tags.get("opening_hours", "")
        place["website"] = tags.get("website") or tags.get("contact:website") or ""
        place["address"] = _address_from(nominatim, tags)
        place["phone_e164"] = _phone_from_tags(tags)
        place["source"] = "nominatim_overpass"
        if not place["phone_e164"] and place["website"]:
            place["phone_e164"] = await _phone_from_website(place["website"], context)
            if place["phone_e164"]:
                place["source"] = "website"

    if not place["phone_e164"]:
        google = await _google_places_phone(place_name, place_city, context)
        if google is not None:
            for key, value in google.items():
                if value not in (None, ""):
                    place[key] = value

    place["cached"] = False
    write_place_cache(place_name, place_city, place, ttl)
    return place


def place_as_json(place: dict[str, Any]) -> str:
    """Serialize a place record for a tool result."""
    return json.dumps(place, default=str)
