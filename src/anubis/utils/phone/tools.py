"""LangChain tools for personal-avatar place lookup, travel, and SIP calls.

``lookup_local_place`` and ``estimate_travel`` attach whenever the owner talks
to their personal avatar. They never call a mobile MCP tool. SIP tools attach
only after a live Phone connection and use names that do not collide with
iOS ``place_call``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164

logger = logging.getLogger(__name__)

LOOKUP_TRAVEL_TOOL_NAMES = ("lookup_local_place", "estimate_travel")
SIP_TOOL_NAMES = (
    "request_outbound_phone_call",
    "get_phone_call_status",
    "end_phone_call",
)
PHONE_TOOL_NAMES = LOOKUP_TRAVEL_TOOL_NAMES + SIP_TOOL_NAMES


def _runtime_store(runtime: dict[str, Any]) -> Any:
    return runtime.get("store")


def phone_record_from_accounts(accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first live Phone connection, if any."""
    for record in accounts:
        if record.get("kind") != "telephony":
            continue
        transport = record.get("transport") or {}
        if transport.get("sip_enabled") is True and (
            transport.get("owner_mobile_e164") or record.get("account_address")
        ):
            return record
    return None


def owner_mobile_from_record(record: dict[str, Any]) -> str | None:
    """The verified mobile stored on a Phone connection."""
    transport = record.get("transport") or {}
    raw = transport.get("owner_mobile_e164") or record.get("account_address")
    try:
        return normalize_e164(str(raw or ""))
    except PhoneNumberError:
        return None


def build_personal_avatar_place_tools(
    context: Any,
    *,
    store: Any = None,
    user_id: str = "",
    assistant_id: str = "",
) -> list[Any]:
    """Always-on personal-avatar tools: lookup and travel. No SIP, no iOS."""
    from langchain_core.tools import tool

    from src.anubis.utils.phone.places import lookup_local_place as lookup_place
    from src.anubis.utils.phone.travel import estimate_travel as estimate

    @tool
    async def lookup_local_place(name: str, city: str) -> dict[str, Any]:
        """Look up a local shop or restaurant by name and city.

        Returns the dialable number, street address, hours, and coordinates
        from OpenStreetMap (Nominatim and Overpass). Use this before asking
        to place a phone order. This tool never opens the conversation
        partner's phone and never calls a mobile Model Context Protocol tool.

        Args:
            name: The place's name, for example Kanji or Mellow Mushroom.
            city: The city or neighbourhood to search in.
        """
        place = await lookup_place(name, city, context)
        if not place.get("phone_e164"):
            place["can_request_outbound_phone_call"] = False
            place["message"] = (
                f"No dialable number was found for {name} in {city}. "
                "Say so plainly. Do not call request_outbound_phone_call."
            )
        else:
            place["can_request_outbound_phone_call"] = True
        return place

    @tool
    async def estimate_travel(
        destination: str,
        origin: str | None = None,
        destination_latitude: float | None = None,
        destination_longitude: float | None = None,
    ) -> dict[str, Any]:
        """Estimate driving time from the conversation partner to a place.

        Origin is, in order: the origin argument if the conversation partner
        said where they are; the last web-reported location; a recent geo
        visit; or origin_needed. This tool never calls iOS get_location.

        Args:
            destination: A place name, an address, or "lat,lon".
            origin: Optional address or "lat,lon" the conversation partner stated.
            destination_latitude: Optional latitude from a prior lookup.
            destination_longitude: Optional longitude from a prior lookup.
        """
        return await estimate(
            destination,
            context,
            origin=origin,
            store=store,
            user_id=user_id,
            assistant_id=assistant_id,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
        )

    return [lookup_local_place, estimate_travel]


def build_phone_sip_tools(
    context: Any,
    accounts: list[dict[str, Any]],
    **runtime: Any,
) -> list[Any]:
    """SIP tools, only when a live Phone connection is in ``accounts``."""
    from langchain_core.tools import tool
    from langgraph.types import interrupt

    from src.anubis.utils.phone.call_session import new_call_id
    from src.anubis.utils.phone.livekit_sip import (
        LiveKitNotConfigured,
        livekit_is_configured,
        ring_owner_then_destination,
    )
    from src.anubis.utils.phone.repository import get_phone_call_repository
    from src.anubis.utils.phone.travel import estimate_travel as estimate

    record = phone_record_from_accounts(accounts)
    if record is None:
        return []
    owner_mobile = owner_mobile_from_record(record)
    if not owner_mobile:
        return []
    user_id = str(record.get("user_id") or runtime.get("user_id") or "")
    assistant_id = str(record.get("assistant_id") or runtime.get("assistant_id") or "")
    store = _runtime_store(runtime)

    @tool
    async def request_outbound_phone_call(
        destination_name: str,
        destination_phone_e164: str,
        item: str,
        city: str = "",
        size: str = "",
        pickup_name: str = "",
        confirm: bool = False,
        destination_address: str = "",
        destination_latitude: float | None = None,
        destination_longitude: float | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Place an outbound restaurant order over the shared SIP trunk.

        Rings the conversation partner's already-verified mobile first so
        they can listen, then dials the restaurant. Pickup and pay at the
        counter only — never speak a card number. Do not call this tool
        unless lookup_local_place returned a phone_e164. This is not
        place_call and does not open the iOS dialer.

        Args:
            destination_name: Restaurant name, for example Mellow Mushroom.
            destination_phone_e164: Dialable E.164 number from lookup_local_place.
            item: What to order, in the conversation partner's words.
            city: City used for the lookup, when known.
            size: Size if the conversation partner named one.
            pickup_name: Name to give the restaurant.
            confirm: True only after the conversation partner accepted the confirm card.
            destination_address: Street address from the lookup, when known.
            destination_latitude: Latitude from the lookup, when known.
            destination_longitude: Longitude from the lookup, when known.
            thread_id: Originating chat thread, when the runtime supplies one.
        """
        try:
            destination = normalize_e164(destination_phone_e164)
        except PhoneNumberError as number_error:
            return {
                "status": "refused",
                "reason": "no_phone",
                "message": str(number_error),
            }

        travel: dict[str, Any] = {}
        destination_for_travel = destination_address or destination_name
        if destination_for_travel or (
            destination_latitude is not None and destination_longitude is not None
        ):
            try:
                travel = await estimate(
                    destination_for_travel or destination_name,
                    context,
                    store=store,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    destination_latitude=destination_latitude,
                    destination_longitude=destination_longitude,
                )
            except Exception:
                logger.info("Travel estimate before the confirm card failed", exc_info=True)
                travel = {"duration_text": "unknown"}

        if not confirm:
            decision = interrupt(
                {
                    "kind": "phone_call_confirm",
                    "destination_name": destination_name,
                    "destination_phone_e164": destination,
                    "destination_address": destination_address,
                    "item": item,
                    "size": size,
                    "pickup_name": pickup_name,
                    "travel": travel,
                    "payment_rule": "pickup_pay_at_counter",
                    "message": (
                        f"Call {destination_name} to order {item}"
                        + (f" ({size})" if size else "")
                        + "? The avatar will ring your mobile first so you can listen. "
                        "Pickup and pay at the counter only — no card on the line."
                    ),
                }
            )
            accepted = False
            if isinstance(decision, dict):
                accepted = str(decision.get("type") or decision.get("decision") or "").lower() in {
                    "apply",
                    "accept",
                    "confirm",
                    "yes",
                } or decision.get("confirm") is True
            elif isinstance(decision, str):
                accepted = decision.strip().lower() in {"apply", "accept", "confirm", "yes"}
            if not accepted:
                return {
                    "status": "cancelled",
                    "message": "The conversation partner closed the confirm card. No call was placed.",
                }

        repository = get_phone_call_repository()
        call_id = new_call_id()
        brief = {
            "item": item,
            "size": size,
            "pickup_name": pickup_name,
            "destination_name": destination_name,
            "city": city,
            "do_not_pay": True,
            "travel": travel,
        }
        record_row = {
            "call_id": call_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "direction": "outbound",
            "state": "ringing_owner",
            "owner_mobile_e164": owner_mobile,
            "destination_e164": destination,
            "destination_name": destination_name,
            "room_name": f"phone-{call_id}",
            "brief": brief,
        }
        if repository is not None:
            await repository.create_call(record_row)

        if livekit_is_configured(context):
            try:
                await ring_owner_then_destination(
                    context,
                    room_name=record_row["room_name"],
                    owner_mobile_e164=owner_mobile,
                    destination_e164=destination,
                )
            except (LiveKitNotConfigured, Exception) as dial_error:
                logger.warning("Outbound SIP dial failed: %s", dial_error)
                if repository is not None:
                    await repository.update_call(call_id, {"state": "failed"})
                return {
                    "status": "failed",
                    "call_id": call_id,
                    "message": "The outbound call could not be placed. Try again in a moment.",
                }
        else:
            logger.info(
                "LiveKit is not configured; recorded outbound call %s without dialing.",
                call_id,
            )

        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            writer(
                {
                    "type": "phone_call",
                    "phase": "ringing_owner",
                    "call_id": call_id,
                    "destination_name": destination_name,
                }
            )
        except Exception:
            logger.debug("Could not emit a phone_call stream frame", exc_info=True)

        return {
            "status": "started",
            "call_id": call_id,
            "state": "ringing_owner",
            "owner_mobile_e164": owner_mobile,
            "destination_e164": destination,
            "destination_name": destination_name,
            "travel": travel,
            "listen_path": f"/phone_calls/{call_id}/listen",
        }

    @tool
    async def get_phone_call_status(call_id: str) -> dict[str, Any]:
        """Return the state and result of one outbound or inbound phone call.

        Args:
            call_id: The identifier request_outbound_phone_call returned.
        """
        repository = get_phone_call_repository()
        if repository is None:
            return {"status": "unknown", "call_id": call_id, "message": "No phone-call store is published."}
        stored = await repository.get_call(call_id)
        if stored is None:
            return {"status": "not_found", "call_id": call_id}
        return {
            "status": stored.get("state"),
            "call_id": call_id,
            "destination_name": stored.get("destination_name"),
            "result": stored.get("result") or {},
            "transcript": stored.get("transcript"),
        }

    @tool
    async def end_phone_call(call_id: str) -> dict[str, Any]:
        """Hang up an in-progress phone call.

        Args:
            call_id: The identifier request_outbound_phone_call returned.
        """
        repository = get_phone_call_repository()
        if repository is None:
            return {"status": "unknown", "call_id": call_id}
        stored = await repository.update_call(call_id, {"state": "ended"})
        if stored is None:
            return {"status": "not_found", "call_id": call_id}
        return {"status": "ended", "call_id": call_id}

    return [request_outbound_phone_call, get_phone_call_status, end_phone_call]


def phone_tools_as_mcp_descriptors(include_sip: bool) -> list[dict[str, Any]]:
    """JSON-RPC ``tools/list`` rows. SIP names are omitted until connected."""
    descriptors = [
        {
            "name": "lookup_local_place",
            "description": "Look up a local shop or restaurant by name and city.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["name", "city"],
            },
        },
        {
            "name": "estimate_travel",
            "description": "Estimate driving time to a place. Never calls iOS get_location.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "origin": {"type": "string"},
                },
                "required": ["destination"],
            },
        },
    ]
    if include_sip:
        descriptors.extend(
            [
                {
                    "name": "request_outbound_phone_call",
                    "description": (
                        "Place an outbound restaurant order over SIP after a confirm. "
                        "Not place_call. Never speaks a card number."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "destination_name": {"type": "string"},
                            "destination_phone_e164": {"type": "string"},
                            "item": {"type": "string"},
                            "confirm": {"type": "boolean"},
                        },
                        "required": [
                            "destination_name",
                            "destination_phone_e164",
                            "item",
                        ],
                    },
                },
                {
                    "name": "get_phone_call_status",
                    "description": "Return the state of one phone call.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"call_id": {"type": "string"}},
                        "required": ["call_id"],
                    },
                },
                {
                    "name": "end_phone_call",
                    "description": "Hang up an in-progress phone call.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"call_id": {"type": "string"}},
                        "required": ["call_id"],
                    },
                },
            ]
        )
    return descriptors


def serialize_tool_result(value: Any) -> str:
    """JSON-RPC tool result text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)
