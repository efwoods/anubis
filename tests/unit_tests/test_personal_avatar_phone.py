"""Personal-avatar phone: lookup, travel, SIP names, connect gate, results."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.connected_accounts.connect_handlers import (
    ConnectNeedsCode,
    ConnectRefused,
    ConnectRequest,
    connect_phone_account,
)
from src.anubis.utils.connected_accounts.providers import PHONE_PROVIDER, get_provider
from src.anubis.utils.connected_accounts.store import public_account_view
from src.anubis.utils.connected_accounts.tool_factories import tool_names_for
from src.anubis.utils.inbox.repository import NOTIFY_ONLY_SOURCE_KINDS
from src.anubis.utils.phone.dispatch import inbound_reject_message, owner_for_inbound_caller
from src.anubis.utils.phone.enterprise import (
    PrivateNumberRefused,
    refuse_private_number_unless_enterprise,
    wants_private_number,
)
from src.anubis.utils.phone.extract import extract_order_result
from src.anubis.utils.phone.livekit_sip import ring_owner_then_destination
from src.anubis.utils.phone.mcp_server import handle_phone_mcp_request
from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164
from src.anubis.utils.phone.owner_location import write_owner_location
from src.anubis.utils.phone.places import (
    clear_place_cache,
    lookup_local_place,
    write_place_cache,
)
from src.anubis.utils.phone.repository import InMemoryPhoneCallRepository
from src.anubis.utils.phone.results import write_phone_call_result
from src.anubis.utils.phone.tools import (
    LOOKUP_TRAVEL_TOOL_NAMES,
    SIP_TOOL_NAMES,
    build_personal_avatar_place_tools,
    build_phone_sip_tools,
    phone_tools_as_mcp_descriptors,
)
from src.anubis.utils.phone.travel import clear_travel_cache, estimate_travel, resolve_origin
from src.anubis.utils.phone.verify import (
    check_verification_code,
    clear_verification_codes,
    store_verification_code,
)
from src.anubis.utils.phone.worker import finish_outbound_call, outbound_order_script
from src.api import webapp as webapp_module


ASSISTANT_ID = "assistant-personal"
USER_ID = "auth0-owner"
OWNER_MOBILE = "+14045550100"


class _MemoryStore:
    def __init__(self) -> None:
        self.items: dict[tuple, dict] = {}

    async def aput(self, namespace, key, value):
        self.items[(tuple(namespace), key)] = value

    async def aget(self, namespace, key):
        payload = self.items.get((tuple(namespace), key))
        if payload is None:
            return None
        return SimpleNamespace(value=payload, key=key)

    async def asearch(self, namespace, query=None, limit=20):
        prefix = tuple(namespace)
        found = []
        for (stored_namespace, stored_key), payload in self.items.items():
            if stored_namespace == prefix:
                found.append(SimpleNamespace(value=payload, key=stored_key))
        return found[:limit]


def _context(**overrides):
    values = dict(
        phone_verify_allow_undelivered="TRUE",
        nominatim_base_url="https://nominatim.test",
        overpass_base_url="https://overpass.test",
        place_lookup_cache_ttl_seconds=604800,
        google_places_api_key="",
        openroute_api_key="",
        travel_origin_max_age_seconds=1800,
        platform_phone_number="+18005550199",
        livekit_url="",
        livekit_api_key="",
        livekit_api_secret="",
        livekit_sip_outbound_trunk_id="",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _phone_record(mobile: str = OWNER_MOBILE) -> dict:
    return {
        "kind": "telephony",
        "user_id": USER_ID,
        "assistant_id": ASSISTANT_ID,
        "account_address": mobile,
        "transport": {"owner_mobile_e164": mobile, "sip_enabled": True},
    }


@pytest.fixture(autouse=True)
def _clean_phone_state():
    clear_place_cache()
    clear_travel_cache()
    clear_verification_codes()
    yield
    clear_place_cache()
    clear_travel_cache()
    clear_verification_codes()


def test_phone_is_in_the_catalog_not_as_a_device_row():
    provider = get_provider("phone")
    assert provider is PHONE_PROVIDER
    assert provider.kind == "telephony"
    assert provider.category == "phone"
    assert provider.credential_mechanism == "phone_verify"
    assert provider.uses_form is True


def test_sip_tool_names_do_not_include_place_call():
    assert "place_call" not in SIP_TOOL_NAMES
    assert "place_call" not in LOOKUP_TRAVEL_TOOL_NAMES
    names = [row["name"] for row in phone_tools_as_mcp_descriptors(True)]
    assert "place_call" not in names
    assert set(SIP_TOOL_NAMES) <= set(names)


def test_ios_place_call_still_only_opens_the_system_dialer():
    mobile = Path(__file__).resolve().parents[3] / (
        "f-anubis-mcp-server-mobile/AnubisMCP/Tools/CommsTools.swift"
    )
    source = mobile.read_text(encoding="utf-8")
    assert "func place_call" in source or "place_call" in source
    assert "tel://" in source
    assert "CallKit" not in source
    assert "CreateSIPParticipant" not in source


def test_lookup_and_travel_attach_without_a_phone_connection():
    tools = build_personal_avatar_place_tools(_context(), user_id=USER_ID, assistant_id=ASSISTANT_ID)
    assert {tool.name for tool in tools} == set(LOOKUP_TRAVEL_TOOL_NAMES)
    assert build_phone_sip_tools(_context(), []) == []


def test_sip_tools_attach_only_after_phone_is_connected():
    tools = build_phone_sip_tools(_context(), [_phone_record()])
    assert {tool.name for tool in tools} == set(SIP_TOOL_NAMES)
    assert tool_names_for(PHONE_PROVIDER, _phone_record()) == list(SIP_TOOL_NAMES)


def test_non_enterprise_private_number_is_refused():
    assert wants_private_number({"want_private_number": True}) is True
    with pytest.raises(PrivateNumberRefused):
        refuse_private_number_unless_enterprise("pro")
    with pytest.raises(PrivateNumberRefused):
        refuse_private_number_unless_enterprise("free")
    refuse_private_number_unless_enterprise("premium")


@pytest.mark.asyncio
async def test_connect_writes_owner_mobile_and_never_a_per_user_did():
    store_verification_code(OWNER_MOBILE, "123456")
    record = await connect_phone_account(
        ConnectRequest(
            provider=PHONE_PROVIDER,
            fields={"phone_number": "4045550100", "verification_code": "123456"},
            assistant_id=ASSISTANT_ID,
            context=_context(),
        )
    )
    assert record["transport"]["owner_mobile_e164"] == OWNER_MOBILE
    assert record["transport"]["sip_enabled"] is True
    assert "platform_number_e164" not in record["transport"]
    view = public_account_view(record)
    assert view["owner_mobile_e164"] == OWNER_MOBILE
    assert "platform_number_e164" not in view
    assert view.get("encrypted_secret") is None


@pytest.mark.asyncio
async def test_connect_first_step_sends_a_code():
    with pytest.raises(ConnectNeedsCode) as raised:
        await connect_phone_account(
            ConnectRequest(
                provider=PHONE_PROVIDER,
                fields={"phone_number": OWNER_MOBILE},
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert raised.value.as_response()["action"] == "enter_verification_code"


@pytest.mark.asyncio
async def test_connect_refuses_a_private_number_on_pro():
    with pytest.raises(ConnectRefused) as raised:
        await connect_phone_account(
            ConnectRequest(
                provider=PHONE_PROVIDER,
                fields={
                    "phone_number": OWNER_MOBILE,
                    "want_private_number": True,
                    "subscription_tier": "pro",
                },
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert raised.value.status_code == 403


def test_normalize_e164_nanp():
    assert normalize_e164("4045550100") == OWNER_MOBILE
    with pytest.raises(PhoneNumberError):
        normalize_e164("")


@pytest.mark.asyncio
async def test_inbound_dispatch_is_caller_id_to_owner():
    class _Repo:
        async def list_by_kind(self, kind, status="connected"):
            assert kind == "telephony"
            return [_phone_record()]

    found = await owner_for_inbound_caller(OWNER_MOBILE, repository=_Repo())
    assert found is not None
    assert found["transport"]["owner_mobile_e164"] == OWNER_MOBILE
    missing = await owner_for_inbound_caller("+14045550999", repository=_Repo())
    assert missing is None
    assert "verified" in inbound_reject_message()


@pytest.mark.asyncio
async def test_outbound_rings_owner_then_the_restaurant(monkeypatch):
    order: list[str] = []

    async def _fake_create(context, *, room_name, phone_e164, participant_identity, participant_name):
        order.append(phone_e164)
        return {"identity": participant_identity}

    monkeypatch.setattr(
        "src.anubis.utils.phone.livekit_sip.create_sip_participant", _fake_create
    )
    monkeypatch.setattr(
        "src.anubis.utils.phone.livekit_sip.livekit_is_configured", lambda context: True
    )
    result = await ring_owner_then_destination(
        _context(),
        room_name="phone-1",
        owner_mobile_e164=OWNER_MOBILE,
        destination_e164="+14045550111",
    )
    assert order == [OWNER_MOBILE, "+14045550111"]
    assert result["order"] == ["owner_mobile_e164", "destination_e164"]


@pytest.mark.asyncio
async def test_lookup_cache_hit_and_refuse_when_no_phone():
    write_place_cache(
        "Kanji",
        "Atlanta",
        {
            "name": "Kanji",
            "city": "Atlanta",
            "phone_e164": None,
            "address": "",
            "source": "not_found",
        },
        600,
    )
    place = await lookup_local_place("Kanji", "Atlanta", _context())
    assert place["cached"] is True
    assert place["phone_e164"] is None


@pytest.mark.asyncio
async def test_lookup_fills_phone_from_overpass(monkeypatch):
    async def _nominatim(name, city, context):
        return {"display_name": "Kanji, Atlanta", "lat": "33.7", "lon": "-84.3", "osm_type": "node", "osm_id": 1}

    async def _overpass(kind, osm_id, context):
        return {"phone": "+1 404 555 0188", "opening_hours": "Mo-Su 11:00-22:00"}

    monkeypatch.setattr("src.anubis.utils.phone.places._nominatim_search", _nominatim)
    monkeypatch.setattr("src.anubis.utils.phone.places._overpass_tags", _overpass)
    place = await lookup_local_place("Kanji", "Atlanta", _context())
    assert place["phone_e164"] == "+14045550188"
    assert place["source"] == "nominatim_overpass"


@pytest.mark.asyncio
async def test_travel_origin_order_stated_then_stored_then_visit_then_needed():
    store = _MemoryStore()
    context = _context()
    stated = await resolve_origin(
        "33.75,-84.39",
        store=store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        context=context,
    )
    assert stated["origin_source"] == "stated"

    await write_owner_location(store, USER_ID, latitude=33.74, longitude=-84.38)
    stored = await resolve_origin(
        None, store=store, user_id=USER_ID, assistant_id=ASSISTANT_ID, context=context
    )
    assert stored["origin_source"] == "owner_location"

    empty = _MemoryStore()
    visited_at = datetime.now(UTC).isoformat()
    await empty.aput(
        (USER_ID, ASSISTANT_ID, "geo_visit"),
        visited_at,
        {"value": {"latitude": 33.73, "longitude": -84.37, "visited_at": visited_at}},
    )
    visit = await resolve_origin(
        None, store=empty, user_id=USER_ID, assistant_id=ASSISTANT_ID, context=context
    )
    assert visit["origin_source"] == "geo_visit"

    needed = await resolve_origin(
        None, store=_MemoryStore(), user_id=USER_ID, assistant_id=ASSISTANT_ID, context=context
    )
    assert needed["origin_needed"] is True


@pytest.mark.asyncio
async def test_travel_uses_openrouteservice_not_distance_matrix(monkeypatch):
    seen: list[str] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"routes": [{"summary": {"duration": 600, "distance": 3200}}]}

    class _Client:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers=None, json=None):
            seen.append(url)
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    result = await estimate_travel(
        "Kanji",
        _context(openroute_api_key="ors-test"),
        origin="33.75,-84.39",
        destination_latitude=33.76,
        destination_longitude=-84.40,
    )
    assert seen
    assert "openrouteservice.org" in seen[0]
    assert "distancematrix" not in seen[0].lower()
    assert result["routing"] == "openrouteservice"
    assert result["duration_seconds"] == 600


def test_worker_extracts_kanji_and_mellow_mushroom_transcripts():
    kanji = extract_order_result(
        "Your large tonkotsu is $18.50 and will be ready in 20 minutes "
        "at 123 Peachtree Street, Atlanta.",
        destination_name="Kanji",
    )
    assert kanji["cost"] == 18.5
    assert "20" in kanji["ready_at"]
    assert "Peachtree" in kanji["location"]
    assert kanji["outcome"] == "placed"

    mellow = extract_order_result(
        "A large pepperoni from Mellow Mushroom is 22 dollars, pickup in 15 minutes "
        "at 456 Ponce de Leon Avenue.",
        destination_name="Mellow Mushroom",
    )
    assert mellow["cost"] == 22.0
    assert mellow["outcome"] == "placed"

    card = extract_order_result(
        "We will need a credit card number to start the order.",
        destination_name="Kanji",
    )
    assert card["outcome"] == "ended_payment_required"


@pytest.mark.asyncio
async def test_result_writer_produces_inbox_item_and_chat_fields(monkeypatch):
    from src.anubis.utils.inbox import repository as inbox_repo

    memory = inbox_repo.InMemoryInboxRepository()
    inbox_repo.set_inbox_repository(memory)
    try:
        written = await write_phone_call_result(
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            call_id="call-1",
            result={
                "cost": 18.5,
                "ready_at": "20 min",
                "location": "123 Peachtree Street",
                "travel_minutes": 12,
                "outcome": "placed",
                "destination_name": "Kanji",
            },
            transcript="Your large tonkotsu is $18.50.",
            thread_id="thread-1",
        )
        assert written["inbox_item"]["source_kind"] == "phone_call"
        assert "18.50" in written["chat_message"]["content"]
        assert written["chat_message"]["cost"] == 18.5
        assert written["chat_message"]["ready_at"] == "20 min"
        assert written["chat_message"]["location"] == "123 Peachtree Street"
        assert written["chat_message"]["travel_minutes"] == 12
        assert "phone_call" in NOTIFY_ONLY_SOURCE_KINDS
    finally:
        inbox_repo.set_inbox_repository(None)


@pytest.mark.asyncio
async def test_finish_outbound_call_writes_result():
    from src.anubis.utils.inbox import repository as inbox_repo
    from src.anubis.utils.phone import repository as phone_repo

    calls = InMemoryPhoneCallRepository()
    phone_repo.set_phone_call_repository(calls)
    inbox_repo.set_inbox_repository(inbox_repo.InMemoryInboxRepository())
    try:
        await calls.create_call(
            {
                "call_id": "call-kanji",
                "user_id": USER_ID,
                "assistant_id": ASSISTANT_ID,
                "direction": "outbound",
                "destination_name": "Kanji",
            }
        )
        written = await finish_outbound_call(
            call_id="call-kanji",
            transcript="That will be $16 ready in 25 minutes at 10 Main Street.",
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            destination_name="Kanji",
            travel_minutes=8,
        )
        stored = await calls.get_call("call-kanji")
        assert stored["state"] == "ended"
        assert stored["result"]["cost"] == 16.0
        assert written["chat_message"]["travel_minutes"] == 8
    finally:
        phone_repo.set_phone_call_repository(None)
        inbox_repo.set_inbox_repository(None)


@pytest.mark.asyncio
async def test_mcp_phone_server_hides_sip_until_connected():
    listed = await handle_phone_mcp_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        context=_context(),
        store=None,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        accounts=[],
        is_personal_avatar=True,
    )
    names = [row["name"] for row in listed["result"]["tools"]]
    assert "lookup_local_place" in names
    assert "estimate_travel" in names
    assert "request_outbound_phone_call" not in names

    refused = await handle_phone_mcp_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        context=_context(),
        store=None,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        accounts=[],
        is_personal_avatar=False,
    )
    assert refused["error"]["code"] == -32000

    unknown = await handle_phone_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "place_call", "arguments": {}},
        },
        context=_context(),
        store=None,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        accounts=[_phone_record()],
        is_personal_avatar=True,
    )
    assert "place_call" in unknown["error"]["message"]


def test_outbound_script_never_asks_for_a_card():
    script = outbound_order_script(
        {"item": "large pepperoni", "pickup_name": "Evan"}
    )
    assert "pay at the counter" in script.casefold()
    assert "card number" in script.casefold()
    assert "do not take a card" in script.casefold()


@pytest.mark.asyncio
async def test_outbound_refuses_when_confirm_is_not_accepted(monkeypatch):
    from src.anubis.utils.phone import repository as phone_repo

    phone_repo.set_phone_call_repository(InMemoryPhoneCallRepository())

    def _interrupt(payload):
        assert payload["kind"] == "phone_call_confirm"
        return {"type": "cancel"}

    monkeypatch.setattr("langgraph.types.interrupt", _interrupt)
    tools = build_phone_sip_tools(_context(), [_phone_record()])
    request_tool = next(tool for tool in tools if tool.name == "request_outbound_phone_call")
    result = await request_tool.ainvoke(
        {
            "destination_name": "Kanji",
            "destination_phone_e164": "+14045550188",
            "item": "tonkotsu",
            "confirm": False,
        }
    )
    assert result["status"] == "cancelled"
    phone_repo.set_phone_call_repository(None)


def test_phone_call_frames_are_forwarded_to_the_browser():
    assert "phone_call" in webapp_module.BROWSER_DIRECTED_FRAMES


def test_lookup_and_travel_never_import_mobile_mcp_get_location():
    import src.anubis.utils.phone.places as places
    import src.anubis.utils.phone.travel as travel

    for module in (places, travel):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "mcp/relay" not in source
        assert "from src.anubis.utils.tools.data_analysis" not in source
        assert "invoke(\"get_location\"" not in source
        assert "place_call(" not in source
