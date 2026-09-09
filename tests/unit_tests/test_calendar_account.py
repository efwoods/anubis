"""Connecting a calendar with an address and a password, and booking on it.

The client itself is covered in ``test_caldav_client.py``. What is asserted
here is the layer the owner and the avatar actually meet: that connecting
stores a usable record and no plaintext, that a refused password and an absent
server are different answers, and that the tools never write to a calendar the
account may only read.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import caldav_client, get_provider
from src.anubis.utils.connected_accounts.caldav_client import (
    CalDavAccount,
    CalDavAuthenticationError,
    CalDavUnreachableError,
    CalendarCollection,
    CalendarEvent,
)
from src.anubis.utils.connected_accounts.caldav_tools import build_caldav_tools
from src.anubis.utils.connected_accounts.connect_handlers import (
    ConnectRefused,
    ConnectRequest,
    connect_account,
)

ADDRESS = "evan@example.com"
PASSWORD = "an account password"
HOME = "https://example.com/calendars/evan/"
ASSISTANT_ID = "assistant-1"

PERSONAL = CalendarCollection(url=f"{HOME}personal/", display_name="Personal")
SHARED = CalendarCollection(
    url=f"{HOME}shared/", display_name="Team", read_only=True
)


def _context():
    return SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key()
    )


def _record(context):
    return {
        "account_key": f"calendar_account:{ADDRESS}",
        "provider": "calendar_account",
        "kind": "calendar",
        "credential_mechanism": "app_password",
        "account_address": ADDRESS,
        "display_label": "evan",
        "encrypted_secret": secret_store.encrypt_secret(PASSWORD, context),
        "transport": {
            "caldav": {
                "base_url": "https://example.com/",
                "principal_url": "https://example.com/principals/evan/",
                "calendar_home_url": HOME,
                "username": ADDRESS,
            }
        },
    }


def _tools(context, record):
    return {tool.name: tool for tool in build_caldav_tools(context, [record])}


def _connect(monkeypatch, context, **fields):
    request = ConnectRequest(
        provider=get_provider("calendar_account"),
        fields={"email_address": ADDRESS, "password": PASSWORD, **fields},
        assistant_id=ASSISTANT_ID,
        context=context,
    )
    return asyncio.run(connect_account(request))


def test_connecting_stores_the_proven_addresses_and_no_plaintext(monkeypatch):
    context = _context()

    async def _connect_caldav(**kwargs):
        assert kwargs["email_address"] == ADDRESS
        assert kwargs["password"] == PASSWORD
        return CalDavAccount(
            base_url="https://example.com/",
            username=ADDRESS,
            password=PASSWORD,
            principal_url="https://example.com/principals/evan/",
            calendar_home_url=HOME,
        )

    async def _list(account, **kwargs):
        return [PERSONAL, SHARED]

    monkeypatch.setattr(caldav_client, "connect_caldav_account", _connect_caldav)
    monkeypatch.setattr(caldav_client, "list_calendars", _list)

    record = _connect(monkeypatch, context)

    assert record["provider"] == "calendar_account"
    assert record["kind"] == "calendar"
    caldav = record["transport"]["caldav"]
    # Discovery ran once, at connect time; no later turn repeats it.
    assert caldav["calendar_home_url"] == HOME
    assert caldav["principal_url"] == "https://example.com/principals/evan/"
    assert [entry["name"] for entry in caldav["calendars"]] == ["Personal", "Team"]
    assert PASSWORD not in str(record)
    assert secret_store.decrypt_secret(record["encrypted_secret"], context) == PASSWORD


def test_a_refused_password_and_an_absent_server_are_different_answers(monkeypatch):
    context = _context()

    async def _refuse(**kwargs):
        raise CalDavAuthenticationError("401")

    monkeypatch.setattr(caldav_client, "connect_caldav_account", _refuse)
    with pytest.raises(ConnectRefused) as refused:
        _connect(monkeypatch, context)
    assert "rejected that password" in refused.value.detail

    async def _unreachable(**kwargs):
        raise CalDavUnreachableError("nothing answered")

    monkeypatch.setattr(caldav_client, "connect_caldav_account", _unreachable)
    with pytest.raises(ConnectRefused) as missing:
        _connect(monkeypatch, context)
    assert "calendar server address" in missing.value.detail


def test_the_calendars_are_listed_with_their_write_access(monkeypatch):
    context = _context()

    async def _list(account, **kwargs):
        return [PERSONAL, SHARED]

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    tools = _tools(context, _record(context))

    result = asyncio.run(tools["list_calendars"].coroutine())

    assert result["status"] == "ok"
    assert result["calendars"] == [
        {"name": "Personal", "read_only": False, "color": ""},
        {"name": "Team", "read_only": True, "color": ""},
    ]


def test_an_appointment_is_booked_on_a_writable_calendar(monkeypatch):
    context = _context()
    booked: dict[str, object] = {}

    async def _list(account, **kwargs):
        return [SHARED, PERSONAL]

    async def _create(account, calendar_url, event, **kwargs):
        booked["url"] = calendar_url
        booked["summary"] = event.summary
        event.url = f"{calendar_url}1.ics"
        return event

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    monkeypatch.setattr(caldav_client, "create_event", _create)
    tools = _tools(context, _record(context))

    result = asyncio.run(
        tools["create_calendar_event"].coroutine(
            summary="Call with Ana", start="2026-09-10T15:00:00Z", end="2026-09-10T15:30:00Z"
        )
    )

    assert result["status"] == "created"
    # The read-only calendar is listed first; a writable one must still be chosen.
    assert booked["url"] == PERSONAL.url
    assert result["calendar"] == "Personal"
    assert result["event"]["summary"] == "Call with Ana"


def test_a_read_only_calendar_is_refused_before_the_server_sees_it(monkeypatch):
    """The server would refuse this too, with a status code nobody can read."""
    context = _context()

    async def _list(account, **kwargs):
        return [SHARED]

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("nothing may be written to a read-only calendar")

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    monkeypatch.setattr(caldav_client, "create_event", _must_not_run)
    tools = _tools(context, _record(context))

    result = asyncio.run(
        tools["create_calendar_event"].coroutine(
            summary="Call", start="2026-09-10T15:00:00Z", calendar="Team"
        )
    )

    assert result["status"] == "read_only"
    assert "Team" in result["error"]


def test_a_bare_date_books_a_whole_day(monkeypatch):
    context = _context()
    captured: dict[str, object] = {}

    async def _list(account, **kwargs):
        return [PERSONAL]

    async def _create(account, calendar_url, event, **kwargs):
        captured["start"] = event.starts_at
        captured["end"] = event.ends_at
        return event

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    monkeypatch.setattr(caldav_client, "create_event", _create)
    tools = _tools(context, _record(context))

    asyncio.run(
        tools["create_calendar_event"].coroutine(summary="Holiday", start="2026-09-10")
    )

    assert str(captured["start"]) == "2026-09-10"
    assert str(captured["end"]) == "2026-09-11"


def test_an_unreadable_calendar_returns_a_sentence_not_an_exception(monkeypatch):
    """A calendar that is briefly down must not end the avatar's turn."""
    context = _context()

    async def _fail(account, **kwargs):
        raise CalDavUnreachableError("connection reset")

    monkeypatch.setattr(caldav_client, "list_calendars", _fail)
    tools = _tools(context, _record(context))

    result = asyncio.run(tools["calendar_events"].coroutine())

    assert result["status"] == "unreachable"
    assert "connection reset" in result["error"]


def test_a_rejected_stored_password_asks_for_a_reconnection(monkeypatch):
    context = _context()

    async def _reject(account, **kwargs):
        raise CalDavAuthenticationError("401")

    monkeypatch.setattr(caldav_client, "list_calendars", _reject)
    tools = _tools(context, _record(context))

    result = asyncio.run(tools["list_calendars"].coroutine())

    assert result["status"] == "needs_reconnect"
    assert "connect the calendar again" in result["error"]


def test_free_time_is_reported_for_every_calendar(monkeypatch):
    context = _context()
    asked: dict[str, object] = {}

    async def _list(account, **kwargs):
        return [PERSONAL, SHARED]

    async def _free(account, urls, **kwargs):
        asked["urls"] = list(urls)
        asked["duration"] = kwargs["duration_minutes"]
        return [{"start": "2026-09-10T13:00:00+00:00", "end": "2026-09-10T15:00:00+00:00"}]

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    monkeypatch.setattr(caldav_client, "find_free_time", _free)
    tools = _tools(context, _record(context))

    result = asyncio.run(tools["find_free_time"].coroutine(duration_minutes=45))

    assert result["status"] == "ok"
    # A read-only calendar still blocks time, so it must be consulted.
    assert asked["urls"] == [PERSONAL.url, SHARED.url]
    assert asked["duration"] == 45
    assert result["openings"][0]["start"] == "2026-09-10T13:00:00+00:00"


def test_the_events_of_every_calendar_are_merged_in_time_order(monkeypatch):
    context = _context()

    async def _list(account, **kwargs):
        return [PERSONAL, SHARED]

    async def _events(account, calendar_url, **kwargs):
        if calendar_url == PERSONAL.url:
            return [
                CalendarEvent(
                    summary="Later",
                    starts_at=datetime(2026, 9, 10, 16, tzinfo=UTC),
                )
            ]
        return [
            CalendarEvent(
                summary="Earlier",
                starts_at=datetime(2026, 9, 10, 9, tzinfo=UTC),
            )
        ]

    monkeypatch.setattr(caldav_client, "list_calendars", _list)
    monkeypatch.setattr(caldav_client, "list_events", _events)
    tools = _tools(context, _record(context))

    result = asyncio.run(tools["calendar_events"].coroutine())

    assert [event["summary"] for event in result["events"]] == ["Earlier", "Later"]
    assert [event["calendar"] for event in result["events"]] == ["Team", "Personal"]
