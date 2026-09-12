"""Booking on a Google Calendar connected through this API's own OAuth client.

Calendar scopes are *sensitive*, not restricted, so this path needs Google's
ordinary verification and no security assessment — which is why it is the
Google connector that runs on our own client rather than through anyone else's.

The six verbs here are deliberately the same six the CalDAV path exposes, so
the prompts, the inbox, and the conversation do not care which kind of calendar
the owner connected.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils.connected_accounts import vendor_api_tools

ADDRESS = "evan@example.com"


def _record():
    return {
        "account_key": f"google_calendar:{ADDRESS}",
        "provider": "google_calendar",
        "kind": "calendar",
        "credential_mechanism": "oauth",
        "account_address": ADDRESS,
        "display_label": "evan",
    }


def _tools(monkeypatch, *, get=None, post=None, request=None):
    async def _bearer(context, store, record):
        return "token-1", None

    monkeypatch.setattr(vendor_api_tools, "_bearer", _bearer)
    if get is not None:
        monkeypatch.setattr(vendor_api_tools, "_get_json", get)
    if post is not None:
        monkeypatch.setattr(vendor_api_tools, "_post_json", post)
    if request is not None:
        monkeypatch.setattr(vendor_api_tools, "_request_json", request)
    built = vendor_api_tools.build_vendor_api_tools(
        SimpleNamespace(), [_record()], store=None, pool=None
    )
    return {tool.name: tool for tool in built}


def test_the_calendar_presents_the_same_six_verbs_as_a_caldav_calendar(monkeypatch):
    tools = _tools(monkeypatch)

    assert {
        "list_calendars",
        "calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        "find_free_time",
    } <= set(tools)


def test_calendars_report_which_ones_can_hold_an_appointment(monkeypatch):
    async def _get(url, token, *, headers=None, params=None, timeout=20.0):
        return 200, {
            "items": [
                {"id": "primary", "summary": "Evan", "primary": True, "accessRole": "owner"},
                {"id": "team@x.com", "summary": "Team", "accessRole": "reader"},
                {"id": "busy@x.com", "summary": "Busy", "accessRole": "freeBusyReader"},
            ]
        }

    tools = _tools(monkeypatch, get=_get)

    result = asyncio.run(tools["list_calendars"].coroutine())

    assert result["status"] == "ok"
    assert [calendar["read_only"] for calendar in result["calendars"]] == [
        False,
        True,
        True,
    ]


def test_booking_sends_the_times_and_returns_the_event(monkeypatch):
    captured: dict[str, object] = {}

    async def _post(url, token, body, *, headers=None, timeout=20.0):
        captured["url"] = url
        captured["body"] = body
        return 200, {
            "id": "event-1",
            "summary": body["summary"],
            "start": body["start"],
            "end": body["end"],
            "htmlLink": "https://calendar.google.com/event?eid=1",
        }

    tools = _tools(monkeypatch, post=_post)

    result = asyncio.run(
        tools["create_calendar_event"].coroutine(
            summary="Call with Ana",
            start="2026-09-10T15:00:00Z",
            end="2026-09-10T15:30:00Z",
            attendees=["ana@example.com"],
        )
    )

    assert result["status"] == "created"
    assert result["event"]["id"] == "event-1"
    assert result["event"]["url"].startswith("https://calendar.google.com/")
    body = captured["body"]
    assert body["start"]["dateTime"].startswith("2026-09-10T15:00")
    assert body["attendees"] == [{"email": "ana@example.com"}]


def test_a_bare_date_books_a_whole_day(monkeypatch):
    captured: dict[str, object] = {}

    async def _post(url, token, body, *, headers=None, timeout=20.0):
        captured["body"] = body
        return 200, {"id": "event-1", "start": body["start"], "end": body["end"]}

    tools = _tools(monkeypatch, post=_post)

    asyncio.run(tools["create_calendar_event"].coroutine(summary="Holiday", start="2026-09-10"))

    # A whole day is a date, not a timestamp, and it ends on the following day.
    assert captured["body"]["start"] == {"date": "2026-09-10"}
    assert captured["body"]["end"] == {"date": "2026-09-11"}


def test_an_appointment_with_no_end_lasts_an_hour(monkeypatch):
    captured: dict[str, object] = {}

    async def _post(url, token, body, *, headers=None, timeout=20.0):
        captured["body"] = body
        return 200, {"id": "event-1", "start": body["start"], "end": body["end"]}

    tools = _tools(monkeypatch, post=_post)

    asyncio.run(
        tools["create_calendar_event"].coroutine(
            summary="Call", start="2026-09-10T15:00:00Z"
        )
    )

    assert captured["body"]["end"]["dateTime"].startswith("2026-09-10T16:00")


def test_an_unreadable_time_is_refused_before_anything_is_booked(monkeypatch):
    async def _post(url, token, body, *, headers=None, timeout=20.0):
        raise AssertionError("nothing may be booked from an unreadable time")

    tools = _tools(monkeypatch, post=_post)

    result = asyncio.run(
        tools["create_calendar_event"].coroutine(summary="Call", start="next Thursday-ish")
    )

    assert result["status"] == "error"
    assert "next Thursday-ish" in result["error"]


def test_changing_an_appointment_sends_only_what_changed(monkeypatch):
    captured: dict[str, object] = {}

    async def _request(method, url, token, *, body=None, params=None, timeout=20.0):
        captured["method"] = method
        captured["url"] = url
        captured["body"] = body
        return 200, {"id": "event-1", "summary": "Moved", "start": {}, "end": {}}

    tools = _tools(monkeypatch, request=_request)

    result = asyncio.run(
        tools["update_calendar_event"].coroutine(event_id="event-1", summary="Moved")
    )

    assert result["status"] == "updated"
    assert captured["method"] == "PATCH"
    # Only the named field travels, so nothing else on the appointment is lost.
    assert captured["body"] == {"summary": "Moved"}
    assert captured["url"].endswith("/events/event-1")


def test_changing_nothing_is_refused_rather_than_sent(monkeypatch):
    async def _request(method, url, token, *, body=None, params=None, timeout=20.0):
        raise AssertionError("an empty change must not reach the calendar")

    tools = _tools(monkeypatch, request=_request)

    result = asyncio.run(tools["update_calendar_event"].coroutine(event_id="event-1"))

    assert result["status"] == "error"
    assert "at least one thing" in result["error"]


def test_deleting_an_appointment_uses_the_delete_verb(monkeypatch):
    captured: dict[str, object] = {}

    async def _request(method, url, token, *, body=None, params=None, timeout=20.0):
        captured["method"] = method
        captured["url"] = url
        return 204, {}

    tools = _tools(monkeypatch, request=_request)

    result = asyncio.run(tools["delete_calendar_event"].coroutine(event_id="event-1"))

    assert result["status"] == "deleted"
    assert captured["method"] == "DELETE"


def test_a_calendar_identifier_with_an_at_sign_is_escaped_into_the_path(monkeypatch):
    """An unescaped address in a URL path is a request sent to the wrong place."""
    captured: dict[str, object] = {}

    async def _post(url, token, body, *, headers=None, timeout=20.0):
        captured["url"] = url
        return 200, {"id": "event-1", "start": {}, "end": {}}

    tools = _tools(monkeypatch, post=_post)

    asyncio.run(
        tools["create_calendar_event"].coroutine(
            summary="Call", start="2026-09-10T15:00:00Z", calendar="team@example.com"
        )
    )

    assert "team%40example.com" in captured["url"]


def test_free_time_skips_the_busy_stretches_and_merges_overlaps(monkeypatch):
    async def _post(url, token, body, *, headers=None, timeout=20.0):
        return 200, {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": "2026-09-10T09:00:00Z", "end": "2026-09-10T10:00:00Z"},
                        # Overlaps the first: one busy stretch, not two.
                        {"start": "2026-09-10T09:30:00Z", "end": "2026-09-10T11:00:00Z"},
                    ]
                }
            }
        }

    tools = _tools(monkeypatch, post=_post)

    result = asyncio.run(
        tools["find_free_time"].coroutine(
            since="2026-09-10T08:00:00Z",
            until="2026-09-10T13:00:00Z",
            duration_minutes=60,
        )
    )

    assert result["status"] == "ok"
    assert result["openings"] == [
        {"start": "2026-09-10T08:00:00+00:00", "end": "2026-09-10T09:00:00+00:00"},
        {"start": "2026-09-10T11:00:00+00:00", "end": "2026-09-10T13:00:00+00:00"},
    ]


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("list_calendars", {}),
        ("create_calendar_event", {"summary": "x", "start": "2026-09-10T15:00:00Z"}),
    ],
)
def test_a_refused_call_returns_a_sentence_rather_than_raising(
    monkeypatch, tool_name, arguments
):
    """A calendar that answers with an error must not end the avatar's turn."""

    async def _get(url, token, *, headers=None, params=None, timeout=20.0):
        return 403, {"error": "insufficient permissions"}

    async def _post(url, token, body, *, headers=None, timeout=20.0):
        return 403, {"error": "insufficient permissions"}

    tools = _tools(monkeypatch, get=_get, post=_post)

    result = asyncio.run(tools[tool_name].coroutine(**arguments))

    assert result["status"] == "error"
    assert result["status_code"] == 403
