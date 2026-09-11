"""Adding a calendar with an address and a password, and using it.

Offline throughout: a mock transport answers the WebDAV requests exactly as a
server would, so discovery order, the conditional headers that stop one change
from silently overwriting another, and the iCalendar text itself can all be
asserted directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import httpx
import pytest

from src.anubis.utils.connected_accounts.caldav_client import (
    CalDavAccount,
    CalDavAuthenticationError,
    CalDavUnreachableError,
    CalendarEvent,
    build_icalendar,
    connect_caldav_account,
    create_event,
    delete_event,
    find_free_time,
    list_calendars,
    list_events,
    parse_icalendar,
    update_event,
)

ADDRESS = "evan@example.com"
PASSWORD = "an account password"
HOME = "https://example.com/calendars/evan/"

PRINCIPAL_RESPONSE = """<?xml version="1.0"?>
<d:multistatus xmlns:d="DAV:">
  <d:response>
    <d:href>/.well-known/caldav</d:href>
    <d:propstat><d:prop>
      <d:current-user-principal><d:href>/principals/evan/</d:href></d:current-user-principal>
    </d:prop></d:propstat>
  </d:response>
</d:multistatus>"""

HOME_RESPONSE = """<?xml version="1.0"?>
<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:response>
    <d:href>/principals/evan/</d:href>
    <d:propstat><d:prop>
      <c:calendar-home-set><d:href>/calendars/evan/</d:href></c:calendar-home-set>
    </d:prop></d:propstat>
  </d:response>
</d:multistatus>"""

CALENDAR_LIST_RESPONSE = """<?xml version="1.0"?>
<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"
               xmlns:a="http://apple.com/ns/ical/">
  <d:response>
    <d:href>/calendars/evan/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat>
  </d:response>
  <d:response>
    <d:href>/calendars/evan/personal/</d:href>
    <d:propstat><d:prop>
      <d:resourcetype><d:collection/><c:calendar/></d:resourcetype>
      <d:displayname>Personal</d:displayname>
      <a:calendar-color>#FF0000</a:calendar-color>
      <d:current-user-privilege-set>
        <d:privilege><d:read/></d:privilege>
        <d:privilege><d:write/></d:privilege>
      </d:current-user-privilege-set>
    </d:prop></d:propstat>
  </d:response>
  <d:response>
    <d:href>/calendars/evan/holidays/</d:href>
    <d:propstat><d:prop>
      <d:resourcetype><d:collection/><c:calendar/></d:resourcetype>
      <d:displayname>Holidays</d:displayname>
      <d:current-user-privilege-set>
        <d:privilege><d:read/></d:privilege>
      </d:current-user-privilege-set>
    </d:prop></d:propstat>
  </d:response>
</d:multistatus>"""


def _events_response(*events: str) -> str:
    entries = "".join(
        f"""<d:response><d:href>/calendars/evan/personal/{index}.ics</d:href>
        <d:propstat><d:prop><d:getetag>"etag-{index}"</d:getetag>
        <c:calendar-data>{body}</c:calendar-data></d:prop></d:propstat></d:response>"""
        for index, body in enumerate(events)
    )
    return (
        '<?xml version="1.0"?><d:multistatus xmlns:d="DAV:" '
        f'xmlns:c="urn:ietf:params:xml:ns:caldav">{entries}</d:multistatus>'
    )


def _event_text(uid: str, summary: str, start: str, end: str) -> str:
    return (
        "BEGIN:VCALENDAR\nBEGIN:VEVENT\n"
        f"UID:{uid}\nSUMMARY:{summary}\nDTSTART:{start}\nDTEND:{end}\n"
        "END:VEVENT\nEND:VCALENDAR"
    )


def _server(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def _account() -> CalDavAccount:
    return CalDavAccount(
        base_url="https://example.com/",
        username=ADDRESS,
        password=PASSWORD,
        principal_url="https://example.com/principals/evan/",
        calendar_home_url=HOME,
    )


def _run(coroutine):
    return asyncio.run(coroutine)


# -- iCalendar -------------------------------------------------------------


def test_punctuation_in_a_summary_survives_a_round_trip():
    """An unescaped comma or semicolon corrupts the whole object."""
    event = CalendarEvent(
        summary="Call with Ana, Inc.; re: budget",
        location="Room 3; floor 2",
        description="Line one\nline two",
        starts_at=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
        ends_at=datetime(2026, 9, 10, 15, 30, tzinfo=UTC),
    )

    parsed = parse_icalendar(build_icalendar(event))[0]

    assert parsed.summary == "Call with Ana, Inc.; re: budget"
    assert parsed.location == "Room 3; floor 2"
    assert parsed.description == "Line one\nline two"


def test_a_long_line_is_folded_within_the_octet_limit_and_reads_back():
    event = CalendarEvent(
        summary="x" * 300, starts_at=datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
    )

    text = build_icalendar(event)

    for line in text.split("\r\n"):
        assert len(line.encode("utf-8")) <= 75
    assert parse_icalendar(text)[0].summary == "x" * 300


def test_an_all_day_appointment_uses_a_date_not_a_timestamp():
    event = CalendarEvent(
        summary="Holiday", starts_at=date(2026, 9, 10), ends_at=date(2026, 9, 11)
    )

    text = build_icalendar(event)

    assert "DTSTART;VALUE=DATE:20260910" in text
    assert parse_icalendar(text)[0].starts_at == date(2026, 9, 10)


def test_attendees_and_organizer_round_trip():
    event = CalendarEvent(
        summary="Review",
        starts_at=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
        organizer="evan@example.com",
        attendees=["ana@example.com", "sam@example.com"],
    )

    parsed = parse_icalendar(build_icalendar(event))[0]

    assert parsed.organizer == "evan@example.com"
    assert parsed.attendees == ["ana@example.com", "sam@example.com"]


# -- discovery -------------------------------------------------------------


def test_an_address_and_password_find_the_calendar_home():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        if str(request.url).startswith("https://dns.google/"):
            return httpx.Response(200, json={})
        if request.method == "PROPFIND" and "well-known" in str(request.url):
            return httpx.Response(207, text=PRINCIPAL_RESPONSE)
        if request.method == "PROPFIND" and "principals" in str(request.url):
            return httpx.Response(207, text=HOME_RESPONSE)
        return httpx.Response(404)

    async def run():
        async with _server(handler) as client:
            return await connect_caldav_account(
                email_address=ADDRESS, password=PASSWORD, http_client=client
            )

    account = _run(run())

    assert account.calendar_home_url == HOME
    assert account.principal_url == "https://example.com/principals/evan/"
    assert account.auth == (ADDRESS, PASSWORD)


def test_a_refused_password_is_reported_as_a_refused_password():
    """And is not retried against the next candidate with the same password."""
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://dns.google/"):
            return httpx.Response(200, json={})
        attempts.append(str(request.url))
        return httpx.Response(401)

    async def run():
        async with _server(handler) as client:
            return await connect_caldav_account(
                email_address=ADDRESS, password=PASSWORD, http_client=client
            )

    with pytest.raises(CalDavAuthenticationError):
        _run(run())

    assert len(attempts) == 1


def test_a_domain_with_no_calendar_server_says_so():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://dns.google/"):
            return httpx.Response(200, json={})
        return httpx.Response(404)

    async def run():
        async with _server(handler) as client:
            return await connect_caldav_account(
                email_address=ADDRESS, password=PASSWORD, http_client=client
            )

    with pytest.raises(CalDavUnreachableError):
        _run(run())


def test_calendars_are_listed_with_their_names_and_write_access():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, text=CALENDAR_LIST_RESPONSE)

    async def run():
        async with _server(handler) as client:
            return await list_calendars(_account(), http_client=client)

    calendars = _run(run())

    # The calendar home itself is a plain collection and must not be listed.
    assert [calendar.display_name for calendar in calendars] == ["Personal", "Holidays"]
    assert calendars[0].url == "https://example.com/calendars/evan/personal/"
    assert calendars[0].color == "#FF0000"
    assert calendars[0].read_only is False
    # A calendar the owner may only read must not be offered as a place to book.
    assert calendars[1].read_only is True


# -- reading and writing ---------------------------------------------------


def test_events_are_read_in_order_with_their_addresses():
    body = _events_response(
        _event_text("2", "Later", "20260910T160000Z", "20260910T163000Z"),
        _event_text("1", "Earlier", "20260910T090000Z", "20260910T093000Z"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "REPORT"
        assert b"time-range" in request.content
        return httpx.Response(207, text=body)

    async def run():
        async with _server(handler) as client:
            return await list_events(
                _account(),
                f"{HOME}personal/",
                since=datetime(2026, 9, 10, tzinfo=UTC),
                until=datetime(2026, 9, 11, tzinfo=UTC),
                http_client=client,
            )

    events = _run(run())

    assert [event.summary for event in events] == ["Earlier", "Later"]
    assert events[0].url == "https://example.com/calendars/evan/personal/1.ics"
    assert events[0].etag == '"etag-1"'


def test_creating_an_appointment_refuses_to_overwrite_an_existing_one():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["if_none_match"] = request.headers.get("if-none-match")
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(201, headers={"etag": '"new"'})

    event = CalendarEvent(
        summary="Call with Ana",
        starts_at=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
        ends_at=datetime(2026, 9, 10, 15, 30, tzinfo=UTC),
        uid="appointment-1",
    )

    async def run():
        async with _server(handler) as client:
            return await create_event(
                _account(), f"{HOME}personal/", event, http_client=client
            )

    created = _run(run())

    assert captured["method"] == "PUT"
    assert captured["url"] == f"{HOME}personal/appointment-1.ics"
    assert captured["if_none_match"] == "*"
    assert "SUMMARY:Call with Ana" in captured["body"]
    assert created.etag == '"new"'
    assert created.url == f"{HOME}personal/appointment-1.ics"


def test_updating_an_appointment_only_replaces_the_version_that_was_read():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["if_match"] = request.headers.get("if-match")
        return httpx.Response(204, headers={"etag": '"second"'})

    event = CalendarEvent(
        summary="Moved",
        starts_at=datetime(2026, 9, 11, 15, 0, tzinfo=UTC),
        uid="appointment-1",
        url=f"{HOME}personal/appointment-1.ics",
        etag='"first"',
    )

    async def run():
        async with _server(handler) as client:
            return await update_event(_account(), event, http_client=client)

    updated = _run(run())

    assert captured["if_match"] == '"first"'
    assert updated.etag == '"second"'


def test_deleting_an_appointment_sends_the_delete():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(204)

    async def run():
        async with _server(handler) as client:
            await delete_event(
                _account(), f"{HOME}personal/appointment-1.ics", http_client=client
            )

    _run(run())

    assert captured["method"] == "DELETE"
    assert captured["url"] == f"{HOME}personal/appointment-1.ics"


def test_free_time_skips_the_busy_stretches_and_merges_overlaps():
    body = _events_response(
        _event_text("1", "A", "20260910T090000Z", "20260910T100000Z"),
        # Overlaps the first: the pair is one busy stretch, not two.
        _event_text("2", "B", "20260910T093000Z", "20260910T110000Z"),
        _event_text("3", "C", "20260910T130000Z", "20260910T133000Z"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, text=body)

    async def run():
        async with _server(handler) as client:
            return await find_free_time(
                _account(),
                [f"{HOME}personal/"],
                since=datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
                until=datetime(2026, 9, 10, 17, 0, tzinfo=UTC),
                duration_minutes=60,
                http_client=client,
            )

    openings = _run(run())

    assert openings == [
        {"start": "2026-09-10T08:00:00+00:00", "end": "2026-09-10T09:00:00+00:00"},
        # 09:00-11:00 is one busy stretch, not two: the overlapping pair merged.
        {"start": "2026-09-10T11:00:00+00:00", "end": "2026-09-10T13:00:00+00:00"},
        {"start": "2026-09-10T13:30:00+00:00", "end": "2026-09-10T17:00:00+00:00"},
    ]
    # The 08:00-09:00 gap is exactly one hour and qualifies; a 90-minute
    # requirement must drop it rather than round it up.
    async def run_ninety():
        async with _server(handler) as client:
            return await find_free_time(
                _account(),
                [f"{HOME}personal/"],
                since=datetime(2026, 9, 10, 8, 0, tzinfo=UTC),
                until=datetime(2026, 9, 10, 17, 0, tzinfo=UTC),
                duration_minutes=90,
                http_client=client,
            )

    assert all(
        opening["start"] != "2026-09-10T08:00:00+00:00" for opening in _run(run_ninety())
    )
