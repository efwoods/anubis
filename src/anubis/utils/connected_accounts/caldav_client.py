"""Read and write a calendar the way a desktop client does: CalDAV and a password.

The calendar half of "add any account with an address and a password". A mail
client finds a person's calendars from the same two things it uses for mail,
over CalDAV (RFC 4791), and so does this: Fastmail, Nextcloud, iCloud, Zimbra,
Zoho, and every self-hosted or corporate server speak it, with no vendor
application and nothing for the owner to generate.

Discovery follows RFC 6764 in the order a client is expected to try:

1. ``https://<domain>/.well-known/caldav``, following redirects — the answer a
   server is supposed to publish.
2. The ``_caldavs._tcp.<domain>`` SRV record, resolved over HTTPS so no resolver
   library is needed, exactly as ``mail_autoconfig`` does for mail.
3. The domain itself, which is where a surprising number of small servers live.

From whichever answers: ``PROPFIND`` for ``current-user-principal``, then that
principal's ``calendar-home-set``, then one depth-1 ``PROPFIND`` listing the
calendar collections inside it.

There is no CalDAV or iCalendar package in this image, and adding one would mean
a rebuilt base layer for a format that is a handful of lines of text. So the
iCalendar this module writes and reads is built here, deliberately narrowly: a
single ``VEVENT`` with the properties a person actually sets. Two details are
easy to get wrong and are handled once, here, rather than at each call site —
values are escaped (a comma, a semicolon, or a newline inside a summary
otherwise corrupts the whole object) and long lines are folded to 75 octets on
character boundaries, which is what the format requires and what several
servers enforce.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

DAV_NAMESPACE = "DAV:"
CALDAV_NAMESPACE = "urn:ietf:params:xml:ns:caldav"
APPLE_NAMESPACE = "http://apple.com/ns/ical/"

DNS_OVER_HTTPS_URL = "https://dns.google/resolve"
WELL_KNOWN_PATH = "/.well-known/caldav"

REQUEST_TIMEOUT_SECONDS = 20.0

# Line folding, as the format defines it.
MAX_LINE_OCTETS = 75


class CalDavError(Exception):
    """A calendar server refused or could not be reached."""


class CalDavAuthenticationError(CalDavError):
    """The server rejected the address and password."""


class CalDavUnreachableError(CalDavError):
    """The server could not be reached at all."""


@dataclass(frozen=True)
class CalendarCollection:
    """One calendar on a server."""

    url: str
    display_name: str
    color: str = ""
    read_only: bool = False


@dataclass
class CalendarEvent:
    """One appointment, in the terms a person would describe it.

    ``starts_at`` and ``ends_at`` are timezone-aware datetimes for a timed
    event, or ``date`` objects for an all-day one. Everything else is optional
    because an appointment with a title and a time is already a usable
    appointment.
    """

    summary: str = ""
    starts_at: datetime | date | None = None
    ends_at: datetime | date | None = None
    description: str = ""
    location: str = ""
    attendees: list[str] = field(default_factory=list)
    uid: str = ""
    url: str = ""
    etag: str = ""
    organizer: str = ""
    status: str = ""

    def as_public_dict(self) -> dict[str, Any]:
        """Return the shape a tool hands back to the model and the owner."""
        return {
            "uid": self.uid,
            "summary": self.summary,
            "start": _isoformat(self.starts_at),
            "end": _isoformat(self.ends_at),
            "location": self.location,
            "description": self.description,
            "attendees": list(self.attendees),
            "url": self.url,
            "status": self.status,
        }


def _isoformat(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


# -- iCalendar -------------------------------------------------------------


def _escape(value: str) -> str:
    """Escape one property value.

    A comma, a semicolon, or a newline inside a summary is what turns a valid
    object into one the server stores and no client can read back.
    """
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _unescape(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\" and index + 1 < len(value):
            following = value[index + 1]
            result.append(
                "\n" if following in ("n", "N") else following
            )
            index += 2
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _fold(line: str) -> str:
    """Fold one content line to the format's octet limit."""
    encoded = line.encode("utf-8")
    if len(encoded) <= MAX_LINE_OCTETS:
        return line
    pieces: list[str] = []
    current = ""
    current_octets = 0
    for character in line:
        width = len(character.encode("utf-8"))
        # Continuation lines carry a leading space, which counts toward the
        # limit, hence the one-octet allowance after the first piece.
        limit = MAX_LINE_OCTETS if not pieces else MAX_LINE_OCTETS - 1
        if current_octets + width > limit:
            pieces.append(current)
            current = character
            current_octets = width
            continue
        current += character
        current_octets += width
    pieces.append(current)
    return "\r\n ".join(pieces)


def _unfold(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        if raw_line[:1] in (" ", "\t") and lines:
            lines[-1] += raw_line[1:]
            continue
        lines.append(raw_line)
    return lines


def _format_moment(value: datetime | date) -> tuple[str, str]:
    """Return ``(parameters, value)`` for one date or timestamp."""
    if isinstance(value, datetime):
        moment = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return "", moment.strftime("%Y%m%dT%H%M%SZ")
    return ";VALUE=DATE", value.strftime("%Y%m%d")


def _parse_moment(raw_value: str, parameters: str) -> datetime | date | None:
    value = raw_value.strip()
    if not value:
        return None
    if "VALUE=DATE" in parameters.upper() and "DATE-TIME" not in parameters.upper():
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            return None
    for pattern, is_utc in (("%Y%m%dT%H%M%SZ", True), ("%Y%m%dT%H%M%S", False)):
        try:
            moment = datetime.strptime(value, pattern)
        except ValueError:
            continue
        return moment.replace(tzinfo=UTC) if is_utc else moment.replace(tzinfo=UTC)
    return None


def build_icalendar(event: CalendarEvent) -> str:
    """Render one appointment as an iCalendar object a server will accept."""
    if not event.uid:
        event.uid = f"{uuid.uuid4()}@neuralnexus"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Neural Nexus//Avatar Calendar//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{event.uid}",
        f"DTSTAMP:{stamp}",
        f"SUMMARY:{_escape(event.summary)}",
    ]
    if event.starts_at is not None:
        parameters, value = _format_moment(event.starts_at)
        lines.append(f"DTSTART{parameters}:{value}")
    if event.ends_at is not None:
        parameters, value = _format_moment(event.ends_at)
        lines.append(f"DTEND{parameters}:{value}")
    if event.description:
        lines.append(f"DESCRIPTION:{_escape(event.description)}")
    if event.location:
        lines.append(f"LOCATION:{_escape(event.location)}")
    if event.organizer:
        lines.append(f"ORGANIZER:mailto:{event.organizer}")
    for attendee in event.attendees:
        lines.append(
            "ATTENDEE;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;RSVP=TRUE:"
            f"mailto:{attendee}"
        )
    if event.status:
        lines.append(f"STATUS:{event.status.upper()}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


def parse_icalendar(text: str) -> list[CalendarEvent]:
    """Read every ``VEVENT`` in one iCalendar object."""
    events: list[CalendarEvent] = []
    current: CalendarEvent | None = None
    for line in _unfold(text):
        stripped = line.strip()
        if stripped == "BEGIN:VEVENT":
            current = CalendarEvent()
            continue
        if stripped == "END:VEVENT":
            if current is not None:
                events.append(current)
            current = None
            continue
        if current is None or ":" not in stripped:
            continue
        name_part, _, value = stripped.partition(":")
        name, _, parameters = name_part.partition(";")
        name = name.upper()
        if name == "UID":
            current.uid = value.strip()
        elif name == "SUMMARY":
            current.summary = _unescape(value)
        elif name == "DESCRIPTION":
            current.description = _unescape(value)
        elif name == "LOCATION":
            current.location = _unescape(value)
        elif name == "STATUS":
            current.status = value.strip().lower()
        elif name == "DTSTART":
            current.starts_at = _parse_moment(value, parameters)
        elif name == "DTEND":
            current.ends_at = _parse_moment(value, parameters)
        elif name == "ORGANIZER":
            current.organizer = value.strip().removeprefix("mailto:")
        elif name == "ATTENDEE":
            current.attendees.append(value.strip().removeprefix("mailto:"))
    return events


# -- WebDAV plumbing -------------------------------------------------------


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _first_href(element: Any) -> str:
    href = element.find(f".//{_tag(DAV_NAMESPACE, 'href')}")
    return (href.text or "").strip() if href is not None and href.text else ""


def _absolute(base_url: str, href: str) -> str:
    from urllib.parse import urljoin

    return urljoin(base_url, href)


@dataclass(frozen=True)
class CalDavAccount:
    """A proven calendar account: where it is and how to sign in."""

    base_url: str
    username: str
    password: str
    principal_url: str = ""
    calendar_home_url: str = ""

    @property
    def auth(self) -> tuple[str, str]:
        """Return the basic-auth pair for httpx."""
        return (self.username, self.password)


async def _request(
    account: CalDavAccount,
    method: str,
    url: str,
    *,
    body: str | None = None,
    headers: dict[str, str] | None = None,
    http_client: Any = None,
) -> Any:
    import httpx

    request_headers = {"content-type": 'application/xml; charset="utf-8"'}
    request_headers.update(headers or {})
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        response = await client.request(
            method,
            url,
            content=body.encode("utf-8") if body is not None else None,
            headers=request_headers,
            auth=account.auth,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as transport_error:  # noqa: BLE001 - reported as unreachable
        raise CalDavUnreachableError(str(transport_error)) from transport_error
    finally:
        if owns_client:
            await client.aclose()
    if response.status_code in (401, 403):
        raise CalDavAuthenticationError(
            f"{url} rejected the address and password ({response.status_code})."
        )
    if response.status_code >= 400:
        raise CalDavError(f"{url} answered {response.status_code}.")
    return response


_PRINCIPAL_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:"><d:prop><d:current-user-principal/></d:prop>'
    "</d:propfind>"
)

_CALENDAR_HOME_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "<d:prop><c:calendar-home-set/></d:prop></d:propfind>"
)

_CALENDAR_LIST_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav" '
    'xmlns:a="http://apple.com/ns/ical/">'
    "<d:prop><d:resourcetype/><d:displayname/><d:current-user-privilege-set/>"
    "<a:calendar-color/></d:prop></d:propfind>"
)


async def _service_urls(domain: str, http_client: Any = None) -> list[str]:
    """Return candidate CalDAV roots for ``domain``, best first."""
    candidates = [f"https://{domain}{WELL_KNOWN_PATH}"]

    import httpx

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        response = await client.get(
            DNS_OVER_HTTPS_URL,
            params={"name": f"_caldavs._tcp.{domain}", "type": "SRV"},
            headers={"accept": "application/dns-json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            for answer in (response.json().get("Answer") or []):
                if int(answer.get("type") or 0) != 33:
                    continue
                parts = str(answer.get("data") or "").split()
                if len(parts) != 4:
                    continue
                port, target = parts[2], parts[3].strip().rstrip(".")
                if not target or target == ".":
                    continue
                suffix = "" if port == "443" else f":{port}"
                candidates.append(f"https://{target}{suffix}{WELL_KNOWN_PATH}")
    except Exception:  # noqa: BLE001 - no SRV answer just means fewer candidates
        pass
    finally:
        if owns_client:
            await client.aclose()

    candidates.append(f"https://{domain}/")
    return candidates


async def connect_caldav_account(
    *,
    email_address: str,
    password: str,
    server_url: str = "",
    username: str = "",
    http_client: Any = None,
) -> CalDavAccount:
    """Find, prove, and describe one calendar account.

    Raises :class:`CalDavAuthenticationError` when the password is refused and
    :class:`CalDavUnreachableError` when nothing answers, so the caller can tell
    the owner which of the two happened.
    """
    domain = email_address.partition("@")[2].strip().lower()
    account_username = username or email_address
    candidates = [server_url] if server_url else await _service_urls(domain, http_client)

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        account = CalDavAccount(
            base_url=candidate, username=account_username, password=password
        )
        try:
            response = await _request(
                account,
                "PROPFIND",
                candidate,
                body=_PRINCIPAL_BODY,
                headers={"depth": "0"},
                http_client=http_client,
            )
            # Scope the search to the property itself. The first ``href`` in a
            # multistatus document is the response's OWN address, so reading
            # the document's first href would "discover" the URL just asked
            # for and loop back on itself.
            principal_element = ElementTree.fromstring(response.text).find(
                f".//{_tag(DAV_NAMESPACE, 'current-user-principal')}"
            )
            principal_href = (
                _first_href(principal_element) if principal_element is not None else ""
            )
            if not principal_href:
                continue
            principal_url = _absolute(str(response.url), principal_href)

            response = await _request(
                account,
                "PROPFIND",
                principal_url,
                body=_CALENDAR_HOME_BODY,
                headers={"depth": "0"},
                http_client=http_client,
            )
            root = ElementTree.fromstring(response.text)
            home_element = root.find(
                f".//{_tag(CALDAV_NAMESPACE, 'calendar-home-set')}"
            )
            home_href = _first_href(home_element) if home_element is not None else ""
            if not home_href:
                continue
            return CalDavAccount(
                base_url=candidate,
                username=account_username,
                password=password,
                principal_url=principal_url,
                calendar_home_url=_absolute(principal_url, home_href),
            )
        except CalDavAuthenticationError:
            # A refused password is the answer, not a reason to try the next
            # candidate with the same password.
            raise
        except CalDavError as error:
            last_error = error
            continue

    raise CalDavUnreachableError(
        f"No calendar server answered for {domain}."
        if last_error is None
        else f"No calendar server answered for {domain}: {last_error}"
    )


async def list_calendars(
    account: CalDavAccount, *, http_client: Any = None
) -> list[CalendarCollection]:
    """Return every calendar collection in the account's calendar home."""
    response = await _request(
        account,
        "PROPFIND",
        account.calendar_home_url,
        body=_CALENDAR_LIST_BODY,
        headers={"depth": "1"},
        http_client=http_client,
    )
    root = ElementTree.fromstring(response.text)
    collections: list[CalendarCollection] = []
    for entry in root.findall(_tag(DAV_NAMESPACE, "response")):
        resourcetype = entry.find(f".//{_tag(DAV_NAMESPACE, 'resourcetype')}")
        if resourcetype is None:
            continue
        if resourcetype.find(_tag(CALDAV_NAMESPACE, "calendar")) is None:
            continue
        href = _first_href(entry)
        if not href:
            continue
        display_name_element = entry.find(f".//{_tag(DAV_NAMESPACE, 'displayname')}")
        color_element = entry.find(f".//{_tag(APPLE_NAMESPACE, 'calendar-color')}")
        privileges = entry.find(
            f".//{_tag(DAV_NAMESPACE, 'current-user-privilege-set')}"
        )
        may_write = True
        if privileges is not None and len(privileges):
            may_write = any(
                privilege.find(_tag(DAV_NAMESPACE, "write")) is not None
                or privilege.find(_tag(DAV_NAMESPACE, "write-content")) is not None
                or privilege.find(_tag(DAV_NAMESPACE, "all")) is not None
                for privilege in privileges
            )
        collections.append(
            CalendarCollection(
                url=_absolute(account.calendar_home_url, href),
                display_name=(
                    (display_name_element.text or "").strip()
                    if display_name_element is not None
                    else ""
                )
                or href.rstrip("/").rsplit("/", 1)[-1],
                color=(
                    (color_element.text or "").strip()
                    if color_element is not None
                    else ""
                ),
                read_only=not may_write,
            )
        )
    return collections


def _calendar_query_body(since: datetime, until: datetime) -> str:
    start = since.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    end = until.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
        "<d:prop><d:getetag/><c:calendar-data/></d:prop>"
        '<c:filter><c:comp-filter name="VCALENDAR">'
        '<c:comp-filter name="VEVENT">'
        f'<c:time-range start="{start}" end="{end}"/>'
        "</c:comp-filter></c:comp-filter></c:filter></c:calendar-query>"
    )


async def list_events(
    account: CalDavAccount,
    calendar_url: str,
    *,
    since: datetime,
    until: datetime,
    http_client: Any = None,
) -> list[CalendarEvent]:
    """Return every appointment on one calendar between two moments."""
    response = await _request(
        account,
        "REPORT",
        calendar_url,
        body=_calendar_query_body(since, until),
        headers={"depth": "1"},
        http_client=http_client,
    )
    root = ElementTree.fromstring(response.text)
    events: list[CalendarEvent] = []
    for entry in root.findall(_tag(DAV_NAMESPACE, "response")):
        data_element = entry.find(f".//{_tag(CALDAV_NAMESPACE, 'calendar-data')}")
        if data_element is None or not (data_element.text or "").strip():
            continue
        etag_element = entry.find(f".//{_tag(DAV_NAMESPACE, 'getetag')}")
        href = _first_href(entry)
        for event in parse_icalendar(data_element.text or ""):
            event.url = _absolute(calendar_url, href) if href else ""
            event.etag = (
                (etag_element.text or "").strip() if etag_element is not None else ""
            )
            events.append(event)
    events.sort(
        key=lambda item: (
            item.starts_at.isoformat() if item.starts_at is not None else ""
        )
    )
    return events


async def create_event(
    account: CalDavAccount,
    calendar_url: str,
    event: CalendarEvent,
    *,
    http_client: Any = None,
) -> CalendarEvent:
    """Put one new appointment on a calendar."""
    if not event.uid:
        event.uid = f"{uuid.uuid4()}@neuralnexus"
    target = calendar_url.rstrip("/") + f"/{_safe_resource_name(event.uid)}.ics"
    response = await _request(
        account,
        "PUT",
        target,
        body=build_icalendar(event),
        headers={
            "content-type": "text/calendar; charset=utf-8",
            # Refuse to silently overwrite an existing object at this address.
            "if-none-match": "*",
        },
        http_client=http_client,
    )
    event.url = target
    event.etag = response.headers.get("etag", "")
    return event


async def update_event(
    account: CalDavAccount,
    event: CalendarEvent,
    *,
    http_client: Any = None,
) -> CalendarEvent:
    """Replace one existing appointment."""
    if not event.url:
        raise CalDavError("The appointment to update has no address on the server.")
    headers = {"content-type": "text/calendar; charset=utf-8"}
    if event.etag:
        # Only overwrite the version that was read, so a change made elsewhere
        # in the meantime is reported rather than discarded.
        headers["if-match"] = event.etag
    response = await _request(
        account,
        "PUT",
        event.url,
        body=build_icalendar(event),
        headers=headers,
        http_client=http_client,
    )
    event.etag = response.headers.get("etag", event.etag)
    return event


async def delete_event(
    account: CalDavAccount, event_url: str, *, etag: str = "", http_client: Any = None
) -> None:
    """Remove one appointment."""
    headers = {"if-match": etag} if etag else {}
    await _request(
        account, "DELETE", event_url, headers=headers, http_client=http_client
    )


async def find_free_time(
    account: CalDavAccount,
    calendar_urls: list[str],
    *,
    since: datetime,
    until: datetime,
    duration_minutes: int,
    http_client: Any = None,
) -> list[dict[str, str]]:
    """Return openings of at least ``duration_minutes`` across the calendars.

    Computed from the appointments themselves rather than from a free-busy
    report, because free-busy is the CalDAV feature servers most often decline
    to implement, and the events are already readable.
    """
    busy: list[tuple[datetime, datetime]] = []
    for calendar_url in calendar_urls:
        for event in await list_events(
            account, calendar_url, since=since, until=until, http_client=http_client
        ):
            start = _as_datetime(event.starts_at)
            end = _as_datetime(event.ends_at) or (
                start + timedelta(hours=1) if start else None
            )
            if start and end:
                busy.append((start, end))

    busy.sort()
    merged: list[list[datetime]] = []
    for start, end in busy:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])

    minimum = timedelta(minutes=max(1, int(duration_minutes)))
    openings: list[dict[str, str]] = []
    cursor = since
    for start, end in merged:
        if start - cursor >= minimum:
            openings.append(
                {"start": cursor.isoformat(), "end": start.isoformat()}
            )
        cursor = max(cursor, end)
    if until - cursor >= minimum:
        openings.append({"start": cursor.isoformat(), "end": until.isoformat()})
    return openings


def _as_datetime(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _safe_resource_name(uid: str) -> str:
    """Return a file name for one appointment that no server will reject."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", uid)[:120] or uuid.uuid4().hex
