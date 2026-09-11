"""The avatar's calendar tools for an account connected with a password.

Six tools over :mod:`caldav_client`, shaped the way the mail tools are shaped:
every one returns a dictionary and none of them raise, because a calendar that
is briefly unreachable must produce a sentence the avatar can say rather than a
traceback that ends the turn.

Two rules are enforced here rather than left to the prompt:

* **An appointment is never created, changed, or deleted without the owner
  saying so.** The docstrings state it, and the inbox path that proposes an
  appointment always waits for the owner's decision before calling these.
* **A calendar the owner may only read is never written to.** The server
  reports that in its privilege set, so a read-only calendar is refused here
  with the reason, instead of the write failing at the server with a status
  code the owner would have to interpret.

Times are accepted as ISO 8601. A bare date means an all-day appointment,
which is what a person means when they say "Thursday" and not "Thursday at
three".
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

CALDAV_TOOL_NAMES: tuple[str, ...] = (
    "list_calendars",
    "calendar_events",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "find_free_time",
)

# How far either side of today an appointment named only by its identifier is
# looked for. A year covers everything a person asks about by name; beyond that
# the owner has the address from a listing.
UID_SEARCH_DAYS = 365

DEFAULT_EVENT_MINUTES = 60


def _parse_moment(value: str | None, *, end_of_day: bool = False) -> datetime | date | None:
    """Read one ISO 8601 date or timestamp the way a person wrote it."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if len(text) == 10 and text.count("-") == 2:
            return date.fromisoformat(text)
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _as_datetime(value: datetime | date | None, *, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    hour = 23 if end_of_day else 0
    minute = 59 if end_of_day else 0
    return datetime(value.year, value.month, value.day, hour, minute, tzinfo=UTC)


def _window(since: str | None, until: str | None, *, default_days: int = 7):
    start = _as_datetime(_parse_moment(since)) or datetime.now(UTC)
    end = _as_datetime(_parse_moment(until), end_of_day=True) or (
        start + timedelta(days=default_days)
    )
    return start, end


async def caldav_account_for(
    record: dict[str, Any], context: Any
) -> Any:
    """Rebuild the signed-in calendar account one record describes."""
    from src.anubis.utils.connected_accounts.caldav_client import CalDavAccount
    from src.anubis.utils.secret_store import decrypt_secret

    transport = (record.get("transport") or {}).get("caldav") or {}
    return CalDavAccount(
        base_url=str(transport.get("base_url") or ""),
        username=str(transport.get("username") or record.get("account_address") or ""),
        password=decrypt_secret(record["encrypted_secret"], context),
        principal_url=str(transport.get("principal_url") or ""),
        calendar_home_url=str(transport.get("calendar_home_url") or ""),
    )


def _selector(records: list[dict[str, Any]]):
    """Choose which connected calendar account a call is about."""

    def select(connection: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not records:
            return None, {
                "status": "not_connected",
                "error": "No calendar account is connected.",
            }
        wanted = str(connection or "").strip().lower()
        if not wanted:
            return records[0], None
        for record in records:
            if wanted in (
                str(record.get("display_label") or "").lower(),
                str(record.get("account_address") or "").lower(),
            ):
                return record, None
        return None, {
            "status": "unknown_connection",
            "error": (
                f"No connected calendar account matches {connection!r}. "
                "Connected: "
                + ", ".join(str(record.get("display_label")) for record in records)
            ),
        }

    return select


def build_caldav_tools(
    context: Any, accounts: list[dict[str, Any]], *, store: Any = None
) -> list[Any]:
    """Build the calendar tools for every password-connected calendar account."""
    from langchain_core.tools import tool

    from src.anubis.utils.connected_accounts import caldav_client

    select = _selector(accounts)

    async def _resolve_calendar(
        account: Any, calendar: str | None, *, for_writing: bool
    ) -> tuple[Any, dict[str, Any] | None]:
        collections = await caldav_client.list_calendars(account)
        if not collections:
            return None, {
                "status": "error",
                "error": "The account has no calendars.",
            }
        wanted = str(calendar or "").strip().lower()
        chosen = None
        if wanted:
            for collection in collections:
                if wanted in (collection.display_name.lower(), collection.url.lower()):
                    chosen = collection
                    break
            if chosen is None:
                return None, {
                    "status": "unknown_calendar",
                    "error": (
                        f"No calendar named {calendar!r}. Calendars: "
                        + ", ".join(item.display_name for item in collections)
                    ),
                }
        else:
            writable = [item for item in collections if not item.read_only]
            chosen = (writable or collections)[0]
        if for_writing and chosen.read_only:
            return None, {
                "status": "read_only",
                "error": (
                    f"{chosen.display_name} is a calendar this account may only "
                    "read, so nothing can be booked on it. Name a different "
                    "calendar."
                ),
            }
        return chosen, None

    def _failure(error: Exception) -> dict[str, Any]:
        if isinstance(error, caldav_client.CalDavAuthenticationError):
            return {
                "status": "needs_reconnect",
                "error": (
                    "The calendar server rejected its saved password. Ask the "
                    "owner to connect the calendar again."
                ),
            }
        if isinstance(error, caldav_client.CalDavUnreachableError):
            return {"status": "unreachable", "error": str(error)}
        logger.info("Calendar call failed: %s", error)
        return {"status": "error", "error": str(error)}

    @tool
    async def list_calendars(connection: str | None = None) -> dict[str, Any]:
        """List the owner's calendars, and say which ones can be written to."""
        record, problem = select(connection)
        if problem:
            return problem
        try:
            account = await caldav_account_for(record, context)
            collections = await caldav_client.list_calendars(account)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        return {
            "status": "ok",
            "account": record.get("display_label"),
            "calendars": [
                {
                    "name": collection.display_name,
                    "read_only": collection.read_only,
                    "color": collection.color,
                }
                for collection in collections
            ],
        }

    @tool
    async def calendar_events(
        since: str | None = None,
        until: str | None = None,
        calendar: str | None = None,
        connection: str | None = None,
    ) -> dict[str, Any]:
        """Read the owner's appointments between two moments.

        ``since`` and ``until`` are ISO 8601; with neither, the next seven days.
        """
        record, problem = select(connection)
        if problem:
            return problem
        start, end = _window(since, until)
        try:
            account = await caldav_account_for(record, context)
            if calendar:
                collection, calendar_problem = await _resolve_calendar(
                    account, calendar, for_writing=False
                )
                if calendar_problem:
                    return calendar_problem
                targets = [collection]
            else:
                targets = await caldav_client.list_calendars(account)
            events: list[dict[str, Any]] = []
            for collection in targets:
                for event in await caldav_client.list_events(
                    account, collection.url, since=start, until=end
                ):
                    entry = event.as_public_dict()
                    entry["calendar"] = collection.display_name
                    events.append(entry)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        events.sort(key=lambda item: item.get("start") or "")
        return {
            "status": "ok",
            "account": record.get("display_label"),
            "since": start.isoformat(),
            "until": end.isoformat(),
            "events": events,
        }

    @tool
    async def create_calendar_event(
        summary: str,
        start: str,
        end: str | None = None,
        description: str = "",
        location: str = "",
        attendees: list[str] | None = None,
        calendar: str | None = None,
        connection: str | None = None,
    ) -> dict[str, Any]:
        """Book one appointment on the owner's calendar.

        Call this only when the conversation partner has asked for the
        appointment to be created and has seen the day, the time, and the
        title. ``start`` and ``end`` are ISO 8601; a bare date books an all-day
        appointment. With no ``end``, the appointment lasts one hour.
        """
        record, problem = select(connection)
        if problem:
            return problem
        starts_at = _parse_moment(start)
        if starts_at is None:
            return {
                "status": "error",
                "error": f"{start!r} is not a date or time this can read.",
            }
        ends_at = _parse_moment(end)
        if ends_at is None:
            ends_at = (
                starts_at + timedelta(days=1)
                if isinstance(starts_at, date) and not isinstance(starts_at, datetime)
                else starts_at + timedelta(minutes=DEFAULT_EVENT_MINUTES)
            )
        try:
            account = await caldav_account_for(record, context)
            collection, calendar_problem = await _resolve_calendar(
                account, calendar, for_writing=True
            )
            if calendar_problem:
                return calendar_problem
            event = caldav_client.CalendarEvent(
                summary=summary,
                starts_at=starts_at,
                ends_at=ends_at,
                description=description,
                location=location,
                attendees=list(attendees or []),
                organizer=str(record.get("account_address") or ""),
            )
            created = await caldav_client.create_event(account, collection.url, event)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        return {
            "status": "created",
            "calendar": collection.display_name,
            "event": created.as_public_dict(),
        }

    async def _find_event(account: Any, uid: str, event_url: str):
        if event_url:
            now = datetime.now(UTC)
            for collection in await caldav_client.list_calendars(account):
                for event in await caldav_client.list_events(
                    account,
                    collection.url,
                    since=now - timedelta(days=UID_SEARCH_DAYS),
                    until=now + timedelta(days=UID_SEARCH_DAYS),
                ):
                    if event.url == event_url:
                        return event, collection
            return None, None
        now = datetime.now(UTC)
        for collection in await caldav_client.list_calendars(account):
            for event in await caldav_client.list_events(
                account,
                collection.url,
                since=now - timedelta(days=UID_SEARCH_DAYS),
                until=now + timedelta(days=UID_SEARCH_DAYS),
            ):
                if event.uid == uid:
                    return event, collection
        return None, None

    @tool
    async def update_calendar_event(
        uid: str = "",
        event_url: str = "",
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
        connection: str | None = None,
    ) -> dict[str, Any]:
        """Change one existing appointment.

        Name the appointment by the ``uid`` or the ``url`` a listing returned.
        Only the fields given are changed. Call this only when the conversation
        partner has asked for the change.
        """
        record, problem = select(connection)
        if problem:
            return problem
        if not uid and not event_url:
            return {
                "status": "error",
                "error": "Name the appointment by its uid or its url.",
            }
        try:
            account = await caldav_account_for(record, context)
            event, collection = await _find_event(account, uid, event_url)
            if event is None:
                return {
                    "status": "not_found",
                    "error": "No appointment with that identifier is on the calendar.",
                }
            if collection is not None and collection.read_only:
                return {
                    "status": "read_only",
                    "error": (
                        f"{collection.display_name} is a calendar this account "
                        "may only read."
                    ),
                }
            if summary is not None:
                event.summary = summary
            if description is not None:
                event.description = description
            if location is not None:
                event.location = location
            if start:
                parsed_start = _parse_moment(start)
                if parsed_start is None:
                    return {"status": "error", "error": f"{start!r} is not a time."}
                event.starts_at = parsed_start
            if end:
                parsed_end = _parse_moment(end)
                if parsed_end is None:
                    return {"status": "error", "error": f"{end!r} is not a time."}
                event.ends_at = parsed_end
            updated = await caldav_client.update_event(account, event)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        return {"status": "updated", "event": updated.as_public_dict()}

    @tool
    async def delete_calendar_event(
        uid: str = "", event_url: str = "", connection: str | None = None
    ) -> dict[str, Any]:
        """Remove one appointment from the owner's calendar.

        Call this only when the conversation partner has asked for the
        appointment to be removed and knows which one.
        """
        record, problem = select(connection)
        if problem:
            return problem
        if not uid and not event_url:
            return {
                "status": "error",
                "error": "Name the appointment by its uid or its url.",
            }
        try:
            account = await caldav_account_for(record, context)
            event, collection = await _find_event(account, uid, event_url)
            if event is None:
                return {
                    "status": "not_found",
                    "error": "No appointment with that identifier is on the calendar.",
                }
            if collection is not None and collection.read_only:
                return {
                    "status": "read_only",
                    "error": (
                        f"{collection.display_name} is a calendar this account "
                        "may only read."
                    ),
                }
            await caldav_client.delete_event(account, event.url, etag=event.etag)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        return {"status": "deleted", "summary": event.summary}

    @tool
    async def find_free_time(
        since: str | None = None,
        until: str | None = None,
        duration_minutes: int = 30,
        calendar: str | None = None,
        connection: str | None = None,
    ) -> dict[str, Any]:
        """Find openings of at least ``duration_minutes`` in the owner's week."""
        record, problem = select(connection)
        if problem:
            return problem
        start, end = _window(since, until)
        try:
            account = await caldav_account_for(record, context)
            if calendar:
                collection, calendar_problem = await _resolve_calendar(
                    account, calendar, for_writing=False
                )
                if calendar_problem:
                    return calendar_problem
                urls = [collection.url]
            else:
                urls = [item.url for item in await caldav_client.list_calendars(account)]
            openings = await caldav_client.find_free_time(
                account,
                urls,
                since=start,
                until=end,
                duration_minutes=duration_minutes,
            )
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return _failure(error)
        return {
            "status": "ok",
            "duration_minutes": duration_minutes,
            "openings": openings,
        }

    return [
        list_calendars,
        calendar_events,
        create_calendar_event,
        update_calendar_event,
        delete_calendar_event,
        find_free_time,
    ]
