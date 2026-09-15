"""One LiveKit room and the people on it for a single phone call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class PhoneCallSession:
    """Identifiers the worker and the tools share for one call."""

    call_id: str
    user_id: str
    assistant_id: str
    thread_id: str | None
    direction: str
    room_name: str
    owner_mobile_e164: str | None = None
    destination_e164: str | None = None
    destination_name: str | None = None
    brief: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PhoneCallSession":
        """Build a session from a ``phone_calls`` row."""
        return cls(
            call_id=str(record["call_id"]),
            user_id=str(record["user_id"]),
            assistant_id=str(record["assistant_id"]),
            thread_id=record.get("thread_id"),
            direction=str(record.get("direction") or "outbound"),
            room_name=str(record.get("room_name") or f"phone-{record['call_id']}"),
            owner_mobile_e164=record.get("owner_mobile_e164"),
            destination_e164=record.get("destination_e164"),
            destination_name=record.get("destination_name"),
            brief=dict(record.get("brief") or {}),
        )


def new_call_id() -> str:
    """Return a fresh call identifier."""
    return str(uuid4())
