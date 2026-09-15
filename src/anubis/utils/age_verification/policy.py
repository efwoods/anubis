"""Adult-only avatars and the age that unlocks them in search.

An avatar marked adult-only stays out of public discovery — the gallery
search bar, the public listing, the world map — until the signed-in viewer
has confirmed they are old enough. The platform administrator is the one
exception: that account has to see the adult-only names to mark and unmark
them, so those avatars stay in the administrator's search without age
verification. Being the creator does not put the name in public search;
the owner still reaches the avatar from their own carousel or a share
link. A share link that names one assistant stays reachable: hiding from
search is not the same as making the avatar unreachable.

Date of birth is the confirmation. The stored date is never returned to the
browser; the routes answer only whether the account is verified.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

ADULT_ONLY_METADATA_KEY = "adult_only"
DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS = 18
MAXIMUM_PLAUSIBLE_AGE_YEARS = 130


def _truthy_flag(value: Any) -> bool:
    """Return whether a metadata flag is an explicit true."""
    return value is True or (
        isinstance(value, str) and value.strip().lower() == "true"
    )


def adult_only_flag_of(assistant: dict[str, Any] | None) -> bool:
    """Return whether this avatar is marked adult-only.

    Reads the flag from metadata first, then from a lifted top-level field,
    so a public listing that has already stripped metadata still answers.
    """
    if not isinstance(assistant, dict):
        return False
    metadata = assistant.get("metadata")
    if isinstance(metadata, dict) and _truthy_flag(
        metadata.get(ADULT_ONLY_METADATA_KEY)
    ):
        return True
    return _truthy_flag(assistant.get(ADULT_ONLY_METADATA_KEY))


def creator_user_id_of(assistant: dict[str, Any] | None) -> str | None:
    """Return the creator stamped on this avatar, or None when unknown."""
    if not isinstance(assistant, dict):
        return None
    metadata = assistant.get("metadata")
    if isinstance(metadata, dict):
        creator_user_id = str(metadata.get("user_id") or "").strip()
        if creator_user_id:
            return creator_user_id
    return None


def minimum_verification_years(context: Any | None) -> int:
    """Return the configured minimum age, defaulting to eighteen."""
    raw_value = getattr(context, "age_verification_minimum_years", None)
    try:
        years = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS
    if years < 1:
        return DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS
    return years


def parse_date_of_birth(value: Any) -> date:
    """Parse a calendar date of birth from ``YYYY-MM-DD``.

    Raises ``ValueError`` when the value is missing, not that shape, or not a
    real calendar day.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("Supply a date of birth as YYYY-MM-DD.")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as invalid_date:
        raise ValueError("Date of birth must be a real calendar day as YYYY-MM-DD.") from invalid_date
    return parsed


def years_elapsed_since(date_of_birth: date, on_date: date | None = None) -> int:
    """Return how many full years have passed from ``date_of_birth`` to ``on_date``."""
    today = on_date or date.today()
    years = today.year - date_of_birth.year
    if (today.month, today.day) < (date_of_birth.month, date_of_birth.day):
        years -= 1
    return years


def date_of_birth_verification_error(
    date_of_birth: date,
    *,
    minimum_years: int = DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
    on_date: date | None = None,
) -> str | None:
    """Return why this date of birth cannot verify the account, or None when it can."""
    today = on_date or date.today()
    if date_of_birth > today:
        return "Date of birth cannot be in the future."
    years = years_elapsed_since(date_of_birth, today)
    if years > MAXIMUM_PLAUSIBLE_AGE_YEARS:
        return "Date of birth is not a plausible age."
    if years < int(minimum_years):
        return (
            f"You must be {int(minimum_years)} or older to see adult-only avatars "
            "in search."
        )
    return None


def as_date(value: Any) -> date | None:
    """Coerce a stored date or ISO string to a date, or None when unreadable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def verification_is_current(
    row: dict[str, Any] | None,
    *,
    minimum_years: int = DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
    on_date: date | None = None,
) -> bool:
    """Return whether a stored verification still meets the minimum age."""
    if not row:
        return False
    date_of_birth = as_date(row.get("date_of_birth"))
    if date_of_birth is None:
        return False
    return (
        date_of_birth_verification_error(
            date_of_birth, minimum_years=minimum_years, on_date=on_date
        )
        is None
    )


def viewer_may_discover_adult_only_avatar(
    assistant: dict[str, Any],
    *,
    viewer_age_verified: bool,
    viewer_user_id: str | None = None,
    viewer_is_admin: bool = False,
) -> bool:
    """Return whether this avatar may appear in search and public listings.

    Avatars that are not adult-only are always discoverable. Adult-only
    avatars appear after the viewer has verified their age, or when the
    viewer is the platform administrator. The creator is not an exception:
    that role still reaches the avatar from the owner's own list or a share
    link, not from public search. ``viewer_user_id`` is accepted so callers
    can pass the full viewer record; it does not unlock search.
    """
    if not adult_only_flag_of(assistant):
        return True
    return bool(viewer_age_verified) or bool(viewer_is_admin)


def filter_discoverable_avatars(
    avatars: list[dict[str, Any]],
    *,
    viewer_age_verified: bool,
    viewer_user_id: str | None = None,
    viewer_is_admin: bool = False,
    allow_direct_lookup: bool = False,
) -> list[dict[str, Any]]:
    """Drop adult-only avatars the viewer may not discover.

    ``allow_direct_lookup`` keeps every row: that is the share-link case, where
    the caller already named one assistant.
    """
    if allow_direct_lookup:
        return list(avatars)
    return [
        assistant
        for assistant in avatars
        if viewer_may_discover_adult_only_avatar(
            assistant,
            viewer_age_verified=viewer_age_verified,
            viewer_user_id=viewer_user_id,
            viewer_is_admin=viewer_is_admin,
        )
    ]
