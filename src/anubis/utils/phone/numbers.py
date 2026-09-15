"""Normalize and refuse phone numbers.

The connection is the mobile the owner already has. A number that cannot be
written as E.164 is not a number we can SMS-verify, ring, or dispatch on.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D+")


class PhoneNumberError(ValueError):
    """A value that cannot be written as an E.164 telephone number."""


def normalize_e164(value: str | None) -> str:
    """Return ``value`` as E.164 (leading plus, digits only).

    Ten national digits with no country code are treated as +1 (NANP). Anything
    else must already carry a country code. Empty or punctuation-only input is
    refused rather than guessed.
    """
    raw = str(value or "").strip()
    if not raw:
        raise PhoneNumberError("A phone number is required.")
    has_plus = raw.startswith("+")
    digits = _DIGITS.sub("", raw)
    if not digits:
        raise PhoneNumberError("A phone number is required.")
    if not digits.isdigit():
        raise PhoneNumberError("A phone number may contain only digits and a leading +.")
    if has_plus:
        if len(digits) < 8 or len(digits) > 15:
            raise PhoneNumberError("That phone number is not a valid length.")
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    raise PhoneNumberError(
        "Write the number with a country code (for example +1 404 555 0100)."
    )
