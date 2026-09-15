"""Personal-avatar phone calling: lookup, travel, SIP, and the MCP surface.

Place lookup and travel belong to the personal avatar itself. They do not
need a mobile Model Context Protocol daemon and they do not need a Phone
connection. SIP tools attach only after the owner verifies the mobile they
already have. Per-user phone numbers are enterprise-only and are refused
on every other tier.
"""

from src.anubis.utils.phone.numbers import (
    PhoneNumberError,
    normalize_e164,
)
from src.anubis.utils.phone.tools import (
    LOOKUP_TRAVEL_TOOL_NAMES,
    SIP_TOOL_NAMES,
)

__all__ = [
    "LOOKUP_TRAVEL_TOOL_NAMES",
    "PhoneNumberError",
    "SIP_TOOL_NAMES",
    "normalize_e164",
]
