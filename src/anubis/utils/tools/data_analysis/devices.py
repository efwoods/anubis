"""Device identity for the multi-device Model Context Protocol capability.

A user runs the Neural Nexus Model Context Protocol daemon on several machines
at once (Ubuntu desktop, macOS, mobile, Windows). Every store record and every
live relay session is keyed by the daemon's ``device_id`` — a random opaque
token such as ``kQ7f2xR9tLp0``. That token is useless in a conversation: an
avatar cannot say "I found twelve files on kQ7f2xR9tLp0". Each device therefore
also carries a human-readable ``device_label`` and a coarse ``platform``.

Where the label comes from, in priority order:

1. The daemon sends an explicit ``device_label``. The newer daemons default the
   label to the machine's hostname, which distinguishes two Ubuntu desktops from
   each other in a way a platform name never can.
2. This module derives a label from the announced ``server_name``. Derivation is
   required regardless of what the daemons send, because an already-installed
   daemon keeps announcing the old payload until the user updates the daemon —
   so the API can never assume the explicit field is present.

Derivation matches ``server_name`` by KEYWORD rather than against an exact table
of known names. The Windows daemon is being templated from the macOS daemon and
has not chosen its exact ``server_name`` string yet; keyword matching means that
daemon resolves correctly whatever exact string the template settles on, instead
of silently falling back to "Unknown device" on the day the daemon first
connects.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Ordered keyword table used to derive ``(device_label, platform)`` from an
# announced ``server_name``. Order matters: the first entry whose keywords
# appear in the lower-cased server name wins, so more specific platforms are
# listed before the families that would also match them. In particular "darwin"
# and "macos" precede the generic Unix-ish keywords, and the mobile platforms
# precede "linux" because an Android server name may legitimately mention both.
_PLATFORM_KEYWORD_TABLE: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("iphone", "ipad", "ios", "ipados"), "iPhone", "ios"),
    (("android",), "Android", "android"),
    (("macos", "mac-os", "mac os", "darwin", "osx"), "macOS", "macos"),
    (("windows", "win32", "win-32"), "Windows", "windows"),
    (("ubuntu",), "Ubuntu", "ubuntu"),
    (("debian",), "Debian", "linux"),
    (("fedora",), "Fedora", "linux"),
    (("linux",), "Linux", "linux"),
)

# Label used when no keyword matches and the daemon sent no explicit label. A
# device that reaches this state is still fully usable — the label only has to
# be stable and distinct, and :func:`deduplicate_label` guarantees distinctness.
UNKNOWN_DEVICE_LABEL = "Unknown device"
UNKNOWN_PLATFORM = "unknown"


def derive_device_identity(announcement: dict[str, Any]) -> tuple[str, str]:
    """Resolve ``(device_label, platform)`` for one register frame or body.

    Explicit daemon-supplied fields always win; derivation from ``server_name``
    is the compatibility path for daemons that predate those fields.

    Args:
        announcement: The daemon's register frame (relay WebSocket) or the
            ``POST /mcp/register`` request body. Both carry the same identity
            fields, so one resolver serves both entry points.

    Returns:
        The label and platform, neither of which is ever empty.
    """
    explicit_label = str(announcement.get("device_label") or "").strip()
    explicit_platform = str(announcement.get("platform") or "").strip().lower()

    server_name = str(announcement.get("server_name") or "").lower()
    derived_label = UNKNOWN_DEVICE_LABEL
    derived_platform = UNKNOWN_PLATFORM
    for keywords, label, platform in _PLATFORM_KEYWORD_TABLE:
        if any(keyword in server_name for keyword in keywords):
            derived_label = label
            derived_platform = platform
            break

    # The platform is resolved independently of the label: the mobile daemon
    # already sends ``platform: "ios"`` while sending no label at all, so taking
    # both fields from the same source would throw away the field the daemon
    # does supply.
    return (
        explicit_label or derived_label,
        explicit_platform or derived_platform,
    )


def deduplicate_label(
    label: str,
    existing_records: Iterable[dict[str, Any]],
    device_id: str,
) -> str:
    """Return a label unique among the user's devices.

    Two Ubuntu desktops both derive the label "Ubuntu", and a conversation that
    reports results per device cannot distinguish two identically named
    machines. The second and later devices therefore receive a counted suffix
    ("Ubuntu 2", "Ubuntu 3").

    The device's own existing record is ignored while scanning, so re-registering
    the same device — which happens on every daemon restart — keeps the label it
    already had rather than climbing to "Ubuntu 2", "Ubuntu 3", "Ubuntu 4" across
    restarts.

    Args:
        label: The label derived or supplied for this device.
        existing_records: The user's other stored registration records.
        device_id: The identifier of the device being labelled.

    Returns:
        ``label`` when no other device holds it, otherwise ``label`` with the
        lowest free counted suffix appended.
    """
    taken = {
        str(record.get("device_label") or "")
        for record in existing_records
        if record.get("device_id") != device_id
    }
    if label not in taken:
        return label
    candidate_index = 2
    while f"{label} {candidate_index}" in taken:
        candidate_index += 1
    return f"{label} {candidate_index}"


def resolve_device_for_path(file_path: str, connections: Iterable[Any]) -> Any | None:
    """Return the single connection whose allow-listed roots contain a path.

    Fan-out is the right shape for "what files do I have?", but a call that
    names one absolute host path — previewing or ingesting a specific file —
    already identifies its machine implicitly, and broadcasting that path to
    every device would waste a round-trip per device and invite a confusing
    partial failure from the machines that do not hold the file.

    Matching is by directory prefix against each connection's ``allowed_roots``,
    which is exactly the allow-list the daemon enforces, so a path that matches
    no root would have been rejected by every daemon anyway.

    Args:
        file_path: Absolute host path named by the model.
        connections: The avatar's bound connections.

    Returns:
        The one matching connection, or ``None`` when no connection matches or
        when more than one does (an ambiguity the caller reports to the model
        so the model can name the device explicitly).
    """
    matches = []
    for connection in connections:
        for root in getattr(connection, "allowed_roots", ()) or ():
            normalized_root = str(root).rstrip("/")
            if file_path == normalized_root or file_path.startswith(
                f"{normalized_root}/"
            ):
                matches.append(connection)
                break
    return matches[0] if len(matches) == 1 else None


def connection_label_map(connections: Iterable[Any]) -> dict[str, Any]:
    """Index connections by their device label for model-supplied lookup.

    Labels are compared case-insensitively and with surrounding whitespace
    stripped, because the label reaches this lookup by way of the model echoing
    back a name a human typed ("my macbook" for "My MacBook").
    """
    return {
        str(getattr(connection, "device_label", "") or "").strip().lower(): connection
        for connection in connections
    }
