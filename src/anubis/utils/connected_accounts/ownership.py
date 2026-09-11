"""Proving that a connected social account belongs to the personal avatar's person.

This module is a gate, not a convenience. A personal avatar reconstructs one
real person, and the whole value of subscribing to that person's accounts is
that everything arriving through them is *theirs*. An account nobody proved
they own is a stranger's voice, and letting a stranger's posts land in the
``identity`` and ``quote`` namespaces would teach the avatar to speak as
somebody else. So ownership is checked once at connect time, recorded on the
account record, and consulted again by every later path that could write
identity — the initial crawl and the subscription intake both.

**The three proofs, and why they differ in strength.**

- An OAuth account proves itself. The owner completed the vendor's own sign-in
  and the vendor handed back the identity of the account that signed in; there
  is nothing further to check, and the handle the vendor reported is the handle
  we trust.
- A browser-session account proves itself the same way, one step later. Control
  of a signed-in session is control of the account, so reading the signed-in
  handle off the account's own profile page is the proof. It is weaker only in
  that we read it rather than being told it.
- A feed or a profile address proves nothing at all. Anyone can name anyone's
  blog. These stay ``unproven`` until the owner links them back from an account
  that IS proven (a ``rel="me"`` link, the IndieAuth convention) or places a
  one-time token we issue into the feed. Until then the source may be connected
  and read in conversation, but it never reaches identity.

That last distinction is the reason this module exists as its own file rather
than as a flag set inside each connect handler: the gate has to be readable in
one place to be trustworthy.
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.connected_accounts.providers import (
    KIND_SOCIAL,
    MECHANISM_BROWSER_SESSION,
    MECHANISM_OAUTH,
    MECHANISM_URL_ONLY,
    get_provider,
)

logger = logging.getLogger(__name__)

OWNERSHIP_PROVEN = "proven"
OWNERSHIP_UNPROVEN = "unproven"

METHOD_OAUTH_IDENTITY = "oauth_identity"
METHOD_SIGNED_IN_SESSION = "signed_in_session"
METHOD_LINKED_FROM_PROVEN = "linked_from_proven"
METHOD_VERIFICATION_TOKEN = "verification_token"

# Prefix of the token the owner pastes into a feed description or profile bio
# to prove a no-login source is theirs. Distinctive on purpose: it has to be
# findable in a page of arbitrary text without matching anything else.
VERIFICATION_TOKEN_PREFIX = "neural-nexus-verify-"


def new_verification_token() -> str:
    """Mint the token an owner places in an unprovable source to claim it."""
    return f"{VERIFICATION_TOKEN_PREFIX}{secrets.token_hex(8)}"


def build_ownership(
    *,
    state: str,
    method: str | None = None,
    handle: str | None = None,
    profile_url: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Assemble the ``ownership`` block stored on an account record."""
    return {
        "state": state,
        "method": method,
        "handle": (handle or "").strip().lstrip("@") or None,
        "profile_url": profile_url,
        "detail": detail,
        "proven_at": datetime.now(UTC).isoformat() if state == OWNERSHIP_PROVEN else None,
    }


def ownership_of(record: dict[str, Any]) -> dict[str, Any]:
    """Return a record's ownership block, defaulting to unproven.

    Records written before this feature carry no block at all. They default to
    unproven rather than to proven, because the safe direction for a gate that
    decides whose words become the avatar's is to refuse what it cannot vouch
    for. An owner reconnecting the account proves it in one click.
    """
    ownership = record.get("ownership")
    if isinstance(ownership, dict) and ownership.get("state") in {
        OWNERSHIP_PROVEN,
        OWNERSHIP_UNPROVEN,
    }:
        return ownership
    return build_ownership(state=OWNERSHIP_UNPROVEN, detail="No ownership proof recorded.")


def is_ownership_proven(record: dict[str, Any]) -> bool:
    """Whether this account's content may be used to build identity."""
    return ownership_of(record).get("state") == OWNERSHIP_PROVEN


def is_owned_by_personal_avatar(
    record: dict[str, Any], *, personal_avatar_id: str | None
) -> bool:
    """Answer the single gate question every identity-writing path asks.

    Three things must hold together, and all three are about the same question —
    is this the avatar's own person publishing? The account must be a social
    identity (a mailbox proves nothing about a likeness), it must be bound to
    the personal avatar being updated, and its ownership must have been proven.
    """
    if record.get("kind") != KIND_SOCIAL:
        return False
    if not is_ownership_proven(record):
        return False
    bound_assistant_id = str(record.get("assistant_id") or "")
    wanted = str(personal_avatar_id or "")
    if not wanted or not bound_assistant_id:
        return False
    return bound_assistant_id == wanted


def refusal_reason(
    record: dict[str, Any], *, personal_avatar_id: str | None
) -> str | None:
    """Explain why an account may not feed identity, or ``None`` when it may."""
    if record.get("kind") != KIND_SOCIAL:
        return (
            f"{record.get('display_label')} is not a social account, so it cannot "
            "establish what this avatar's person publishes."
        )
    bound_assistant_id = str(record.get("assistant_id") or "")
    if not personal_avatar_id or bound_assistant_id != str(personal_avatar_id):
        return (
            f"{record.get('display_label')} is not connected to this personal "
            "avatar."
        )
    if not is_ownership_proven(record):
        return (
            f"Nobody has proved {record.get('display_label')} belongs to you yet, "
            "so nothing it publishes will be used to build your avatar."
        )
    return None


async def prove_ownership(
    context: Any,
    store: Any,
    record: dict[str, Any],
    *,
    identity_hint: str | None = None,
) -> dict[str, Any]:
    """Establish ownership for a freshly connected account.

    Never raises: a proof that cannot be completed leaves the account connected
    and unproven, which is a state the owner can see and fix, rather than
    failing the connection they just made.
    """
    provider = get_provider(str(record.get("provider") or ""))
    if provider is None or provider.kind != KIND_SOCIAL:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail="Only a social account can carry an ownership proof.",
        )

    mechanism = str(record.get("credential_mechanism") or "")
    try:
        if mechanism == MECHANISM_OAUTH:
            return await _prove_by_oauth_identity(context, store, record, provider)
        if mechanism == MECHANISM_BROWSER_SESSION:
            return await _prove_by_signed_in_session(context, store, record, provider)
        if mechanism == MECHANISM_URL_ONLY:
            return await _prove_by_back_link(context, store, record, identity_hint)
    except Exception as proof_error:  # noqa: BLE001 - a failed proof is a state
        logger.warning(
            "Ownership proof failed for %s: %s", record.get("account_key"), proof_error
        )
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail=f"The ownership check could not be completed: {proof_error}",
        )
    return build_ownership(
        state=OWNERSHIP_UNPROVEN,
        detail=f"No ownership proof is defined for {mechanism!r} accounts.",
    )


async def _prove_by_oauth_identity(
    context: Any, store: Any, record: dict[str, Any], provider: Any
) -> dict[str, Any]:
    """Trust the vendor's own statement of who signed in; that is the proof."""
    handle = str(record.get("account_address") or "").strip()
    if not handle:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail="The vendor returned no account identity to record.",
        )
    profile_url = provider.profile_url_for(handle)
    if provider.name == "youtube":
        channel = await _read_youtube_channel(context, store, record)
        if channel:
            return build_ownership(
                state=OWNERSHIP_PROVEN,
                method=METHOD_OAUTH_IDENTITY,
                handle=channel.get("title") or handle,
                profile_url=channel.get("channel_url"),
                detail=f"Signed in to the YouTube channel {channel.get('title')!r}.",
            )
    return build_ownership(
        state=OWNERSHIP_PROVEN,
        method=METHOD_OAUTH_IDENTITY,
        handle=handle,
        profile_url=profile_url,
        detail=f"Signed in to {provider.display_name} as {handle}.",
    )


async def _read_youtube_channel(
    context: Any, store: Any, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Read the signed-in channel, its id, and its uploads playlist.

    The uploads playlist is the crawl seed and the channel id is the WebSub
    topic, so both are captured here, at the one moment a fresh token is
    guaranteed to exist.
    """
    from src.anubis.utils.connected_accounts.vendor_api_tools import _bearer, _get_json

    token, failure = await _bearer(context, store, record)
    if failure or not token:
        return None
    status_code, document = await _get_json(
        "https://www.googleapis.com/youtube/v3/channels",
        token,
        params={"part": "snippet,contentDetails", "mine": "true"},
    )
    if status_code >= 400:
        return None
    items = (document or {}).get("items") or []
    if not items:
        return None
    entry = items[0]
    snippet = entry.get("snippet") or {}
    content_details = entry.get("contentDetails") or {}
    channel_id = entry.get("id")
    uploads_playlist_id = (content_details.get("relatedPlaylists") or {}).get("uploads")
    custom_url = snippet.get("customUrl")
    return {
        "channel_id": channel_id,
        "uploads_playlist_id": uploads_playlist_id,
        "title": snippet.get("title"),
        "channel_url": (
            f"https://www.youtube.com/{custom_url}"
            if custom_url
            else f"https://www.youtube.com/channel/{channel_id}"
        ),
    }


async def _prove_by_signed_in_session(
    context: Any, store: Any, record: dict[str, Any], provider: Any
) -> dict[str, Any]:
    """Read the signed-in handle off the account's own page.

    Holding a live session for an account is holding the account. The handle is
    read rather than asserted so the record names the real account and not
    whatever the owner typed.
    """
    home_url = provider.home_url or provider.login_url
    if not home_url:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail=f"{provider.display_name} declares no page to read the handle from.",
        )
    page_html = await _read_page_html(context, store, record, home_url)
    handle = _extract_handle(page_html, provider.name)
    if not handle:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail=(
                f"Signed in to {provider.display_name}, but the account handle "
                "could not be read back from the page."
            ),
        )
    return build_ownership(
        state=OWNERSHIP_PROVEN,
        method=METHOD_SIGNED_IN_SESSION,
        handle=handle,
        profile_url=provider.profile_url_for(handle),
        detail=f"Signed in to {provider.display_name} as {handle}.",
    )


async def _read_page_html(
    context: Any, store: Any, record: dict[str, Any], url: str
) -> str:
    """Fetch one page inside the account's signed-in session, as raw HTML.

    Raw HTML rather than the rendered text the chat tools return, because the
    signed-in handle is usually in an embedded JSON blob rather than in
    anything the page displays.
    """
    from src.anubis.utils.connected_accounts.browser_sessions import (
        open_session,
        release_session,
    )

    handle = await open_session(
        context, store, str(record.get("user_id") or ""), record, lease=True
    )
    try:
        async with handle.lock:
            await handle.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return await handle.page.content()
    finally:
        release_session(handle)


_HANDLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "instagram": (r'"username"\s*:\s*"([A-Za-z0-9._]{1,30})"',),
    "tiktok": (r'"uniqueId"\s*:\s*"([A-Za-z0-9._]{1,30})"', r"@([A-Za-z0-9._]{2,30})"),
    "twitch": (r'"login"\s*:\s*"([A-Za-z0-9_]{3,25})"',),
    "facebook": (r'"USER_ID"\s*:\s*"(\d{5,})"',),
    "linkedin": (r"/in/([A-Za-z0-9\-]{3,100})",),
}


def _extract_handle(page_text: str, provider_name: str) -> str | None:
    """Pull the signed-in account's handle out of a page the session fetched."""
    for pattern in _HANDLE_PATTERNS.get(provider_name, ()):
        match = re.search(pattern, page_text)
        if match:
            return match.group(1)
    return None


async def _prove_by_back_link(
    context: Any, store: Any, record: dict[str, Any], identity_hint: str | None
) -> dict[str, Any]:
    """Look for evidence on the source itself, the only proof a no-login source has.

    Two accepted forms, both of which require the owner to have write access to
    the thing they are claiming: a ``rel="me"`` link pointing at an account
    already proven, or the one-time token we issued placed in the page.
    """
    transport = record.get("transport") or {}
    site_url = str(transport.get("site_url") or record.get("account_address") or "")
    if not site_url:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN, detail="The source has no address to check."
        )

    page_text = await _fetch_text(site_url)
    if not page_text:
        return build_ownership(
            state=OWNERSHIP_UNPROVEN,
            detail=f"{site_url} could not be read to check for a proof.",
        )

    expected_token = str(transport.get("verification_token") or "")
    if expected_token and expected_token in page_text:
        return build_ownership(
            state=OWNERSHIP_PROVEN,
            method=METHOD_VERIFICATION_TOKEN,
            handle=identity_hint,
            profile_url=site_url,
            detail="The verification token was found on the page.",
        )

    if identity_hint:
        for match in re.finditer(
            r'rel=["\'][^"\']*\bme\b[^"\']*["\'][^>]*href=["\']([^"\']+)["\']'
            r'|href=["\']([^"\']+)["\'][^>]*rel=["\'][^"\']*\bme\b[^"\']*["\']',
            page_text,
            re.IGNORECASE,
        ):
            linked = match.group(1) or match.group(2) or ""
            if identity_hint.lower().lstrip("@") in linked.lower():
                return build_ownership(
                    state=OWNERSHIP_PROVEN,
                    method=METHOD_LINKED_FROM_PROVEN,
                    handle=identity_hint,
                    profile_url=site_url,
                    detail=f"{site_url} links back to {linked} with rel=\"me\".",
                )

    return build_ownership(
        state=OWNERSHIP_UNPROVEN,
        detail=(
            "Add the verification token to the page, or link back to an account "
            "you have already connected, and check again."
        ),
    )


async def _fetch_text(url: str, *, timeout: float = 20.0) -> str:
    """Fetch a page as text, returning an empty string on any failure."""
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            response = await client.get(url)
            if response.status_code >= 400:
                return ""
            return response.text
    except Exception:  # noqa: BLE001 - an unreachable page proves nothing
        return ""
