"""Sign in to a site in the OWNER'S OWN browser, through the desktop daemon.

The sibling ``browser_login`` module hosts a browser inside this API and
streams it into a popup. That works for ordinary sites and fails for the
strict ones: Google answers an automated browser with "this browser or app
may not be secure", and the vendors that follow Google's lead do the same. The
owner's own browser has none of that problem — the vendor already trusts it,
and the owner is usually already signed in there.

This API server cannot reach the browser on the owner's desk. The connector
daemon (``anubis-mcp-server-ubuntu-desktop``) can, and it already holds one
outbound relay socket to this process for every machine the owner runs it on
(``tools/data_analysis/relay.py``). So the sign-in becomes three calls over
that socket:

1. ``open_browser_sign_in_page`` — the daemon opens the site's real sign-in
   page as a new tab in the owner's default browser.
2. the owner signs in there, taking as long as a password, a second factor,
   and a consent screen take.
3. ``import_browser_sign_in_session`` — the daemon reads back that ONE site's
   session and returns it, and it is stored exactly like a session captured by
   the hosted browser, so every website and browser-session tool that already
   exists works against it unchanged.

Which path a connect card takes is decided by :func:`desktop_sign_in_available`
in ``webapp``: a machine online means the owner's own browser, and no machine
online falls back to the hosted browser rather than refusing.

State: a login in progress lives in this module's dictionary, keyed by login
id, exactly as ``browser_login`` keeps its own — the same single-process
assumption the relay registry already documents.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

from src.anubis.utils.connected_accounts.browser_sessions import (
    BrowserSessionError,
    build_session_transport,
    hostname_of,
    login_page_detected,
)

logger = logging.getLogger(__name__)

LOGIN_MODE_DESKTOP_BROWSER = "desktop_browser"
CAPTURE_SOURCE = "desktop_browser"

# The daemon tools this module calls, named exactly as the daemon registers
# them in ``src/server/browser_tools.py``.
TOOL_OPEN_SIGN_IN_PAGE = "open_browser_sign_in_page"
TOOL_IMPORT_SIGN_IN_SESSION = "import_browser_sign_in_session"
TOOL_LIST_BROWSER_PROFILES = "list_local_browser_profiles"

# How long an unfinished sign-in is kept before it is forgotten. Longer than
# the hosted browser's window, because nothing is being held open for it here
# — only a row in a dictionary — and a person signing in to a bank with a
# hardware key can take a while.
DESKTOP_LOGIN_TTL_SECONDS = 1800


@dataclass
class DesktopLogin:
    """One sign-in the owner is performing in their own browser right now."""

    login_id: str
    nonce: str
    user_id: str
    assistant_id: str
    provider_name: str
    site_url: str
    name: str
    device_id: str
    device_label: str
    expires_at: float
    reconnect_account_key: str | None = None
    finished: bool = False


_desktop_logins: dict[str, DesktopLogin] = {}
_logins_lock = asyncio.Lock()


def get_desktop_login(login_id: str) -> DesktopLogin | None:
    """Return a sign-in in progress in the owner's own browser, or ``None``."""
    return _desktop_logins.get(login_id)


def forget_desktop_login(login_id: str) -> None:
    """Drop a sign-in that is finished, cancelled, or expired."""
    _desktop_logins.pop(login_id, None)


def reap_expired_desktop_logins() -> int:
    """Forget sign-ins nobody finished in time; return how many were dropped."""
    now = time.time()
    expired = [
        login_id
        for login_id, login in _desktop_logins.items()
        if login.expires_at and login.expires_at < now
    ]
    for login_id in expired:
        _desktop_logins.pop(login_id, None)
    return len(expired)


# ---------------------------------------------------------------------------
# Finding the owner's machine
# ---------------------------------------------------------------------------


def online_devices(user_id: str) -> list[Any]:
    """Every machine of this owner's that currently holds a relay socket."""
    from src.anubis.utils.tools.data_analysis import relay

    return list(relay.sessions_for_user(user_id))


def choose_device(user_id: str, device_id: str | None = None) -> Any | None:
    """Return the machine to open the sign-in page on, or ``None`` when none is online.

    A named machine wins when it is that owner's and it is online. Otherwise
    the first machine is taken, which ``sessions_for_user`` orders by label so
    the same machine is chosen turn after turn.
    """
    sessions = online_devices(user_id)
    if not sessions:
        return None
    wanted = str(device_id or "").strip()
    if wanted:
        for session in sessions:
            if session.device_id == wanted:
                return session
        return None
    return sessions[0]


def desktop_sign_in_available(user_id: str, device_id: str | None = None) -> bool:
    """Whether a sign-in can happen in this owner's own browser right now."""
    return choose_device(user_id, device_id) is not None


async def call_device_tool(session: Any, tool_name: str, tool_arguments: dict[str, Any]) -> Any:
    """Call one tool on the owner's machine over its relay socket."""
    from src.anubis.utils.tools.data_analysis.mcp_client import call_mcp_filesystem_tool
    from src.anubis.utils.tools.data_analysis.relay import connection_from_session

    connection = connection_from_session(session)
    try:
        return await call_mcp_filesystem_tool(connection, tool_name, tool_arguments)
    except RuntimeError as call_error:
        message = str(call_error)
        if tool_name in message and "does not expose" in message:
            raise BrowserSessionError(
                409,
                f"The connector on {session.device_label} is too old to sign in "
                "through your own browser. Update it and try again.",
            ) from call_error
        raise BrowserSessionError(
            502, f"{session.device_label} could not be reached: {message}"
        ) from call_error


def _tool_failure_detail(result: Any) -> str | None:
    """Return the owner-safe error a daemon tool reported, or ``None``.

    A tool that raises reaches this side as text (or as a dict carrying an
    error), rather than as an exception, because it crossed a relay socket and
    a Model Context Protocol result on the way.
    """
    if isinstance(result, dict):
        for key in ("error", "detail", "message"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(result, str) and result.strip():
        return result.strip()
    return None


# ---------------------------------------------------------------------------
# Starting and finishing the sign-in
# ---------------------------------------------------------------------------


async def start_desktop_login(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    provider: Any,
    site_url: str | None = None,
    name: str | None = None,
    reconnect_account_key: str | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    """Open the site's sign-in page in the owner's own browser on their machine.

    Returns the card's next step: the login id, the signed token the card
    presents when it says the owner is done, and which machine the tab was
    opened on. ``reconnect_account_key`` names an existing record whose
    session lapsed, which the finished sign-in refreshes instead of adding a
    second record.
    """
    from src.anubis.utils.connected_accounts.browser_login import sign_login_token

    reap_expired_desktop_logins()
    start_url = str(site_url or getattr(provider, "login_url", None) or "").strip()
    if not start_url.startswith("http"):
        raise BrowserSessionError(400, "A site address to sign in to is required.")
    session = choose_device(user_id, device_id)
    if session is None:
        raise BrowserSessionError(
            409,
            "No machine of yours is online to open your browser on. Start the "
            "Neural Nexus connector on the machine you browse with, or sign in "
            "in a hosted window instead.",
        )
    result = await call_device_tool(session, TOOL_OPEN_SIGN_IN_PAGE, {"url": start_url})
    failure = _tool_failure_detail(result)
    if failure or not isinstance(result, dict) or not result.get("opened"):
        raise BrowserSessionError(
            502,
            failure
            or f"The sign-in page could not be opened on {session.device_label}.",
        )
    login_id = secrets.token_urlsafe(16)
    nonce = secrets.token_urlsafe(8)
    async with _logins_lock:
        _desktop_logins[login_id] = DesktopLogin(
            login_id=login_id,
            nonce=nonce,
            user_id=user_id,
            assistant_id=assistant_id,
            provider_name=provider.name,
            site_url=str(result.get("url") or start_url),
            name=str(name or provider.display_name),
            device_id=session.device_id,
            device_label=session.device_label,
            expires_at=time.time() + DESKTOP_LOGIN_TTL_SECONDS,
            reconnect_account_key=str(reconnect_account_key or "").strip() or None,
        )
    token = sign_login_token(context, login_id=login_id, user_id=user_id, nonce=nonce)
    site_hostname = hostname_of(str(result.get("url") or start_url))
    return {
        "login_id": login_id,
        "nonce": nonce,
        "login_mode": LOGIN_MODE_DESKTOP_BROWSER,
        "login_token": token,
        "provider": provider.name,
        "site_url": str(result.get("url") or start_url),
        "site_hostname": site_hostname,
        "device_id": session.device_id,
        "device_label": session.device_label,
        "expires_in": DESKTOP_LOGIN_TTL_SECONDS,
        "instructions": (
            f"{site_hostname} is open in your browser on {session.device_label}. "
            "Sign in there, then say you are done."
        ),
    }


async def finish_desktop_login(
    context: Any,
    *,
    login_id: str,
    user_id: str,
    existing_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bring the signed-in session back from the owner's browser and store it.

    The record produced is the same shape the hosted browser produces, down to
    the credential mechanism, so nothing downstream needs to know which browser
    the owner signed in with.
    """
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
    )
    from src.anubis.utils.tools.data_analysis import relay

    login = _desktop_logins.get(login_id)
    if login is None or login.user_id != user_id:
        raise BrowserSessionError(404, "No sign-in in your browser with that id is open.")
    if login.finished:
        raise BrowserSessionError(409, "This sign-in was already finished.")
    session = relay.get_session(login.device_id)
    if session is None or session.user_id != user_id:
        raise BrowserSessionError(
            409,
            f"{login.device_label} went offline before the sign-in could be kept. "
            "Start the connector there and sign in again.",
        )
    result = await call_device_tool(
        session, TOOL_IMPORT_SIGN_IN_SESSION, {"site_url": login.site_url}
    )
    failure = _tool_failure_detail(result)
    if failure:
        raise BrowserSessionError(400, failure)
    if not isinstance(result, dict) or not isinstance(result.get("storage_state"), dict):
        raise BrowserSessionError(
            502, f"{login.device_label} returned no session for {login.site_url}."
        )
    storage_state = result["storage_state"]
    cookie_count = int(result.get("cookie_count") or len(storage_state.get("cookies") or []))
    if not cookie_count:
        raise BrowserSessionError(
            400,
            f"Your browser holds no session for {hostname_of(login.site_url)} yet. "
            "Finish signing in there, then say you are done.",
        )
    login.finished = True
    forget_desktop_login(login_id)

    provider = get_provider(login.provider_name)
    site_hostname = hostname_of(login.site_url)
    home_url = getattr(provider, "home_url", None)
    if not home_url or not str(home_url).startswith("http"):
        home_url = login.site_url
    # The owner signing in on the site's own page IS the evidence of a session
    # here — there is no page in this process to inspect — so the heuristic
    # only runs over the address the daemon reported, never over page markup.
    heuristic_signed_in = not login_page_detected(
        str(home_url), "", site_hostname=site_hostname
    )
    # One record per SIGN-IN, never per site: the owner may hold two accounts
    # on one site. A "sign in again" refreshes the named record instead.
    existing = next(
        (
            record
            for record in existing_records
            if login.reconnect_account_key
            and record.get("account_key") == login.reconnect_account_key
        ),
        None,
    )
    if existing is not None:
        address = str(existing.get("account_address") or site_hostname)
        key = str(existing.get("account_key"))
        label = str(existing.get("display_label") or login.name or site_hostname)
    else:
        address = f"{site_hostname}#{login.login_id[:8]}"
        key = account_key(provider.name, address)
        label = deduplicate_label(login.name or site_hostname, existing_records, key)
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    record = build_account_record(
        provider=provider,
        account_address=address,
        display_label=label,
        encrypted_secret=None,
        assistant_id=login.assistant_id,
        transport=build_session_transport(
            storage_state=storage_state,
            context=context,
            home_url=str(home_url),
            final_url=str(home_url),
            site_url=login.site_url,
            heuristic_signed_in=heuristic_signed_in,
            recipe_key=getattr(provider, "recipe_key", None),
            extra={
                "captured_from": CAPTURE_SOURCE,
                "device_id": login.device_id,
                "device_label": login.device_label,
                "browser_profile": str(profile.get("identifier") or ""),
            },
        ),
    )
    record["user_id"] = user_id
    # The provider row may sign in through OAuth when an application exists;
    # this record came from a browser session, and the tools must treat it as
    # one.
    record["credential_mechanism"] = "browser_session"
    logger.info(
        "Kept a %s session for %s from %s (%s cookies)",
        provider.name,
        site_hostname,
        login.device_label,
        cookie_count,
    )
    return {
        "record": record,
        "nonce": login.nonce,
        "heuristic_signed_in": heuristic_signed_in,
        "final_url": str(home_url),
        "cookie_count": cookie_count,
        "device_label": login.device_label,
        "browser_profile": str(profile.get("identifier") or ""),
    }


def cancel_desktop_login(login_id: str, user_id: str) -> bool:
    """Forget a sign-in the owner abandoned; the tab in their browser is theirs."""
    login = _desktop_logins.get(login_id)
    if login is None or login.user_id != user_id:
        return False
    forget_desktop_login(login_id)
    return True


async def list_device_browser_profiles(user_id: str, device_id: str | None = None) -> list[dict[str, Any]]:
    """Return the browsers on the owner's machine a session could be kept from."""
    session = choose_device(user_id, device_id)
    if session is None:
        return []
    result = await call_device_tool(session, TOOL_LIST_BROWSER_PROFILES, {})
    if isinstance(result, list):
        return [profile for profile in result if isinstance(profile, dict)]
    return []
