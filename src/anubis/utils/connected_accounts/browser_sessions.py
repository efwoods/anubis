"""Signed-in browser sessions the avatar keeps for sites with no OAuth.

Most business dashboards have no API for a third party and no OAuth: LangSmith,
the OpenAI platform, the Anthropic console, and any custom site. The owner
signs in to those on the site's own login page in a live browser the API
hosts (``browser_login.py``); what this module keeps afterwards is the
RESULT of that sign-in — the browser's storage state (cookies and local
storage) — encrypted in the account record, so later turns open a browser
context that is already signed in.

Durability: a keepalive task revisits every connected site on a schedule
(``BROWSER_SESSION_KEEPALIVE_HOURS``), re-saves the refreshed cookies, and
notices when a visit lands on a login page. A lapsed session is not
silently dropped: the record is marked ``needs_reconnect``, the inbox gets
a notify item, and the avatar re-raises the connect card on the next
relevant turn.

One dedicated headless Chromium serves every session in this process,
separate from the per-conversation browsers of ``browser_tools.py`` (whose
toolkit assumes the browser's first context is the conversation's own).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SESSION_TRANSPORT_KEY = "browser_session"
DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

# Presents the page's JavaScript environment as an ordinary desktop browser.
ORDINARY_BROWSER_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
window.chrome = window.chrome || { runtime: {}, loadTimes: function () {}, csi: function () {} };
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
  window.navigator.permissions.query = (parameters) =>
    parameters && parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : originalQuery(parameters);
}
"""

_LOGIN_PATH_PATTERN = re.compile(
    r"/(login|log-in|signin|sign-in|sign_in|auth|authenticate|session/new|account/login)",
    re.IGNORECASE,
)
_LOGIN_HOST_PATTERN = re.compile(
    r"^(accounts|auth|login|signin|id|sso)\.", re.IGNORECASE
)
_PASSWORD_INPUT_PATTERN = re.compile(r"<input[^>]+type=[\"']?password", re.IGNORECASE)
_SIGNED_IN_MARKER_PATTERN = re.compile(
    r"(logout|log-out|sign-out|signout|sign_out|data-logout|/settings|/account|/profile|avatar)",
    re.IGNORECASE,
)


class BrowserSessionError(Exception):
    """A session could not be opened or used; ``detail`` is owner-safe."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Carry the HTTP status the route should answer with, and why."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class BrowserSessionExpired(Exception):
    """The site no longer treats the stored session as signed in."""


@dataclass
class SessionHandle:
    """One open, signed-in browser context and the bookkeeping around the context."""

    account_key: str
    user_id: str
    context: Any
    page: Any
    last_used_monotonic: float = field(default_factory=time.monotonic)
    lease_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        """Mark the session as just used."""
        self.last_used_monotonic = time.monotonic()


# ---------------------------------------------------------------------------
# The shared session browser
# ---------------------------------------------------------------------------

_playwright_driver: Any | None = None
_session_browser: Any | None = None
_browser_lock = asyncio.Lock()
_sessions: dict[str, SessionHandle] = {}
_sessions_lock = asyncio.Lock()


async def _ensure_driver() -> Any:
    global _playwright_driver
    if _playwright_driver is None:
        from playwright.async_api import async_playwright

        _playwright_driver = await async_playwright().start()
    return _playwright_driver


def launch_arguments(context: Any) -> dict[str, Any]:
    """Return the Chromium launch arguments shared with the conversation browsers."""
    # Sign-in pages of some vendors refuse a browser they recognise as
    # automated. The flags and the init script below present the session
    # browser as the ordinary Chromium the owner would sign in with: no
    # automation banner, no ``navigator.webdriver`` marker, a real user
    # agent, a plugin list and a ``window.chrome`` object like a desktop.
    arguments: dict[str, Any] = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--lang=en-US,en",
        ],
        "ignore_default_args": ["--enable-automation"],
    }
    executable = getattr(context, "browser_chromium_executable_path", None)
    if executable:
        arguments["executable_path"] = executable
    return arguments


async def session_browser(context: Any) -> Any:
    """Return the process-wide session browser, launching one on first use."""
    global _session_browser
    async with _browser_lock:
        if _session_browser is not None and _session_browser.is_connected():
            return _session_browser
        driver = await _ensure_driver()
        try:
            _session_browser = await driver.chromium.launch(**launch_arguments(context))
        except Exception as launch_error:
            raise BrowserSessionError(
                503, f"A browser could not be started on this server: {launch_error}"
            ) from launch_error
        return _session_browser


async def new_context(
    context: Any,
    *,
    storage_state: dict[str, Any] | None = None,
    user_agent: str | None = None,
) -> Any:
    """Open a fresh browser context, optionally restoring a stored session."""
    browser = await session_browser(context)
    options: dict[str, Any] = {
        "viewport": dict(DEFAULT_VIEWPORT),
        "user_agent": user_agent or DEFAULT_USER_AGENT,
        "locale": "en-US",
    }
    if storage_state:
        options["storage_state"] = storage_state
    browser_context = await browser.new_context(**options)
    try:
        await browser_context.add_init_script(ORDINARY_BROWSER_INIT_SCRIPT)
    except Exception:
        logger.debug(
            "Could not install the ordinary-browser init script", exc_info=True
        )
    return browser_context


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def session_details(record: dict[str, Any]) -> dict[str, Any]:
    """Return the ``browser_session`` block of a record (empty when absent)."""
    transport = record.get("transport") or {}
    details = transport.get(SESSION_TRANSPORT_KEY) or {}
    return dict(details) if isinstance(details, dict) else {}


def home_url_for(record: dict[str, Any], provider: Any | None = None) -> str:
    """Return the address a session visits first: the stored home, else the site."""
    details = session_details(record)
    transport = record.get("transport") or {}
    for candidate in (
        details.get("home_url"),
        transport.get("site_url"),
        getattr(provider, "home_url", None),
        getattr(provider, "login_url", None),
    ):
        if candidate and str(candidate).startswith("http"):
            return str(candidate)
    hostname = str(record.get("account_address") or "").split("#", 1)[0]
    return f"https://{hostname}/" if hostname else ""


def hostname_of(url: str) -> str:
    """Return the lower-cased hostname of a URL ('' when unparsable)."""
    try:
        return (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return ""


def same_site(url: str, allowed_hostname: str) -> bool:
    """Whether a URL is on the connected site (the host or a subdomain)."""
    host = hostname_of(url)
    allowed = str(allowed_hostname or "").lower().split("#", 1)[0]
    if not host or not allowed:
        return False
    return (
        host == allowed or host.endswith("." + allowed) or allowed.endswith("." + host)
    )


def decrypt_storage_state(record: dict[str, Any], context: Any) -> dict[str, Any]:
    """Return the stored storage state of a session record."""
    from src.anubis.utils.secret_store import decrypt_secret

    ciphertext = session_details(record).get("storage_state_encrypted")
    if not ciphertext:
        raise BrowserSessionExpired("No signed-in session is stored for this account.")
    raw = decrypt_secret(str(ciphertext), context)
    try:
        state = json.loads(raw)
    except Exception as decode_error:
        raise BrowserSessionExpired(
            "The stored session could not be read."
        ) from decode_error
    return state if isinstance(state, dict) else {}


def build_session_transport(
    *,
    storage_state: dict[str, Any],
    context: Any,
    home_url: str,
    final_url: str,
    site_url: str | None,
    heuristic_signed_in: bool,
    user_agent: str = DEFAULT_USER_AGENT,
    recipe_key: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the transport block of a browser-session record."""
    from src.anubis.utils.secret_store import encrypt_secret

    cookies = storage_state.get("cookies") or []
    return {
        SESSION_TRANSPORT_KEY: {
            "storage_state_encrypted": encrypt_secret(
                json.dumps(storage_state), context
            ),
            "home_url": home_url,
            "final_url": final_url,
            "user_agent": user_agent,
            "saved_at": datetime.now(UTC).isoformat(),
            "cookie_count": len(cookies),
            "heuristic_signed_in": bool(heuristic_signed_in),
        },
        "site_url": site_url or home_url,
        "hostname": hostname_of(final_url or home_url),
        "recipe_key": recipe_key,
        **dict(extra or {}),
    }


# ---------------------------------------------------------------------------
# Login-page detection
# ---------------------------------------------------------------------------


def login_page_detected(
    url: str, html: str, *, site_hostname: str | None = None
) -> bool:
    """Guess whether a page is a sign-in page rather than a signed-in one.

    Best effort, used by the keepalive and reported on the card; the owner's
    explicit "I'm signed in" is authoritative during the login itself. A page
    counts as a login page when the address looks like one, or when a password
    input is present with no sign of a signed-in session (a logout link, an
    account or settings link).
    """
    address = str(url or "")
    parsed = urlparse(address)
    if _LOGIN_PATH_PATTERN.search(parsed.path or ""):
        return True
    host = (parsed.hostname or "").lower()
    if site_hostname and host and not same_site(address, site_hostname):
        # Redirected to a different host entirely (an identity provider).
        if _LOGIN_HOST_PATTERN.search(host):
            return True
    text = str(html or "")
    if _PASSWORD_INPUT_PATTERN.search(text) and not _SIGNED_IN_MARKER_PATTERN.search(
        text
    ):
        return True
    return False


# A bot wall is not a lapsed session, and telling them apart is the difference
# between "sign in again" (useless — the session is fine) and "this site refuses
# automated visits, use its API instead" (actionable). Kept deliberately short:
# every entry is a phrase these services put on the interstitial itself, and a
# loose list here would misread an ordinary page as a block.
_BOT_WALL_MARKERS: tuple[str, ...] = (
    "attention required",
    "just a moment",
    "cf-browser-verification",
    "cf_chl_opt",
    "checking your browser before accessing",
    "verify you are human",
    "enable javascript and cookies to continue",
    "access denied",
    "request unsuccessful. incapsula",
    "pardon our interruption",
)

# Statuses a bot wall answers with. A 401 is deliberately NOT here: that is an
# authentication problem and belongs to the reconnect path.
_BOT_WALL_STATUSES: frozenset[int] = frozenset({403, 429, 503})


def bot_wall_detected(html: str, *, status_code: int | None = None) -> bool:
    """Whether a response is a bot-protection interstitial rather than the page.

    Requires BOTH a status a wall answers with AND a marker phrase in the body,
    because either alone is ordinary: plenty of real pages are 403 for a signed-out
    reader, and plenty of real prose contains "access denied". When no status is
    available (a rendered page rather than a fetch) the markers decide alone,
    which is why the marker list stays narrow.
    """
    body = str(html or "").lower()
    if not body:
        return False
    matched = any(marker in body for marker in _BOT_WALL_MARKERS)
    if not matched:
        return False
    if status_code is None:
        return True
    return int(status_code) in _BOT_WALL_STATUSES


def bot_wall_advice(record: dict[str, Any], hostname: str) -> str:
    """Return what to tell the owner when a site refuses the visit.

    Names the site and points at the route that does work, because "blocked" on
    its own leaves them with nothing to do. A vendor with a documented key page
    is named specifically; anything else gets the general answer.
    """
    from src.anubis.utils.connected_accounts.providers import provider_for_host

    known = provider_for_host(hostname)
    if known is not None and getattr(known, "credential_mechanism", "") == "api_key":
        return (
            f"{hostname} refuses automated visits. It issues an API key, which is "
            f"the route it supports — connect {known.display_name} with a key "
            "instead of a signed-in session."
        )
    return (
        f"{hostname} refuses automated visits, so the signed-in session cannot "
        "read it. The session itself is fine — signing in again will not help. "
        "If the site publishes an API key or a connector, connect that instead."
    )


# ---------------------------------------------------------------------------
# Opening and keeping sessions
# ---------------------------------------------------------------------------


def _max_open(context: Any) -> int:
    return int(getattr(context, "browser_session_max_open", None) or 6)


def _idle_seconds(context: Any) -> int:
    return int(getattr(context, "browser_session_idle_seconds", None) or 900)


async def _evict_idle(context: Any) -> None:
    now = time.monotonic()
    idle_limit = _idle_seconds(context)
    for key in list(_sessions):
        handle = _sessions[key]
        if handle.lease_count > 0:
            continue
        if now - handle.last_used_monotonic > idle_limit:
            await _close_handle(key, "idle")
    while len(_sessions) > _max_open(context):
        candidates = [
            handle for handle in _sessions.values() if handle.lease_count == 0
        ]
        if not candidates:
            break
        oldest = min(candidates, key=lambda handle: handle.last_used_monotonic)
        await _close_handle(oldest.account_key, "cap")


async def _close_handle(key: str, reason: str) -> None:
    handle = _sessions.pop(key, None)
    if handle is None:
        return
    try:
        await handle.context.close()
    except Exception:
        logger.debug("Could not close session %s (%s)", key, reason, exc_info=True)


async def open_session(
    context: Any,
    store: Any,
    user_id: str,
    record: dict[str, Any],
    *,
    lease: bool = True,
) -> SessionHandle:
    """Return an open, signed-in browser context for a record (opening one on demand)."""
    key = str(record.get("account_key") or "")
    async with _sessions_lock:
        await _evict_idle(context)
        handle = _sessions.get(key)
        if handle is None:
            storage_state = decrypt_storage_state(record, context)
            details = session_details(record)
            browser_context = await new_context(
                context,
                storage_state=storage_state,
                user_agent=details.get("user_agent"),
            )
            page = await browser_context.new_page()
            handle = SessionHandle(
                account_key=key, user_id=user_id, context=browser_context, page=page
            )
            _sessions[key] = handle
        if lease:
            handle.lease_count += 1
        handle.touch()
        return handle


def release_session(handle: SessionHandle) -> None:
    """Give a lease back; the session stays open for the idle window."""
    handle.lease_count = max(0, handle.lease_count - 1)
    handle.touch()


async def persist_session_state(
    context: Any,
    store: Any,
    user_id: str,
    record: dict[str, Any],
    handle: SessionHandle,
) -> dict[str, Any]:
    """Re-encrypt the live storage state into the record and save the record."""
    from src.anubis.utils.connected_accounts.store import save_connected_account
    from src.anubis.utils.secret_store import encrypt_secret

    storage_state = await handle.context.storage_state()
    details = session_details(record)
    details["storage_state_encrypted"] = encrypt_secret(
        json.dumps(storage_state), context
    )
    details["saved_at"] = datetime.now(UTC).isoformat()
    details["cookie_count"] = len(storage_state.get("cookies") or [])
    transport = dict(record.get("transport") or {})
    transport[SESSION_TRANSPORT_KEY] = details
    record["transport"] = transport
    record["last_verified_at"] = details["saved_at"]
    await save_connected_account(store, user_id, record)
    return details


async def forget_session(account_key: str) -> None:
    """Close the open context of one account (after a disconnect or reconnect)."""
    async with _sessions_lock:
        await _close_handle(str(account_key), "forget")


async def shutdown_browser_sessions() -> None:
    """Close every session and the shared browser (lifespan shutdown)."""
    global _session_browser
    async with _sessions_lock:
        for key in list(_sessions):
            await _close_handle(key, "shutdown")
    async with _browser_lock:
        if _session_browser is not None:
            try:
                await _session_browser.close()
            except Exception:
                logger.debug("Could not close the session browser", exc_info=True)
            _session_browser = None


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------

_keepalive_semaphore = asyncio.Semaphore(2)


async def _notify_reconnect(record: dict[str, Any], user_id: str, reason: str) -> None:
    """Tell the owner through the inbox that a site needs a new sign-in."""
    try:
        from src.anubis.utils.inbox.repository import (
            STATE_PENDING_OWNER,
            get_inbox_repository,
        )
    except Exception:
        return
    repository = get_inbox_repository()
    if repository is None:
        return
    label = record.get("display_label") or record.get("provider") or "A connected site"
    try:
        await repository.create_item(
            {
                "user_id": user_id,
                "assistant_id": record.get("assistant_id"),
                "source_kind": "connection",
                "account_key": record.get("account_key"),
                "external_id": f"reconnect:{record.get('account_key')}:{int(time.time())}",
                "external_thread_id": None,
                "sender": "Neural Nexus connections",
                "sender_domain": "neuralnexus.site",
                "recipients": [],
                "subject": f"{label} needs sign-in again",
                "body_text": (
                    f"The signed-in session for {label} has lapsed ({reason}). Open the "
                    "connectors menu or ask the avatar to sign in again."
                ),
                "received_at": datetime.now(UTC),
                "message_kind": "connection",
                "decision": "notify",
                "needs_owner_action": True,
                "reason": reason,
                "draft": None,
                "confidence": 1.0,
                "confidence_detail": {
                    "account_key": record.get("account_key"),
                    "reason": "session_expired",
                },
                "state": STATE_PENDING_OWNER,
            }
        )
    except Exception:
        logger.debug("Could not create a reconnect inbox item", exc_info=True)


async def keepalive_record(
    context: Any, store: Any, record: dict[str, Any], *, user_id: str
) -> dict[str, Any]:
    """Visit one site with the stored session; refresh or flag the record."""
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import mark_account_needs_reconnect

    key = str(record.get("account_key") or "")
    provider = get_provider(str(record.get("provider") or ""))
    home_url = home_url_for(record, provider)
    if not home_url:
        return {"account_key": key, "status": "skipped", "reason": "no home url"}
    async with _keepalive_semaphore:
        try:
            handle = await open_session(context, store, user_id, record, lease=True)
        except BrowserSessionExpired as expired:
            await mark_account_needs_reconnect(store, user_id, key)
            await _notify_reconnect(record, user_id, str(expired))
            return {
                "account_key": key,
                "status": "needs_reconnect",
                "reason": str(expired),
            }
        except BrowserSessionError as session_error:
            return {
                "account_key": key,
                "status": "error",
                "reason": session_error.detail,
            }
        try:
            async with handle.lock:
                await handle.page.goto(
                    home_url, wait_until="domcontentloaded", timeout=30000
                )
                await asyncio.sleep(1.0)
                final_url = handle.page.url
                html = await handle.page.content()
            # A bot wall is checked FIRST, because its interstitial carries no
            # sign of a signed-in session and would otherwise read as a login
            # page — flagging a perfectly good session as lapsed and sending the
            # owner to sign in again, which cannot help.
            if bot_wall_detected(html):
                return {
                    "account_key": key,
                    "status": "blocked",
                    "final_url": final_url,
                    "reason": bot_wall_advice(record, hostname_of(home_url) or ""),
                }
            if login_page_detected(
                final_url, html, site_hostname=hostname_of(home_url)
            ):
                await mark_account_needs_reconnect(store, user_id, key)
                await _notify_reconnect(record, user_id, "the site asked for a sign-in")
                await forget_session(key)
                return {
                    "account_key": key,
                    "status": "needs_reconnect",
                    "final_url": final_url,
                }
            await persist_session_state(context, store, user_id, record, handle)
            return {"account_key": key, "status": "refreshed", "final_url": final_url}
        except Exception as visit_error:
            logger.info("Keepalive visit failed for %s: %s", key, visit_error)
            return {"account_key": key, "status": "error", "reason": str(visit_error)}
        finally:
            release_session(handle)


async def keepalive_once(context: Any, store: Any) -> list[dict[str, Any]]:
    """Revisit every connected browser-session account once."""
    from src.anubis.utils.connected_accounts.repository import get_repository

    repository = get_repository()
    if repository is None:
        return []
    results: list[dict[str, Any]] = []
    list_by_kind = getattr(repository, "list_by_kind", None)
    records: list[dict[str, Any]] = []
    if list_by_kind is not None:
        for kind in (
            "analytics",
            "website",
            "social",
            "messaging",
            "platform",
            "developer",
        ):
            try:
                records.extend(await list_by_kind(kind, "connected"))
            except Exception:
                logger.debug(
                    "Could not list %s accounts for keepalive", kind, exc_info=True
                )
    for record in records:
        if record.get("credential_mechanism") != "browser_session":
            continue
        if not session_details(record).get("storage_state_encrypted"):
            continue
        user_id = str(record.get("user_id") or "")
        if not user_id:
            continue
        results.append(await keepalive_record(context, store, record, user_id=user_id))
    return results


async def keepalive_forever(context: Any, store: Any = None) -> None:
    """Background loop: keep every signed-in session alive on a schedule."""
    enabled = str(
        getattr(context, "browser_session_keepalive_enabled", "true") or "true"
    )
    if enabled.strip().lower() not in ("1", "true", "yes", "on"):
        return
    hours = float(getattr(context, "browser_session_keepalive_hours", None) or 12.0)
    interval = max(60.0, hours * 3600.0)
    while True:
        try:
            await asyncio.sleep(interval)
            results = await keepalive_once(context, store)
            if results:
                logger.info(
                    "Browser session keepalive: %s",
                    ", ".join(
                        f"{entry['account_key']}={entry['status']}" for entry in results
                    ),
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Browser session keepalive failed")
