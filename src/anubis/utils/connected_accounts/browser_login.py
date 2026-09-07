"""A live browser the owner signs in through, streamed into the connect card's popup.

For a site with no OAuth the card opens a window showing a real page of the
site — the site's own login page, rendered by a headless Chromium on the
API — and forwards the owner's clicks and keystrokes to that page. The owner
types their email and password on the site, exactly as they would in their
own browser. When they press "I'm signed in" the browser's storage state is
saved (encrypted) as the account's session and the window closes.

Boundaries that keep this safe:
- The view URL carries a signed, expiring token bound to the owner's user id;
  the login id is a 128-bit random value; one client streams a login at a time.
- Frames are sent to the popup and never written anywhere. Keystrokes are
  forwarded to the page and never logged.
- A login lives ``BROWSER_SESSION_LOGIN_TTL_SECONDS`` and is capped per process.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from src.anubis.utils.connected_accounts.browser_sessions import (
    BrowserSessionError,
    build_session_transport,
    hostname_of,
    login_page_detected,
    new_context,
)
from src.anubis.utils.connected_accounts.oauth_flow import LOGIN_RESULT_MESSAGE_TYPE
from src.anubis.utils.connected_accounts.oauth_state import (
    OAuthStateError,
    random_nonce,
    sign_state,
    state_secret,
    verify_state,
)

logger = logging.getLogger(__name__)

LOGIN_VIEW_PATH = "/connect_account/browser"
LOGIN_TOKEN_HEADER = "X-Login-Token"


@dataclass
class LiveLogin:
    """One sign-in window in progress."""

    login_id: str
    nonce: str
    user_id: str
    assistant_id: str
    provider_name: str
    site_url: str
    name: str
    browser_context: Any
    page: Any
    started_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    finished: bool = False
    streaming: bool = False
    screencast_task: Any | None = None


_live_logins: dict[str, LiveLogin] = {}
_logins_lock = asyncio.Lock()


def _ttl(context: Any) -> int:
    return int(getattr(context, "browser_session_login_ttl_seconds", None) or 600)


def _max_concurrent(context: Any) -> int:
    return int(getattr(context, "browser_session_max_concurrent_logins", None) or 3)


def sign_login_token(context: Any, *, login_id: str, user_id: str, nonce: str) -> str:
    """Return the signed token the popup presents on every request."""
    return sign_state(
        {"login_id": login_id, "user_id": user_id, "nonce": nonce, "mode": "browser"},
        state_secret(context),
        _ttl(context),
    )


def verify_login_token(context: Any, token: str, login_id: str) -> dict[str, Any]:
    """Verify a popup's token belongs to ``login_id``; raise ``BrowserSessionError``."""
    try:
        payload = verify_state(token, state_secret(context))
    except OAuthStateError as state_error:
        raise BrowserSessionError(401, str(state_error)) from state_error
    if payload.get("login_id") != login_id:
        raise BrowserSessionError(401, "This sign-in window does not match the login.")
    return payload


async def reap_expired_logins() -> int:
    """Close sign-in windows nobody finished in time; return how many."""
    now = time.time()
    closed = 0
    async with _logins_lock:
        for login_id in list(_live_logins):
            login = _live_logins[login_id]
            if login.expires_at and login.expires_at < now:
                await _discard(login_id)
                closed += 1
    return closed


async def _discard(login_id: str) -> None:
    login = _live_logins.pop(login_id, None)
    if login is None:
        return
    if login.screencast_task is not None:
        login.screencast_task.cancel()
    try:
        await login.browser_context.close()
    except Exception:
        logger.debug("Could not close login context %s", login_id, exc_info=True)


def get_live_login(login_id: str) -> LiveLogin | None:
    """Return a login in progress, or ``None``."""
    return _live_logins.get(login_id)


async def start_login(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    provider: Any,
    site_url: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Open a browser at the site's sign-in page; return ``{login_id, view_url, nonce}``."""
    await reap_expired_logins()
    start_url = str(site_url or getattr(provider, "login_url", None) or "").strip()
    if not start_url.startswith("http"):
        raise BrowserSessionError(400, "A site address to sign in to is required.")
    async with _logins_lock:
        active = [login for login in _live_logins.values() if not login.finished]
        if len(active) >= _max_concurrent(context):
            raise BrowserSessionError(
                409,
                "Too many sign-in windows are open right now; finish or close one first.",
            )
        browser_context = await new_context(context)
        page = await browser_context.new_page()
        login_id = secrets.token_urlsafe(16)
        nonce = random_nonce()
        login = LiveLogin(
            login_id=login_id,
            nonce=nonce,
            user_id=user_id,
            assistant_id=assistant_id,
            provider_name=provider.name,
            site_url=start_url,
            name=str(name or provider.display_name),
            browser_context=browser_context,
            page=page,
            expires_at=time.time() + _ttl(context),
        )
        _live_logins[login_id] = login
    try:
        await page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
    except Exception as navigation_error:
        await _discard(login_id)
        raise BrowserSessionError(
            400, f"{start_url} could not be opened: {navigation_error}"
        ) from navigation_error
    token = sign_login_token(context, login_id=login_id, user_id=user_id, nonce=nonce)
    return {
        "login_id": login_id,
        "nonce": nonce,
        "view_url": f"{LOGIN_VIEW_PATH}/{login_id}?t={token}",
        "expires_in": _ttl(context),
        "provider": provider.name,
        "site_url": start_url,
    }


# ---------------------------------------------------------------------------
# Streaming frames and forwarding input
# ---------------------------------------------------------------------------


async def _send_frame(websocket: Any, data_base64: str, width: int, height: int) -> None:
    await websocket.send_text(
        json.dumps({"type": "frame", "data": data_base64, "width": width, "height": height})
    )


async def stream_login(websocket: Any, login: LiveLogin, context: Any) -> None:
    """Stream the page to the popup and forward the popup's input, until closed.

    Frames come from Chromium's own screencast when available (only changed
    frames, on Chromium's schedule); otherwise a screenshot loop at
    ``BROWSER_SESSION_FRAME_INTERVAL_MS``. Input messages: ``mouse``
    (``action`` move|down|up, ``x``, ``y``, ``button``), ``wheel`` (``deltaX``,
    ``deltaY``), ``key`` (``action`` down|up, ``key``), ``text`` (``text`` to
    insert), ``resize`` (``width``, ``height``), ``navigate`` (``url`` on the same
    site only).
    """
    page = login.page
    login.streaming = True
    interval_ms = int(getattr(context, "browser_session_frame_interval_ms", None) or 250)
    viewport = dict(page.viewport_size or {"width": 1280, "height": 800})

    async def _screenshot_loop() -> None:
        while True:
            try:
                image = await page.screenshot(type="jpeg", quality=55)
                await _send_frame(
                    websocket,
                    base64.b64encode(image).decode("ascii"),
                    viewport["width"],
                    viewport["height"],
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Screenshot frame failed", exc_info=True)
            await asyncio.sleep(interval_ms / 1000.0)

    async def _screencast() -> bool:
        try:
            session = await page.context.new_cdp_session(page)
        except Exception:
            return False
        loop = asyncio.get_running_loop()

        def _on_frame(event: dict[str, Any]) -> None:
            data = event.get("data")
            session_id = event.get("sessionId")
            metadata = event.get("metadata") or {}
            if data:
                loop.create_task(
                    _send_frame(
                        websocket,
                        data,
                        int(metadata.get("deviceWidth") or viewport["width"]),
                        int(metadata.get("deviceHeight") or viewport["height"]),
                    )
                )
            if session_id is not None:
                loop.create_task(session.send("Page.screencastFrameAck", {"sessionId": session_id}))

        try:
            session.on("Page.screencastFrame", _on_frame)
            await session.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 60,
                    "maxWidth": viewport["width"],
                    "maxHeight": viewport["height"],
                    "everyNthFrame": 2,
                },
            )
        except Exception:
            return False
        return True

    if not await _screencast():
        login.screencast_task = asyncio.create_task(_screenshot_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except Exception:
                continue
            if not isinstance(message, dict):
                continue
            await dispatch_input(login, message, viewport)
    except asyncio.CancelledError:
        raise
    except Exception:
        # The popup closed or the socket dropped; the login stays open until
        # finished, cancelled, or expired.
        logger.debug("Login stream ended for %s", login.login_id, exc_info=True)
    finally:
        login.streaming = False
        if login.screencast_task is not None:
            login.screencast_task.cancel()
            login.screencast_task = None


_MOUSE_BUTTONS = {"left", "right", "middle"}


async def dispatch_input(login: LiveLogin, message: dict[str, Any], viewport: dict[str, int]) -> None:
    """Forward one popup input message to the page (never logged)."""
    page = login.page
    kind = str(message.get("type") or "")
    try:
        if kind == "mouse":
            action = str(message.get("action") or "move")
            x = float(message.get("x") or 0)
            y = float(message.get("y") or 0)
            button = str(message.get("button") or "left")
            if button not in _MOUSE_BUTTONS:
                button = "left"
            if action == "move":
                await page.mouse.move(x, y)
            elif action == "down":
                await page.mouse.move(x, y)
                await page.mouse.down(button=button)
            elif action == "up":
                await page.mouse.up(button=button)
            elif action == "click":
                await page.mouse.click(x, y, button=button)
        elif kind == "wheel":
            await page.mouse.wheel(float(message.get("deltaX") or 0), float(message.get("deltaY") or 0))
        elif kind == "key":
            action = str(message.get("action") or "down")
            key = str(message.get("key") or "")
            if not key:
                return
            if action == "down":
                await page.keyboard.down(key)
            elif action == "up":
                await page.keyboard.up(key)
            elif action == "press":
                await page.keyboard.press(key)
        elif kind == "text":
            text = str(message.get("text") or "")
            if text:
                await page.keyboard.insert_text(text)
        elif kind == "resize":
            width = max(320, min(1920, int(message.get("width") or viewport["width"])))
            height = max(240, min(1200, int(message.get("height") or viewport["height"])))
            viewport["width"], viewport["height"] = width, height
            await page.set_viewport_size({"width": width, "height": height})
        elif kind == "navigate":
            url = str(message.get("url") or "")
            if url.startswith("http") and hostname_of(url) == hostname_of(login.site_url):
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception:
        # A stale element or a closed page must not end the stream.
        logger.debug("Input dispatch failed for login %s", login.login_id, exc_info=True)


# ---------------------------------------------------------------------------
# Finishing
# ---------------------------------------------------------------------------


async def finish_login(
    context: Any,
    *,
    login_id: str,
    user_id: str,
    existing_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save the signed-in session as an account record and close the window."""
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
    )

    login = _live_logins.get(login_id)
    if login is None or login.user_id != user_id:
        raise BrowserSessionError(404, "No sign-in window with that id is open.")
    if login.finished:
        raise BrowserSessionError(409, "This sign-in was already finished.")
    login.finished = True
    provider = get_provider(login.provider_name)
    try:
        final_url = login.page.url
        try:
            html = await login.page.content()
        except Exception:
            html = ""
        storage_state = await login.browser_context.storage_state()
    except Exception as capture_error:
        await _discard(login_id)
        raise BrowserSessionError(
            500, f"The signed-in session could not be captured: {capture_error}"
        ) from capture_error
    await _discard(login_id)

    site_hostname = hostname_of(login.site_url)
    heuristic_signed_in = not login_page_detected(final_url, html, site_hostname=site_hostname)
    home_url = getattr(provider, "home_url", None)
    if not home_url or not str(home_url).startswith("http"):
        home_url = final_url if same_host(final_url, site_hostname) else login.site_url
    address = site_hostname if provider.name not in ("custom_site",) else f"{site_hostname}#{login.login_id[:8]}"
    key = account_key(provider.name, address)
    label = deduplicate_label(login.name or site_hostname, existing_records, key)
    extra: dict[str, Any] = {}
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
            final_url=final_url,
            site_url=login.site_url,
            heuristic_signed_in=heuristic_signed_in,
            recipe_key=getattr(provider, "recipe_key", None),
            extra=extra,
        ),
    )
    record["user_id"] = user_id
    # The row may sign in through OAuth when an app exists; this record was
    # made by a live sign-in, so the tools must treat the record as a session.
    record["credential_mechanism"] = "browser_session"
    return {
        "record": record,
        "nonce": login.nonce,
        "heuristic_signed_in": heuristic_signed_in,
        "final_url": final_url,
    }


def same_host(url: str, hostname: str) -> bool:
    """Whether a URL is on the given host or a subdomain of the host."""
    from src.anubis.utils.connected_accounts.browser_sessions import same_site

    return same_site(url, hostname)


async def cancel_login(login_id: str, user_id: str) -> bool:
    """Close a sign-in window without saving anything."""
    login = _live_logins.get(login_id)
    if login is None or login.user_id != user_id:
        return False
    await _discard(login_id)
    return True


# ---------------------------------------------------------------------------
# The popup page
# ---------------------------------------------------------------------------


def render_login_page_html(
    *,
    login_id: str,
    token: str,
    nonce: str,
    provider_name: str,
    display_name: str,
    site_url: str,
    allowed_origins: list[str],
    finish_path: str,
    cancel_path: str,
    stream_path: str,
) -> str:
    """Render the popup: a canvas of the live page, input forwarding, and two buttons."""
    result_template = {
        "type": LOGIN_RESULT_MESSAGE_TYPE,
        "ok": False,
        "nonce": nonce,
        "provider": provider_name,
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sign in to {display_name}</title>
<style>
html,body{{margin:0;height:100%;background:#0b0b0d;color:#e8e8ea;font-family:system-ui,sans-serif;overflow:hidden}}
header{{display:flex;align-items:center;gap:.75rem;padding:.5rem .75rem;background:#141418;border-bottom:1px solid #26262c}}
header .site{{flex:1;color:#a3a3a8;font-size:.85rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
button{{padding:.45rem .9rem;border-radius:999px;border:1px solid #444;background:#1a1a1f;color:#fff;cursor:pointer;font-size:.85rem}}
button.primary{{background:#f5b301;color:#111;border-color:#f5b301;font-weight:600}}
#stage{{position:absolute;top:44px;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:#000}}
canvas{{max-width:100%;max-height:100%;cursor:default;outline:none}}
#status{{position:absolute;left:.75rem;bottom:.5rem;color:#a3a3a8;font-size:.8rem}}
</style></head>
<body>
<header>
  <span class="site" id="site">{site_url}</span>
  <button id="cancel">Cancel</button>
  <button id="done" class="primary">I'm signed in</button>
</header>
<div id="stage"><canvas id="view" tabindex="0" width="1280" height="800"></canvas></div>
<div id="status">Connecting to the sign-in page…</div>
<script>
(function () {{
  var origins = {json.dumps(list(allowed_origins))};
  var base = {json.dumps(result_template)};
  var token = {json.dumps(token)};
  var canvas = document.getElementById('view');
  var drawing = canvas.getContext('2d');
  var statusLine = document.getElementById('status');
  var frameWidth = 1280, frameHeight = 800;
  var socketProtocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
  var socket = new WebSocket(socketProtocol + location.host + {json.dumps(stream_path)} + '?t=' + encodeURIComponent(token));
  var finished = false;
  function post(result) {{
    var payload = Object.assign({{}}, base, result);
    try {{
      if (window.opener && !window.opener.closed) {{
        for (var i = 0; i < origins.length; i += 1) {{ try {{ window.opener.postMessage(payload, origins[i]); }} catch (e) {{}} }}
      }}
    }} catch (e) {{}}
  }}
  function send(message) {{ if (socket.readyState === 1) {{ socket.send(JSON.stringify(message)); }} }}
  socket.onopen = function () {{ statusLine.textContent = 'Sign in on the page, then press "I\\'m signed in".'; }};
  socket.onmessage = function (event) {{
    var message; try {{ message = JSON.parse(event.data); }} catch (e) {{ return; }}
    if (message.type === 'frame') {{
      var image = new Image();
      image.onload = function () {{
        if (canvas.width !== message.width || canvas.height !== message.height) {{ canvas.width = message.width; canvas.height = message.height; }}
        frameWidth = message.width; frameHeight = message.height;
        drawing.drawImage(image, 0, 0);
      }};
      image.src = 'data:image/jpeg;base64,' + message.data;
    }}
  }};
  socket.onclose = function () {{ if (!finished) {{ statusLine.textContent = 'The connection to the sign-in page closed. Reopen the sign-in from the card.'; }} }};
  function pageCoordinates(event) {{
    var rectangle = canvas.getBoundingClientRect();
    return {{ x: (event.clientX - rectangle.left) * (frameWidth / rectangle.width), y: (event.clientY - rectangle.top) * (frameHeight / rectangle.height) }};
  }}
  var buttonNames = ['left', 'middle', 'right'];
  canvas.addEventListener('mousemove', function (event) {{ var point = pageCoordinates(event); send({{ type: 'mouse', action: 'move', x: point.x, y: point.y }}); }});
  canvas.addEventListener('mousedown', function (event) {{ canvas.focus(); var point = pageCoordinates(event); send({{ type: 'mouse', action: 'down', x: point.x, y: point.y, button: buttonNames[event.button] || 'left' }}); event.preventDefault(); }});
  canvas.addEventListener('mouseup', function (event) {{ var point = pageCoordinates(event); send({{ type: 'mouse', action: 'up', x: point.x, y: point.y, button: buttonNames[event.button] || 'left' }}); event.preventDefault(); }});
  canvas.addEventListener('contextmenu', function (event) {{ event.preventDefault(); }});
  canvas.addEventListener('wheel', function (event) {{ send({{ type: 'wheel', deltaX: event.deltaX, deltaY: event.deltaY }}); event.preventDefault(); }}, {{ passive: false }});
  canvas.addEventListener('keydown', function (event) {{
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {{ send({{ type: 'text', text: event.key }}); }}
    else {{ send({{ type: 'key', action: 'press', key: event.key }}); }}
    event.preventDefault();
  }});
  canvas.addEventListener('paste', function (event) {{ var text = (event.clipboardData || window.clipboardData).getData('text'); if (text) {{ send({{ type: 'text', text: text }}); }} event.preventDefault(); }});
  document.getElementById('done').addEventListener('click', function () {{
    finished = true;
    statusLine.textContent = 'Saving your signed-in session…';
    fetch({json.dumps(finish_path)}, {{ method: 'POST', headers: {{ 'X-Login-Token': token, 'Content-Type': 'application/json' }}, body: '{{}}' }})
      .then(function (response) {{ return response.json(); }})
      .then(function (body) {{
        post(body);
        statusLine.textContent = body.ok ? ((body.display_label || 'The site') + ' is connected. You can close this window.') : (body.error || 'The session was not saved.');
        if (body.ok) {{ setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 700); }} else {{ finished = false; }}
      }})
      .catch(function () {{ finished = false; statusLine.textContent = 'The session could not be saved. Try again.'; }});
  }});
  document.getElementById('cancel').addEventListener('click', function () {{
    finished = true;
    fetch({json.dumps(cancel_path)}, {{ method: 'POST', headers: {{ 'X-Login-Token': token }} }}).catch(function () {{}});
    post({{ ok: false, error: 'The sign-in was cancelled.' }});
    setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 300);
  }});
  window.addEventListener('beforeunload', function () {{ if (!finished) {{ navigator.sendBeacon && navigator.sendBeacon({json.dumps(cancel_path)} + '?t=' + encodeURIComponent(token)); }} }});
  canvas.focus();
}})();
</script></body></html>"""
