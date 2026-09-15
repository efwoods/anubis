"""Long-lived hosted computer the platform administrator's personal avatar owns.

Unlike ``browser_login``, finishing a handoff must not close the Chromium
context. The owner takes over that computer, signs in (including two-factor
on the vendor's own page), and presses I'm done. The same browser stays open
so the avatar can walk the next dashboard.
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
from src.anubis.utils.connected_accounts.oauth_state import (
    OAuthStateError,
    random_nonce,
    sign_state,
    state_secret,
    verify_state,
)

logger = logging.getLogger(__name__)

COMPUTER_HANDOFF_INTERRUPT_KIND = "computer_handoff"
COMPUTER_TOKEN_HEADER = "X-Login-Token"


@dataclass(frozen=True)
class DashboardStep:
    """One unsigned dashboard the sequential queue visits on the same computer."""

    provider: str
    url: str
    recipe: str
    task: str


VENDOR_DASHBOARD_QUEUE: tuple[DashboardStep, ...] = (
    DashboardStep(
        provider="cursor",
        url="https://cursor.com/dashboard/spending",
        recipe="spending",
        task="Sign in to Cursor (Google/GitHub/email + any 2FA), then hand back",
    ),
    DashboardStep(
        provider="cursor",
        url="https://cursor.com/dashboard/usage",
        recipe="usage",
        task="Open Cursor usage and hand back if a sign-in wall appears",
    ),
    DashboardStep(
        provider="claude_app",
        url="https://claude.ai/settings/usage",
        recipe="usage_page",
        task="Sign in to Claude.ai (Google SSO may already cover this), then hand back",
    ),
    DashboardStep(
        provider="openai",
        url=(
            "https://platform.openai.com/settings/organization/usage"
            "?usage_section=spend-categories"
        ),
        recipe="usage_page",
        task="Sign in to the OpenAI usage page (spend categories), then hand back",
    ),
    DashboardStep(
        provider="elevenlabs",
        url="https://elevenlabs.io/app/developers/analytics/usage",
        recipe="usage_page",
        task="Sign in to ElevenLabs analytics, then hand back",
    ),
    DashboardStep(
        provider="xai",
        url=(
            "https://console.x.ai/team/1db9c97a-09ce-4be7-bca1-f0fb9e59ec18"
            "/settings/billing"
        ),
        recipe="usage_page",
        task="Sign in to xAI billing, then hand back",
    ),
)


@dataclass
class AgentComputerSession:
    """One long-lived Chromium context the avatar keeps walking dashboards in."""

    session_id: str
    user_id: str
    assistant_id: str
    browser_context: Any
    page: Any
    nonce: str
    started_at: float = field(default_factory=time.time)
    last_used_monotonic: float = field(default_factory=time.monotonic)
    streaming: bool = False
    takeover_active: bool = False
    context_closed: bool = False
    queue: list[DashboardStep] = field(default_factory=list)
    queue_index: int = 0
    current_task: str = ""
    current_provider: str = ""
    current_url: str = ""
    preview_jpeg_b64: str | None = None
    screencast_task: Any | None = None

    def touch(self) -> None:
        """Mark the computer as just used."""
        self.last_used_monotonic = time.monotonic()

    @property
    def context_is_open(self) -> bool:
        """Whether the Chromium context is still alive."""
        return not self.context_closed and self.browser_context is not None


_computers: dict[str, AgentComputerSession] = {}
_computers_by_user: dict[str, str] = {}
_computers_lock = asyncio.Lock()


def agent_computer_is_enabled(context: Any) -> bool:
    """Whether the hosted computer may be opened (``AGENT_COMPUTER_ENABLED``)."""
    raw = str(getattr(context, "agent_computer_enabled", "true") or "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def sign_computer_token(context: Any, *, session_id: str, user_id: str, nonce: str) -> str:
    """Return the signed token the takeover view presents on every request."""
    ttl = int(getattr(context, "browser_session_login_ttl_seconds", None) or 3600)
    return sign_state(
        {
            "login_id": session_id,
            "user_id": user_id,
            "nonce": nonce,
            "mode": "computer",
        },
        state_secret(context),
        ttl,
    )


def verify_computer_token(context: Any, token: str, session_id: str) -> dict[str, Any]:
    """Verify a takeover token belongs to ``session_id``."""
    try:
        payload = verify_state(token, state_secret(context))
    except OAuthStateError as state_error:
        raise BrowserSessionError(401, str(state_error)) from state_error
    if payload.get("login_id") != session_id or payload.get("mode") != "computer":
        raise BrowserSessionError(401, "This computer view does not match the session.")
    return payload


def reset_computers_for_tests() -> None:
    """Drop every in-memory computer. Tests only."""
    _computers.clear()
    _computers_by_user.clear()


def get_computer(session_id: str) -> AgentComputerSession | None:
    """Return a live computer session, or ``None``."""
    return _computers.get(session_id)


def computer_for_user(user_id: str) -> AgentComputerSession | None:
    """Return the owner's current computer when the context is still open."""
    session_id = _computers_by_user.get(user_id)
    if not session_id:
        return None
    session = _computers.get(session_id)
    if session is None or not session.context_is_open:
        return None
    return session


def remaining_providers(session: AgentComputerSession) -> list[str]:
    """Provider names still waiting on the sequential queue, in order."""
    names: list[str] = []
    for step in session.queue[session.queue_index :]:
        if step.provider not in names:
            names.append(step.provider)
    return names


def current_step(session: AgentComputerSession) -> DashboardStep | None:
    """The queue step the computer is on, or ``None`` when the queue is finished."""
    if session.queue_index < 0 or session.queue_index >= len(session.queue):
        return None
    return session.queue[session.queue_index]


def advance_queue(session: AgentComputerSession) -> DashboardStep | None:
    """Move to the next dashboard; return that step or ``None`` when finished."""
    session.queue_index += 1
    session.touch()
    return current_step(session)


def build_handoff_card(
    session: AgentComputerSession,
    *,
    context: Any,
    message: str | None = None,
) -> dict[str, Any]:
    """Describe the Action-needed Computer card (no credentials)."""
    step = current_step(session)
    token = sign_computer_token(
        context,
        session_id=session.session_id,
        user_id=session.user_id,
        nonce=session.nonce,
    )
    return {
        "kind": COMPUTER_HANDOFF_INTERRUPT_KIND,
        "task": session.current_task or (step.task if step else "Hand the computer back"),
        "site_url": session.current_url or (step.url if step else ""),
        "session_id": session.session_id,
        "login_id": session.session_id,
        "preview_frame": session.preview_jpeg_b64,
        "providers_waiting": remaining_providers(session),
        "provider": session.current_provider or (step.provider if step else ""),
        "stream_path": f"/computer/{session.session_id}/stream",
        "preview_path": f"/computer/{session.session_id}/preview",
        "view_token": token,
        "message": message,
        "status": "action_needed",
        "actions": ["takeover", "done", "skip"],
    }


async def start_computer(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    start_url: str | None = None,
    provider: str | None = None,
    task: str | None = None,
    queue: list[DashboardStep] | None = None,
    browser_context: Any | None = None,
    page: Any | None = None,
) -> AgentComputerSession:
    """Create or resume the owner's long-lived computer. Never closes an existing context."""
    async with _computers_lock:
        existing = computer_for_user(user_id)
        if existing is not None:
            session = existing
        else:
            opened_context = browser_context or await new_context(context)
            opened_page = page
            if opened_page is None:
                pages = list(getattr(opened_context, "pages", None) or [])
                opened_page = pages[0] if pages else await opened_context.new_page()
            session_id = secrets.token_hex(16)
            session = AgentComputerSession(
                session_id=session_id,
                user_id=user_id,
                assistant_id=assistant_id,
                browser_context=opened_context,
                page=opened_page,
                nonce=random_nonce(),
                queue=list(queue or VENDOR_DASHBOARD_QUEUE),
            )
            _computers[session_id] = session
            _computers_by_user[user_id] = session_id
        if queue is not None:
            session.queue = list(queue)
            session.queue_index = 0
        session.assistant_id = assistant_id
        session.touch()
    if start_url:
        await navigate_computer(session, start_url, provider=provider, task=task)
    elif current_step(session) is not None:
        step = current_step(session)
        await navigate_computer(
            session, step.url, provider=step.provider, task=step.task
        )
    return session


async def navigate_computer(
    session: AgentComputerSession,
    url: str,
    *,
    provider: str | None = None,
    task: str | None = None,
) -> None:
    """Open a URL on the same context and refresh the preview frame."""
    if not session.context_is_open:
        raise BrowserSessionError(409, "The agent computer is no longer open.")
    session.current_url = url
    if provider:
        session.current_provider = provider
    if task:
        session.current_task = task
    try:
        await session.page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        logger.debug("Computer navigation to %s failed", url, exc_info=True)
    await capture_preview(session)
    session.touch()


async def capture_preview(session: AgentComputerSession) -> str | None:
    """Take a JPEG screenshot and store it on the session."""
    if not session.context_is_open:
        return session.preview_jpeg_b64
    try:
        image = await session.page.screenshot(type="jpeg", quality=50)
        session.preview_jpeg_b64 = base64.b64encode(image).decode("ascii")
    except Exception:
        logger.debug("Computer preview failed for %s", session.session_id, exc_info=True)
    session.touch()
    return session.preview_jpeg_b64


async def page_looks_like_login(session: AgentComputerSession) -> bool:
    """Whether the current page is a vendor login wall."""
    try:
        html = await session.page.content()
        url = str(getattr(session.page, "url", "") or session.current_url)
    except Exception:
        return False
    return login_page_detected(url, html, site_hostname=hostname_of(session.current_url))


async def persist_computer_session(
    context: Any,
    store: Any,
    session: AgentComputerSession,
    *,
    existing_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save cookies from the live context. Does not close the context."""
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
        save_connected_account,
    )

    if not session.context_is_open:
        raise BrowserSessionError(409, "The agent computer is no longer open.")
    storage_state = await session.browser_context.storage_state()
    provider = get_provider(session.current_provider)
    if provider is None:
        return {
            "status": "ok",
            "cookie_count": len(storage_state.get("cookies") or []),
            "context_closed": False,
        }
    site_hostname = hostname_of(session.current_url or provider.login_url or "")
    existing = next(
        (
            record
            for record in (existing_records or [])
            if str(record.get("provider") or "") == provider.name
        ),
        None,
    )
    if existing is not None:
        record = dict(existing)
        from src.anubis.utils.secret_store import encrypt_secret

        transport = dict(record.get("transport") or {})
        session_block = dict(transport.get("browser_session") or {})
        session_block["storage_state_encrypted"] = encrypt_secret(
            json.dumps(storage_state), context
        )
        session_block["cookie_count"] = len(storage_state.get("cookies") or [])
        transport["browser_session"] = session_block
        record["transport"] = transport
    else:
        address = f"{site_hostname}#{session.session_id[:8]}"
        record = build_account_record(
            provider=provider,
            account_address=address,
            display_label=deduplicate_label(
                provider.display_name, existing_records or [], account_key(provider.name, address)
            ),
            encrypted_secret=None,
            assistant_id=session.assistant_id,
            transport=build_session_transport(
                storage_state=storage_state,
                context=context,
                home_url=str(getattr(provider, "home_url", None) or session.current_url),
                final_url=str(getattr(session.page, "url", "") or session.current_url),
                site_url=session.current_url,
                heuristic_signed_in=not await page_looks_like_login(session),
                recipe_key=getattr(provider, "recipe_key", None),
            ),
        )
        record["credential_mechanism"] = "browser_session"
    record["user_id"] = session.user_id
    if store is not None:
        await save_connected_account(store, session.user_id, record)
    return {
        "status": "ok",
        "record": record,
        "cookie_count": len(storage_state.get("cookies") or []),
        "context_closed": False,
    }


async def run_recipe_on_computer(
    session: AgentComputerSession,
    *,
    pool: Any,
    recipe_name: str | None = None,
) -> dict[str, Any]:
    """Run the current provider's recipe on the same open page."""
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.recipes import (
        RECIPE_KIND_JSON,
        recipes_for,
        render_url,
    )

    provider = get_provider(session.current_provider)
    recipe_key = getattr(provider, "recipe_key", None) if provider else session.current_provider
    available = recipes_for(recipe_key)
    step = current_step(session)
    chosen_name = (recipe_name or (step.recipe if step else "") or "").strip().lower()
    chosen = available.get(chosen_name) if chosen_name else None
    if chosen is None and available:
        chosen = next(iter(available.values()))
    if chosen is None:
        return {"status": "no_recipes", "provider": session.current_provider}
    url = render_url(chosen, "30d")
    try:
        if chosen.kind == RECIPE_KIND_JSON:
            response = await session.browser_context.request.fetch(
                url, method=chosen.method, headers={"Accept": "application/json", **chosen.headers}
            )
            text = await response.text()
            try:
                document: Any = json.loads(text)
            except Exception:
                document = text
        else:
            await session.page.goto(url, wait_until="domcontentloaded", timeout=45000)
            document = await session.page.content()
            if login_page_detected(
                str(getattr(session.page, "url", "") or url),
                str(document),
                site_hostname=hostname_of(url),
            ):
                return {"status": "login_required", "url": url, "provider": session.current_provider}
    except Exception as recipe_error:
        logger.debug("Computer recipe failed", exc_info=True)
        return {"status": "error", "error": str(recipe_error)[:300]}
    rows = chosen.parser(document, {})
    stored = 0
    if pool is not None and rows:
        from src.anubis.utils.analytics.vendor_usage import record_rows

        stored = await record_rows(
            pool, session.user_id, session.current_provider, rows, source="agent_computer"
        )
    await capture_preview(session)
    return {
        "status": "ok",
        "provider": session.current_provider,
        "recipe": chosen.name,
        "rows": rows[:50],
        "stored": stored,
        "context_closed": False,
    }


async def finish_handoff(
    context: Any,
    store: Any,
    session: AgentComputerSession,
    *,
    existing_records: list[dict[str, Any]] | None = None,
    pool: Any = None,
    run_recipe: bool = True,
) -> dict[str, Any]:
    """Persist cookies and optionally run the recipe. Never closes the context."""
    if session.context_closed:
        raise BrowserSessionError(409, "The agent computer was already closed.")
    session.takeover_active = False
    persisted = await persist_computer_session(
        context, store, session, existing_records=existing_records
    )
    if await page_looks_like_login(session):
        return {
            "status": "login_required",
            "context_closed": False,
            "cookie_count": persisted.get("cookie_count"),
            "provider": session.current_provider,
            "message": (
                "The page still looks like a sign-in wall. The owner needs to "
                "finish signing in, then press I'm done again."
            ),
        }
    recipe_result: dict[str, Any] = {"status": "skipped"}
    if run_recipe:
        recipe_result = await run_recipe_on_computer(session, pool=pool)
        if recipe_result.get("status") == "login_required":
            return {
                "status": "login_required",
                "context_closed": False,
                "cookie_count": persisted.get("cookie_count"),
                "provider": session.current_provider,
                "recipe": recipe_result,
            }
    return {
        "status": "done",
        "context_closed": False,
        "cookie_count": persisted.get("cookie_count"),
        "provider": session.current_provider,
        "recipe": recipe_result,
        "record": persisted.get("record"),
    }


async def skip_handoff(session: AgentComputerSession) -> dict[str, Any]:
    """Advance the queue without persisting and without closing the context."""
    if session.context_closed:
        raise BrowserSessionError(409, "The agent computer was already closed.")
    session.takeover_active = False
    next_step = advance_queue(session)
    result: dict[str, Any] = {
        "status": "skipped",
        "context_closed": False,
        "provider": session.current_provider,
        "queue_finished": next_step is None,
    }
    if next_step is not None:
        await navigate_computer(
            session, next_step.url, provider=next_step.provider, task=next_step.task
        )
        result["next_provider"] = next_step.provider
        result["next_url"] = next_step.url
        result["login_required"] = await page_looks_like_login(session)
    return result


def mark_takeover(session: AgentComputerSession, active: bool) -> None:
    """Record whether the owner is currently in the fullscreen takeover."""
    session.takeover_active = active
    session.touch()


async def close_computer(session: AgentComputerSession) -> None:
    """Explicitly destroy the computer. Finish and skip must never call this."""
    session.context_closed = True
    if session.screencast_task is not None:
        session.screencast_task.cancel()
        session.screencast_task = None
    try:
        await session.browser_context.close()
    except Exception:
        logger.debug("Could not close computer %s", session.session_id, exc_info=True)
    _computers.pop(session.session_id, None)
    if _computers_by_user.get(session.user_id) == session.session_id:
        _computers_by_user.pop(session.user_id, None)


async def stream_computer(websocket: Any, session: AgentComputerSession, context: Any) -> None:
    """Stream frames of the computer and forward input, without closing the context."""
    from src.anubis.utils.connected_accounts.browser_login import dispatch_input

    page = session.page
    session.streaming = True
    interval_ms = int(getattr(context, "browser_session_frame_interval_ms", None) or 250)
    viewport = dict(getattr(page, "viewport_size", None) or {"width": 1280, "height": 800})

    async def _send_frame(data_base64: str, width: int, height: int) -> None:
        await websocket.send_text(
            json.dumps({"type": "frame", "data": data_base64, "width": width, "height": height})
        )

    async def _screenshot_loop() -> None:
        while True:
            try:
                image = await page.screenshot(type="jpeg", quality=55)
                encoded = base64.b64encode(image).decode("ascii")
                session.preview_jpeg_b64 = encoded
                await _send_frame(encoded, viewport["width"], viewport["height"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Computer screenshot failed", exc_info=True)
            await asyncio.sleep(interval_ms / 1000.0)

    class _LoginShim:
        page = session.page
        site_url = session.current_url
        login_id = session.session_id

    if session.screencast_task is not None:
        session.screencast_task.cancel()
    session.screencast_task = asyncio.create_task(_screenshot_loop())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except Exception:
                continue
            if not isinstance(message, dict):
                continue
            await dispatch_input(_LoginShim(), message, viewport)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Computer stream ended for %s", session.session_id, exc_info=True)
    finally:
        session.streaming = False
        if session.screencast_task is not None:
            session.screencast_task.cancel()
            session.screencast_task = None
