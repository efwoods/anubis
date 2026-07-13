"""Playwright browser tools for the avatar deep agent.

Wraps the LangChain community ``PlayWrightBrowserToolkit`` so the avatar
can browse the live web during a turn: navigate to a URL, click elements,
read the current page, extract visible text, extract hyperlinks, and query
elements by CSS selector. The capability is gated by the
``BROWSER_TOOLS_ENABLED`` environment variable (see ``GlobalContext``).

Why the LangChain helper ``create_async_playwright_browser`` is NOT used:
that helper launches the browser through
``asyncio.get_event_loop().run_until_complete(...)``, which raises
``RuntimeError: this event loop is already running`` when called from
inside the LangGraph server's own event loop (every ``think`` turn runs
there). The browser is therefore launched below with plain ``await``
calls against ``playwright.async_api``.

Conversation isolation — why one Chromium PROCESS per conversation:
every toolkit tool resolves the page to act on via
``aget_current_page(self.async_browser)``, which hard-codes
``browser.contexts[0]`` and "last page wins". Two conversations sharing
one ``Browser`` object would therefore share one page stack, cookies, and
navigation history. A lighter ``browser.new_context()`` per conversation
cannot be used because (a) the tools' pydantic fields validate
``async_browser`` as a real ``playwright.async_api.Browser`` instance and
(b) ``aget_current_page`` ignores every context except index zero — and
the Python Playwright API has no ``launch_server`` for multiplexing
isolated client connections onto one shared process. So each conversation
gets a dedicated ``chromium.launch()`` (all launches share the one
Playwright driver process). Cost model: one idle headless Chromium is
roughly 100-200 MiB resident, so the registry is bounded by
``BROWSER_MAX_CONCURRENT_CONVERSATIONS`` (least-recently-used eviction)
and browsers idle longer than
``BROWSER_CONVERSATION_IDLE_TIMEOUT_SECONDS`` are closed on the next
call. Browsing state is therefore ephemeral: a conversation that returns
after eviction starts from a fresh browser. Eviction only ever touches
browsers between turns: every turn holds a lease on the browser
(``_ConversationBrowser.active_turn_count``) that ``think`` releases in a
``finally``, and leased browsers are exempt from both eviction paths — the
cap is exceeded temporarily instead when every browser is mid-turn.

Chromium binary resolution: when ``BROWSER_CHROMIUM_EXECUTABLE_PATH`` is
set (the production wolfi image sets the variable to ``/usr/bin/chromium``
installed via ``apk add chromium``), that system binary is launched.
When the variable is empty, Playwright falls back to the Playwright-managed
Chromium download (requires ``playwright install chromium`` on the host —
the local-development path).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.anubis.utils.context import GlobalContext

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class _ConversationBrowser:
    """One conversation's dedicated Chromium process plus recency bookkeeping."""

    browser: Any
    last_used_monotonic_seconds: float

    active_turn_count: int = 0
    """Number of ``think`` turns currently holding this browser (the lease).

    Incremented when ``get_browser_toolkit_tools`` hands the browser's tools
    to a turn; decremented by ``release_conversation_browser`` in the turn's
    ``finally``. Eviction never closes a browser whose count is above zero,
    so a conversation mid-browse can never lose the browser under the
    agent's feet — the concurrent-conversation cap is a soft cap that may be
    temporarily exceeded while every browser is leased.
    """


# Process-wide shared Playwright driver (the node.js sidecar every launch
# goes through). ``None`` until the first enabled turn starts the driver.
_playwright_driver: Any | None = None

# conversation key (outer workflow thread id) -> that conversation's browser.
_conversation_browsers: dict[str, _ConversationBrowser] = {}

# Set after a failed driver start / browser launch so a misconfigured
# deployment (for example a missing Chromium binary) logs one warning and
# degrades to a normal browserless turn, instead of re-attempting a doomed
# launch on every message.
_browser_launch_failed: bool = False

# Serializes registry mutation and launch attempts so concurrent turns cannot
# race two Chromium processes into existence for one conversation.
_browser_registry_lock: asyncio.Lock = asyncio.Lock()

# Conversation key used when the outer workflow has no thread id (should not
# happen in served traffic; keeps ad-hoc invocations working).
_FALLBACK_CONVERSATION_KEY = "shared"


def browser_tools_enabled(context: GlobalContext) -> bool:
    """Report whether the BROWSER_TOOLS_ENABLED environment gate is set to TRUE."""
    return (context.browser_tools_enabled or "").strip().upper() == "TRUE"


async def _launch_conversation_browser(context: GlobalContext) -> Any:
    """Launch one headless Chromium on the running event loop and return the browser.

    Launch arguments: ``--no-sandbox`` and ``--disable-dev-shm-usage`` are
    required for Chromium inside the Docker runtime (the container lacks the
    user namespaces the Chromium sandbox needs, and the default 64 MiB
    ``/dev/shm`` is too small for page rendering) and are harmless on a
    development host.
    """
    global _playwright_driver

    if _playwright_driver is None:
        from playwright.async_api import async_playwright

        _playwright_driver = await async_playwright().start()

    launch_keyword_arguments: dict[str, Any] = {
        "headless": True,
        "args": ["--no-sandbox", "--disable-dev-shm-usage"],
    }
    if context.browser_chromium_executable_path:
        launch_keyword_arguments["executable_path"] = (
            context.browser_chromium_executable_path
        )
    return await _playwright_driver.chromium.launch(**launch_keyword_arguments)


async def _close_conversation_browser(
    conversation_key: str, record: _ConversationBrowser, reason: str
) -> None:
    """Close one conversation's browser, tolerating an already-dead process."""
    try:
        await record.browser.close()
    except Exception:
        logger.debug(
            "Playwright browser tools: close failed for conversation %s (%s); "
            "the browser process was likely already gone.",
            conversation_key,
            reason,
            exc_info=True,
        )


async def _evict_stale_conversation_browsers(context: GlobalContext) -> None:
    """Drop idle / dead browsers, then enforce the concurrent-conversation cap.

    Called with ``_browser_registry_lock`` held. Browsers holding an active
    turn lease (``active_turn_count > 0``) are never closed — neither by the
    idle timeout nor by least-recently-used eviction — so an in-flight turn
    can never lose the browser mid-browse. When every browser is leased the
    cap is temporarily exceeded instead (soft cap).
    """
    now_monotonic_seconds = time.monotonic()
    idle_timeout_seconds = context.browser_conversation_idle_timeout_seconds

    for conversation_key, record in list(_conversation_browsers.items()):
        idle_seconds = now_monotonic_seconds - record.last_used_monotonic_seconds
        if not record.browser.is_connected():
            # The Chromium process already died on its own; only the registry
            # entry is left to remove, lease or not.
            del _conversation_browsers[conversation_key]
        elif record.active_turn_count > 0:
            continue
        elif idle_seconds > idle_timeout_seconds:
            del _conversation_browsers[conversation_key]
            await _close_conversation_browser(conversation_key, record, "idle timeout")

    maximum_concurrent = context.browser_max_concurrent_conversations
    while len(_conversation_browsers) >= maximum_concurrent:
        unleased_keys = [
            conversation_key
            for conversation_key, record in _conversation_browsers.items()
            if record.active_turn_count == 0
        ]
        if not unleased_keys:
            # Every browser is mid-turn: exceed the cap rather than close a
            # browser another conversation is actively using.
            break
        least_recent_key = min(
            unleased_keys,
            key=lambda key: _conversation_browsers[key].last_used_monotonic_seconds,
        )
        least_recent_record = _conversation_browsers.pop(least_recent_key)
        await _close_conversation_browser(
            least_recent_key, least_recent_record, "least-recently-used eviction"
        )


async def get_browser_toolkit_tools(
    context: GlobalContext | None = None,
    conversation_key: str | None = None,
) -> list[BaseTool]:
    """Return browser tools bound to this conversation's own Chromium process.

    Args:
        context: Optional pre-instantiated ``GlobalContext``.
        conversation_key: Stable identifier of the conversation — ``think``
            passes the outer workflow thread id — so repeated turns of one
            conversation reuse one browser while distinct conversations get
            distinct browsers (isolated pages, cookies, history).

    Empty list is returned when the ``BROWSER_TOOLS_ENABLED`` gate is off or
    when a previous launch attempt failed — both degrade the turn to the
    normal browserless tool set rather than raising.

    A non-empty return also takes a turn lease on the conversation's browser
    (eviction will not close the browser while the lease is held); the caller
    MUST pair the call with ``release_conversation_browser`` in a ``finally``.
    """
    global _browser_launch_failed

    context = context or GlobalContext()
    if not browser_tools_enabled(context):
        return []

    conversation_key = conversation_key or _FALLBACK_CONVERSATION_KEY

    async with _browser_registry_lock:
        if _browser_launch_failed:
            return []

        record = _conversation_browsers.get(conversation_key)
        if record is None or not record.browser.is_connected():
            await _evict_stale_conversation_browsers(context)
            try:
                browser = await _launch_conversation_browser(context)
            except Exception:
                _browser_launch_failed = True
                logger.warning(
                    "Playwright browser tools: Chromium launch failed; "
                    "browser tools disabled for the lifetime of this process. "
                    "Check BROWSER_CHROMIUM_EXECUTABLE_PATH or run "
                    "`playwright install chromium`.",
                    exc_info=True,
                )
                return []
            record = _ConversationBrowser(
                browser=browser,
                last_used_monotonic_seconds=time.monotonic(),
            )
            _conversation_browsers[conversation_key] = record
            logger.info(
                "Playwright browser tools: launched headless Chromium for "
                "conversation %s (executable_path=%s, active_browsers=%d)",
                conversation_key,
                context.browser_chromium_executable_path or "playwright-managed",
                len(_conversation_browsers),
            )

        record.last_used_monotonic_seconds = time.monotonic()
        record.active_turn_count += 1

    # Lazy import per the cold-start convention: langchain_community pulls in
    # a large dependency tree and must not load at module scope.
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=record.browser)
    return toolkit.get_tools()


async def release_conversation_browser(conversation_key: str | None = None) -> None:
    """Release the turn lease taken by a non-empty ``get_browser_toolkit_tools``.

    Called from the turn's ``finally`` so the lease drops on every exit path —
    normal completion, errors, and human-in-the-loop interrupts alike. The
    recency timestamp is refreshed on release so least-recently-used eviction
    ranks the conversation by when browsing actually ended, not when the turn
    started. Missing registry entries are tolerated (the browser may have
    died and been dropped mid-turn).
    """
    conversation_key = conversation_key or _FALLBACK_CONVERSATION_KEY
    async with _browser_registry_lock:
        record = _conversation_browsers.get(conversation_key)
        if record is not None and record.active_turn_count > 0:
            record.active_turn_count -= 1
            record.last_used_monotonic_seconds = time.monotonic()


__all__ = [
    "browser_tools_enabled",
    "get_browser_toolkit_tools",
    "release_conversation_browser",
]
