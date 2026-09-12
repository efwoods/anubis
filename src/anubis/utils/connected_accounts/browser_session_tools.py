"""Tools that act inside the sites the owner signed in to.

Every tool names the connection by the account's label, opens (or reuses) the
signed-in browser context for that record, and works on the site the record
belongs to — never elsewhere: a call that names another host is refused, so
a session for one dashboard cannot be used to browse a second site.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

BROWSER_SESSION_TOOL_NAMES: tuple[str, ...] = (
    "open_connected_site",
    "read_connected_page",
    "fetch_connected_json",
    "find_on_connected_site",
    "click_connected_element",
    "type_into_connected_field",
    "run_provider_recipe",
)

_MAX_PAGE_CHARACTERS = 12000


def html_to_text(html: str, limit: int = _MAX_PAGE_CHARACTERS) -> str:
    """Reduce a page to readable text (scripts and styles dropped)."""
    text = re.sub(
        r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", str(html or "")
    )
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def build_browser_session_tools(
    context: Any,
    accounts: list[dict[str, Any]],
    *,
    store: Any = None,
    bundle: Any = None,
) -> list[Any]:
    """Build the connected-site tools for every browser-session account."""
    from src.anubis.utils.connected_accounts.browser_sessions import (
        BrowserSessionError,
        BrowserSessionExpired,
        home_url_for,
        hostname_of,
        open_session,
        persist_session_state,
        release_session,
        same_site,
    )
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import mark_account_needs_reconnect

    session_accounts = [
        record
        for record in accounts
        if record.get("credential_mechanism") == "browser_session"
    ]
    if not session_accounts:
        return []

    def _labels() -> list[str]:
        return [str(record.get("display_label") or "") for record in session_accounts]

    def _select(
        connection: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if connection is None or not str(connection).strip():
            if len(session_accounts) == 1:
                return session_accounts[0], None
            return None, {
                "status": "ambiguous",
                "error": f"Several sites are connected; name one with the connection argument: {_labels()}.",
            }
        wanted = str(connection).strip().lower()
        for record in session_accounts:
            if wanted in (
                str(record.get("display_label") or "").lower(),
                str(record.get("provider") or "").lower(),
                str(record.get("account_address") or "").lower(),
            ):
                return record, None
        return None, {
            "status": "unknown_connection",
            "error": f"No connected site named {connection!r}. Connected: {_labels()}.",
        }

    def _site_hostname(record: dict[str, Any]) -> str:
        transport = record.get("transport") or {}
        return str(
            transport.get("hostname")
            or hostname_of(home_url_for(record))
            or record.get("account_address")
            or ""
        ).split("#", 1)[0]

    async def _with_session(record: dict[str, Any], operation: Any) -> dict[str, Any]:
        label = record.get("display_label")
        user_id = str(record.get("user_id") or "")
        try:
            handle = await open_session(context, store, user_id, record, lease=True)
        except BrowserSessionExpired as expired:
            await mark_account_needs_reconnect(
                store, user_id, str(record.get("account_key") or "")
            )
            # Carry what the connect card needs, so raising it is a single call
            # with the right arguments rather than the model reconstructing the
            # site from the label. A lapse should cost the owner one click.
            return {
                "status": "needs_reconnect",
                "connection": label,
                "provider": record.get("provider"),
                "site_url": home_url_for(record),
                "error": (
                    f"{label}: {expired}. The stored sign-in has lapsed — call "
                    "connect_account with this provider and site_url to put the "
                    "sign-in card in front of the owner."
                ),
            }
        except BrowserSessionError as session_error:
            return {
                "status": "error",
                "connection": label,
                "error": session_error.detail,
            }
        try:
            async with handle.lock:
                result = await asyncio.wait_for(operation(handle), timeout=60.0)
            try:
                await persist_session_state(context, store, user_id, record, handle)
            except Exception:
                logger.debug(
                    "Could not persist session state after a tool call", exc_info=True
                )
            return result
        except TimeoutError:
            return {
                "status": "timeout",
                "connection": label,
                "error": "The site did not answer within a minute.",
            }
        except Exception as operation_error:
            logger.info("Connected-site tool failed for %s: %s", label, operation_error)
            return {
                "status": "error",
                "connection": label,
                "error": str(operation_error),
            }
        finally:
            release_session(handle)

    def _resolve_url(record: dict[str, Any], path_or_url: str) -> str | None:
        home = home_url_for(record)
        candidate = str(path_or_url or "").strip() or "/"
        if candidate.startswith("http"):
            url = candidate
        else:
            from urllib.parse import urljoin

            url = urljoin(home, candidate)
        return url if same_site(url, _site_hostname(record)) else None

    def _refuse_off_site(record: dict[str, Any], url: str) -> dict[str, Any]:
        return {
            "status": "refused",
            "error": f"{url} is not on {_site_hostname(record)}; a connected site's session may only be used on that site.",
        }

    @tool
    async def open_connected_site(
        connection: str | None = None, path: str = "/"
    ) -> dict[str, Any]:
        """Open a page of a site the owner signed in to and return the page as text.

        Use for "what does my dashboard say", "check my account on <site>", or
        as the first step before acting on a site. ``connection`` is the site's
        label (optional when only one site is connected); ``path`` is a path on
        that site or a full address on the same site.
        """
        record, error = _select(connection)
        if error:
            return error
        url = _resolve_url(record, path)
        if url is None:
            return _refuse_off_site(record, str(path))

        async def _operation(handle: Any) -> dict[str, Any]:
            from src.anubis.utils.connected_accounts.browser_sessions import (
                bot_wall_advice,
                bot_wall_detected,
                hostname_of,
            )

            await handle.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(0.8)
            html = await handle.page.content()
            # A bot wall renders as an ordinary page, so without this the
            # avatar would read "Sorry, you have been blocked" back as though
            # it were the dashboard — the answer-shaped wrong answer this whole
            # tool exists to avoid.
            if bot_wall_detected(html):
                return {
                    "status": "blocked",
                    "connection": record.get("display_label"),
                    "url": handle.page.url,
                    "detail": bot_wall_advice(record, hostname_of(url) or ""),
                }
            return {
                "status": "ok",
                "connection": record.get("display_label"),
                "url": handle.page.url,
                "title": await handle.page.title(),
                "text": html_to_text(html),
            }

        return await _with_session(record, _operation)

    @tool
    async def read_connected_page(
        connection: str | None = None, url: str = ""
    ) -> dict[str, Any]:
        """Read one page of a connected site as text (same as open_connected_site with a full address)."""
        return await open_connected_site.coroutine(
            connection=connection, path=url or "/"
        )

    @tool
    async def fetch_connected_json(
        connection: str | None = None,
        url: str = "",
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a JSON endpoint of a connected site with the owner's signed-in session.

        Use when a dashboard loads figures from an internal endpoint (found by
        reading the page or from a vendor recipe). The endpoint must be on the
        connected site. Returns the parsed JSON, truncated when large.
        """
        record, error = _select(connection)
        if error:
            return error
        target = _resolve_url(record, url)
        if target is None:
            return _refuse_off_site(record, url)

        async def _operation(handle: Any) -> dict[str, Any]:
            request_context = handle.context.request
            response = await request_context.fetch(
                target,
                method=str(method or "GET").upper(),
                data=body if body else None,
                headers={"Accept": "application/json"},
            )
            text = await response.text()
            try:
                import json

                document = json.loads(text)
            except Exception:
                document = None
            snippet = (
                text
                if len(text) <= _MAX_PAGE_CHARACTERS
                else text[:_MAX_PAGE_CHARACTERS] + "…"
            )
            from src.anubis.utils.connected_accounts.browser_sessions import (
                bot_wall_advice,
                bot_wall_detected,
                hostname_of,
            )

            if bot_wall_detected(text, status_code=response.status):
                return {
                    "status": "blocked",
                    "connection": record.get("display_label"),
                    "url": target,
                    "status_code": response.status,
                    "detail": bot_wall_advice(record, hostname_of(target) or ""),
                }
            return {
                "status": "ok" if response.ok else "http_error",
                "connection": record.get("display_label"),
                "url": target,
                "status_code": response.status,
                "json": document
                if document is not None and len(text) <= 200000
                else None,
                "text": None if document is not None else snippet,
            }

        return await _with_session(record, _operation)

    @tool
    async def find_on_connected_site(
        connection: str | None = None, query: str = ""
    ) -> dict[str, Any]:
        """Search the current page of a connected site for a phrase and return the lines around each hit."""
        record, error = _select(connection)
        if error:
            return error
        needle = str(query or "").strip().lower()
        if not needle:
            return {"status": "error", "error": "A query is required."}

        async def _operation(handle: Any) -> dict[str, Any]:
            text = html_to_text(await handle.page.content(), limit=200000)
            lines = text.splitlines()
            hits = [
                " / ".join(lines[max(0, index - 1) : index + 2])
                for index, line in enumerate(lines)
                if needle in line.lower()
            ]
            return {
                "status": "ok",
                "connection": record.get("display_label"),
                "url": handle.page.url,
                "matches": hits[:30],
                "match_count": len(hits),
            }

        return await _with_session(record, _operation)

    @tool
    async def click_connected_element(
        connection: str | None = None, selector: str = "", text: str = ""
    ) -> dict[str, Any]:
        """Click an element on the current page of a connected site, by CSS selector or by visible text.

        Only when the owner asked for that action in this conversation.
        """
        record, error = _select(connection)
        if error:
            return error

        async def _operation(handle: Any) -> dict[str, Any]:
            if selector:
                await handle.page.click(selector, timeout=15000)
            elif text:
                await handle.page.get_by_text(text, exact=False).first.click(
                    timeout=15000
                )
            else:
                return {
                    "status": "error",
                    "error": "A selector or visible text is required.",
                }
            await asyncio.sleep(0.8)
            return {
                "status": "ok",
                "connection": record.get("display_label"),
                "url": handle.page.url,
                "title": await handle.page.title(),
                "text": html_to_text(await handle.page.content(), limit=4000),
            }

        return await _with_session(record, _operation)

    @tool
    async def type_into_connected_field(
        connection: str | None = None,
        selector: str = "",
        text: str = "",
        submit: bool = False,
    ) -> dict[str, Any]:
        """Type text into a field on the current page of a connected site (optionally press Enter).

        Never use this tool to enter a password or a secret; the owner signs in
        through the connect card. Only when the owner asked for the action.
        """
        record, error = _select(connection)
        if error:
            return error
        if not selector:
            return {"status": "error", "error": "A selector is required."}

        async def _operation(handle: Any) -> dict[str, Any]:
            await handle.page.fill(selector, str(text or ""), timeout=15000)
            if submit:
                await handle.page.press(selector, "Enter")
                await asyncio.sleep(0.8)
            return {
                "status": "ok",
                "connection": record.get("display_label"),
                "url": handle.page.url,
            }

        return await _with_session(record, _operation)

    @tool
    async def run_provider_recipe(
        connection: str | None = None, recipe: str = "", period: str = "30d"
    ) -> dict[str, Any]:
        """Read a vendor's usage or cost figures through the owner's signed-in session.

        Use for "how much did LangSmith / OpenAI / Anthropic cost this month".
        ``recipe`` names one of the vendor's recipes (omit to list them);
        ``period`` is 7d, 30d, or 90d. Rows read are stored as daily vendor usage
        so charts and reports can use them; the tool says plainly when a page
        could not be read.
        """
        from src.anubis.utils.connected_accounts.recipes import (
            RECIPE_KIND_JSON,
            recipes_for,
            render_url,
        )

        record, error = _select(connection)
        if error:
            return error
        transport = record.get("transport") or {}
        recipe_key = (
            transport.get("recipe_key")
            or get_provider(str(record.get("provider") or "")).recipe_key
            if get_provider(str(record.get("provider") or ""))
            else None
        )
        available = recipes_for(recipe_key)
        if not available:
            return {
                "status": "no_recipes",
                "connection": record.get("display_label"),
                "message": "No usage recipe is known for this site; read the usage page with open_connected_site instead.",
            }
        if not recipe:
            return {
                "status": "ok",
                "recipes": {
                    name: entry.description for name, entry in available.items()
                },
            }
        chosen = available.get(str(recipe).strip().lower())
        if chosen is None:
            return {"status": "unknown_recipe", "recipes": list(available)}
        url = render_url(chosen, period)

        async def _operation(handle: Any) -> dict[str, Any]:
            if chosen.kind == RECIPE_KIND_JSON:
                response = await handle.context.request.fetch(
                    url,
                    method=chosen.method,
                    headers={"Accept": "application/json", **chosen.headers},
                )
                text = await response.text()
                if not response.ok:
                    return {
                        "status": "http_error",
                        "status_code": response.status,
                        "url": url,
                        "message": "The vendor did not answer this endpoint for the signed-in session; try the usage_page recipe or add an API key for exact figures.",
                    }
                import json

                try:
                    document = json.loads(text)
                except Exception:
                    return {"status": "unreadable", "url": url}
                rows = chosen.parser(document, {"period": period})
            else:
                await handle.page.goto(
                    url, wait_until="domcontentloaded", timeout=30000
                )
                await asyncio.sleep(1.5)
                rows = chosen.parser(
                    html_to_text(await handle.page.content(), limit=200000),
                    {"period": period},
                )
            stored = 0
            try:
                from src.anubis.utils.analytics.vendor_usage import record_rows
                from src.anubis.utils.runtime_handles import get_postgres_pool

                pool = get_postgres_pool()
                if pool is not None and rows:
                    stored = await record_rows(
                        pool,
                        str(record.get("user_id") or ""),
                        str(record.get("provider") or ""),
                        rows,
                        "browser_recipe",
                    )
            except ImportError:
                pass
            except Exception:
                logger.debug("Could not store vendor usage rows", exc_info=True)
            return {
                "status": "ok",
                "connection": record.get("display_label"),
                "recipe": chosen.name,
                "period": period,
                "rows": rows[:200],
                "row_count": len(rows),
                "stored": stored,
            }

        return await _with_session(record, _operation)

    return [
        open_connected_site,
        read_connected_page,
        fetch_connected_json,
        find_on_connected_site,
        click_connected_element,
        type_into_connected_field,
        run_provider_recipe,
    ]
