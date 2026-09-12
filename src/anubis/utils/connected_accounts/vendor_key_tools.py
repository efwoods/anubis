"""Vendors reached with the API key they issue, rather than a signed-in page.

Anthropic, OpenAI, and LangSmith publish no OAuth for third-party applications,
but each issues the owner a personal API key. That key is the route those
vendors document and support, so it is the route used here — in place of
keeping a signed-in browser session and reading the usage page, which was slow,
brittle, and against the terms of every one of them.

The key is proved at connect time by asking the vendor something harmless that
only a valid key can answer, and stored encrypted like every other credential.

One honest limit is built into the messages below rather than hidden. At all
three vendors, *spend and usage* figures need an **administrator** key, which
is a different key from the ordinary one used to call the models. A read that
needs one and does not have it says so by name and points at the page that
issues it, instead of returning an empty report that reads like "you spent
nothing".
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

VENDOR_KEY_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic_usage", "anthropic_models"),
    "openai": ("openai_usage", "openai_models"),
    "langsmith": ("langsmith_projects", "langsmith_runs"),
}

# Where the owner gets a key, named in every failure so nobody has to search.
KEY_PAGES: dict[str, str] = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
    "langsmith": "https://smith.langchain.com/settings",
}

ADMIN_KEY_PAGES: dict[str, str] = {
    "anthropic": "https://console.anthropic.com/settings/admin-keys",
    "openai": "https://platform.openai.com/settings/organization/admin-keys",
}

REQUEST_TIMEOUT_SECONDS = 20.0
ANTHROPIC_VERSION = "2023-06-01"


class VendorKeyRejected(Exception):
    """The vendor did not accept the key."""


class VendorUnreachable(Exception):
    """The vendor could not be reached at all."""


def _headers(provider: str, key: str) -> dict[str, str]:
    """Return the authentication header each vendor expects."""
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    if provider == "langsmith":
        return {"x-api-key": key}
    return {"Authorization": f"Bearer {key}"}


# The cheapest request at each vendor that only a valid key can answer.
VERIFY_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1/models?limit=1",
    "openai": "https://api.openai.com/v1/models",
    "langsmith": "https://api.smith.langchain.com/api/v1/sessions?limit=1",
}


async def verify_api_key(
    provider: str, key: str, *, http_client: Any = None
) -> dict[str, Any]:
    """Prove one API key before anything is stored.

    Raises :class:`VendorKeyRejected` when the vendor refuses the key and
    :class:`VendorUnreachable` when nothing answers, so a connect card can tell
    the owner which of the two happened.
    """
    import httpx

    url = VERIFY_URLS.get(provider)
    if not url:
        raise VendorKeyRejected(f"{provider} does not connect with an API key.")

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient()
    try:
        response = await client.get(
            url, headers=_headers(provider, key), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except Exception as transport_error:  # noqa: BLE001 - reported as unreachable
        raise VendorUnreachable(str(transport_error)) from transport_error
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code in (401, 403):
        raise VendorKeyRejected(
            f"That key was refused. Copy a current key from {KEY_PAGES.get(provider, '')}."
        )
    if response.status_code >= 400:
        raise VendorUnreachable(f"{url} answered {response.status_code}.")
    return {"status": "ok"}


def _period(since: str | None, until: str | None, default_days: int = 30):
    end = _moment(until) or datetime.now(UTC)
    start = _moment(since) or (end - timedelta(days=default_days))
    return start, end


def _moment(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def build_vendor_key_tools(
    context: Any, accounts: list[dict[str, Any]], *, store: Any = None
) -> list[Any]:
    """Build the tools for every account connected with an API key."""
    from langchain_core.tools import tool

    by_provider: dict[str, dict[str, Any]] = {}
    for record in accounts:
        name = str(record.get("provider") or "")
        if name in VENDOR_KEY_TOOL_NAMES:
            by_provider.setdefault(name, record)

    def _key(provider: str) -> tuple[str | None, dict[str, Any] | None]:
        from src.anubis.utils.secret_store import decrypt_secret

        record = by_provider.get(provider)
        if record is None:
            return None, {"status": "not_connected", "error": f"No {provider} account is connected."}
        try:
            return decrypt_secret(record["encrypted_secret"], context), None
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return None, {
                "status": "needs_reconnect",
                "error": f"The stored {provider} key could not be read: {error}",
            }

    async def _get(
        provider: str,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        reports_spend: bool = False,
    ):
        import httpx

        key, problem = _key(provider)
        if problem:
            return None, problem
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    url, headers=_headers(provider, key), params=params
                )
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return None, {"status": "unreachable", "error": str(error)}
        if response.status_code in (401, 403):
            admin_page = ADMIN_KEY_PAGES.get(provider)
            if admin_page and reports_spend:
                return None, {
                    "status": "needs_admin_key",
                    "error": (
                        f"{provider} reports spend only to an administrator key, "
                        f"which is a different key from the one connected. Create "
                        f"one at {admin_page} and connect it instead."
                    ),
                }
            return None, {
                "status": "needs_reconnect",
                "error": f"{provider} refused the stored key.",
            }
        if response.status_code >= 400:
            return None, {
                "status": "error",
                "status_code": response.status_code,
                "error": response.text[:400],
            }
        try:
            return response.json(), None
        except Exception:  # noqa: BLE001 - a non-JSON answer is an error
            return None, {"status": "error", "error": "The answer was not readable."}

    tools: list[Any] = []

    if "anthropic" in by_provider:

        @tool
        async def anthropic_usage(since: str | None = None, until: str | None = None) -> dict[str, Any]:
            """Report Anthropic API spend over a period. Needs an administrator key."""
            start, end = _period(since, until)
            document, problem = await _get(
                "anthropic",
                "https://api.anthropic.com/v1/organizations/cost_report",
                {"starting_at": start.isoformat(), "ending_at": end.isoformat()},
                reports_spend=True,
            )
            if problem:
                return problem
            return {"status": "ok", "since": start.isoformat(), "until": end.isoformat(), "report": document}

        @tool
        async def anthropic_models() -> dict[str, Any]:
            """List the Anthropic models this key can reach."""
            document, problem = await _get("anthropic", "https://api.anthropic.com/v1/models")
            if problem:
                return problem
            return {
                "status": "ok",
                "models": [entry.get("id") for entry in (document or {}).get("data") or []],
            }

        tools.extend([anthropic_usage, anthropic_models])

    if "openai" in by_provider:

        @tool
        async def openai_usage(since: str | None = None, until: str | None = None) -> dict[str, Any]:
            """Report OpenAI API spend over a period. Needs an administrator key."""
            start, end = _period(since, until)
            document, problem = await _get(
                "openai",
                "https://api.openai.com/v1/organization/costs",
                {"start_time": int(start.timestamp()), "end_time": int(end.timestamp())},
                reports_spend=True,
            )
            if problem:
                return problem
            return {"status": "ok", "since": start.isoformat(), "until": end.isoformat(), "report": document}

        @tool
        async def openai_models() -> dict[str, Any]:
            """List the OpenAI models this key can reach."""
            document, problem = await _get("openai", "https://api.openai.com/v1/models")
            if problem:
                return problem
            return {
                "status": "ok",
                "models": [entry.get("id") for entry in (document or {}).get("data") or []],
            }

        tools.extend([openai_usage, openai_models])

    if "langsmith" in by_provider:

        @tool
        async def langsmith_projects(limit: int = 20) -> dict[str, Any]:
            """List the owner's LangSmith projects."""
            document, problem = await _get(
                "langsmith",
                "https://api.smith.langchain.com/api/v1/sessions",
                {"limit": max(1, min(int(limit), 100))},
            )
            if problem:
                return problem
            entries = document if isinstance(document, list) else (document or {}).get("data") or []
            return {
                "status": "ok",
                "projects": [
                    {"id": entry.get("id"), "name": entry.get("name")}
                    for entry in entries
                    if isinstance(entry, dict)
                ],
            }

        @tool
        async def langsmith_runs(project_id: str, limit: int = 20) -> dict[str, Any]:
            """List recent LangSmith runs in one project, newest first."""
            document, problem = await _get(
                "langsmith",
                "https://api.smith.langchain.com/api/v1/runs",
                {"session": project_id, "limit": max(1, min(int(limit), 100))},
            )
            if problem:
                return problem
            entries = document if isinstance(document, list) else (document or {}).get("runs") or []
            return {
                "status": "ok",
                "runs": [
                    {
                        "id": entry.get("id"),
                        "name": entry.get("name"),
                        "status": entry.get("status"),
                        "started_at": entry.get("start_time"),
                        "latency_ms": entry.get("latency"),
                        "error": entry.get("error"),
                    }
                    for entry in entries
                    if isinstance(entry, dict)
                ],
            }

        tools.extend([langsmith_projects, langsmith_runs])

    return tools
