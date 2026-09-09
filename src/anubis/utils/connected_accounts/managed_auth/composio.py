"""The managed-auth provider backed by Composio's certified OAuth clients.

Composio holds OAuth applications that have already passed the assessments
Google, Microsoft, and Yahoo require, and connects a user's account against a
user identifier we supply. So a Gmail or Google Calendar account connects the
same day, for any user, with no cap and no security assessment of our own — the
one cost being that the consent screen carries Composio's name.

**No Google token reaches this codebase.** A connection is stored as Composio's
connection identifier; the token lives with Composio, which refreshes it. Tools
run by asking Composio to execute them as the user, so the credential is never
in a request this server makes either.

A note on the endpoint table below, because it is the one fragile thing here.
Composio's published reference spans two API versions (``v3`` and ``v3.1``) and
spells some paths differently between them, and these were written from the
documentation rather than against a live key. Every path and field name is
therefore in ``ENDPOINTS`` and ``FIELDS`` at the top of this module rather than
scattered through the functions, and :meth:`ComposioProvider.verify_configuration`
exercises them and reports exactly which call disagreed. Correcting a drifted
path is a one-line edit to the table, and the tests below prove the surrounding
logic independently of which paths are right.
"""

from __future__ import annotations

import logging
from typing import Any

from src.anubis.utils.connected_accounts.managed_auth.base import (
    STATE_CONNECTED,
    STATE_EXPIRED,
    STATE_FAILED,
    STATE_PENDING,
    ManagedAuthError,
    ManagedAuthNotConfigured,
    ManagedConnection,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://backend.composio.dev/api/v3"
API_KEY_HEADER = "x-api-key"

# Every vendor path in one place. See the module docstring.
ENDPOINTS: dict[str, str] = {
    "auth_configs": "/auth_configs",
    "connected_accounts": "/connected_accounts",
    "connected_account": "/connected_accounts/{connection_id}",
    "tools": "/tools",
    "execute_tool": "/tools/execute/{tool_slug}",
}

# Every vendor field name in one place, for the same reason.
FIELDS: dict[str, str] = {
    "auth_config_id": "id",
    "redirect_url": "redirect_url",
    "connection_id": "id",
    "status": "status",
}

# Composio's connection states, mapped onto ours so no caller reads a vendor
# spelling. Anything unrecognised is treated as still pending, which is the
# safe reading: the card keeps waiting rather than declaring a failure.
STATE_BY_VENDOR_STATUS: dict[str, str] = {
    "ACTIVE": STATE_CONNECTED,
    "INITIATED": STATE_PENDING,
    "INITIALIZING": STATE_PENDING,
    "PENDING": STATE_PENDING,
    "EXPIRED": STATE_EXPIRED,
    "FAILED": STATE_FAILED,
    "INACTIVE": STATE_FAILED,
    "DISABLED": STATE_FAILED,
}

# Toolkit slugs for the providers that need this path.
TOOLKIT_BY_PROVIDER: dict[str, str] = {
    "gmail": "gmail",
    "google_calendar": "googlecalendar",
    "outlook": "outlook",
    "yahoo": "yahoo",
}


def toolkit_for_provider(provider_name: str) -> str:
    """Return the vendor's toolkit slug for one of our provider rows."""
    return TOOLKIT_BY_PROVIDER.get(str(provider_name or "").strip().lower(), "")


class ComposioProvider:
    """Composio, behind the :class:`ManagedAuthProvider` protocol."""

    name = "composio"

    def __init__(self, context: Any, *, http_client: Any = None) -> None:
        """Read the configured key and base address off the context."""
        self._context = context
        self._http_client = http_client
        self._api_key = str(getattr(context, "composio_api_key", None) or "").strip()
        self._base_url = str(
            getattr(context, "composio_base_url", None) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = float(
            getattr(context, "composio_http_timeout_seconds", None) or 30.0
        )
        if not self._api_key:
            raise ManagedAuthNotConfigured(
                "COMPOSIO_API_KEY is empty, so accounts that need a certified "
                "sign-in cannot be connected yet."
            )
        self._auth_config_ids: dict[str, str] = {}

    # -- transport ---------------------------------------------------------

    async def _call(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import httpx

        url = f"{self._base_url}{path}"
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient()
        try:
            response = await client.request(
                method,
                url,
                json=json_body,
                params=params,
                headers={API_KEY_HEADER: self._api_key, "accept": "application/json"},
                timeout=self._timeout,
            )
        except Exception as transport_error:  # noqa: BLE001 - reported, not raised on
            raise ManagedAuthError(
                f"Could not reach the managed-auth provider at {url}: "
                f"{transport_error}"
            ) from transport_error
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code in (401, 403):
            raise ManagedAuthError(
                "The managed-auth provider rejected the configured API key."
            )
        if response.status_code >= 400:
            raise ManagedAuthError(
                f"{method} {path} answered {response.status_code}: "
                f"{response.text[:300]}"
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as decode_error:
            raise ManagedAuthError(
                f"{method} {path} did not answer with JSON."
            ) from decode_error

    # -- protocol ----------------------------------------------------------

    async def ensure_auth_config(self, toolkit: str) -> str:
        """Return the configuration identifier for ``toolkit``, creating one once.

        Composio keeps one configuration per toolkit per project. An existing
        one is reused so repeated connections do not accumulate configurations,
        and the identifier is remembered for the life of the process.
        """
        slug = str(toolkit or "").strip().lower()
        if not slug:
            raise ManagedAuthError("No toolkit was named.")
        if slug in self._auth_config_ids:
            return self._auth_config_ids[slug]

        listing = await self._call(
            "GET", ENDPOINTS["auth_configs"], params={"toolkit_slug": slug}
        )
        for item in _as_items(listing):
            identifier = str(item.get(FIELDS["auth_config_id"]) or "")
            if identifier:
                self._auth_config_ids[slug] = identifier
                return identifier

        created = await self._call(
            "POST",
            ENDPOINTS["auth_configs"],
            json_body={
                "toolkit": {"slug": slug},
                "auth_config": {"type": "use_composio_managed_auth"},
            },
        )
        payload = created.get("auth_config") if isinstance(created, dict) else None
        identifier = str(
            (payload or created or {}).get(FIELDS["auth_config_id"]) or ""
        )
        if not identifier:
            raise ManagedAuthError(
                "The managed-auth provider created no configuration identifier."
            )
        self._auth_config_ids[slug] = identifier
        return identifier

    async def start_connection(
        self, *, user_id: str, toolkit: str, callback_url: str = ""
    ) -> ManagedConnection:
        """Begin a sign-in and return the URL the popup should open."""
        auth_config_id = await self.ensure_auth_config(toolkit)
        body: dict[str, Any] = {
            "user_id": user_id,
            "auth_config_id": auth_config_id,
        }
        if callback_url:
            body["callback_url"] = callback_url
        answer = await self._call(
            "POST", ENDPOINTS["connected_accounts"], json_body=body
        )
        answer = answer if isinstance(answer, dict) else {}
        connection_id = str(
            answer.get(FIELDS["connection_id"])
            or (answer.get("connected_account") or {}).get(FIELDS["connection_id"])
            or ""
        )
        authorization_url = str(
            answer.get(FIELDS["redirect_url"])
            or answer.get("redirectUrl")
            or (answer.get("connection_request") or {}).get(FIELDS["redirect_url"])
            or ""
        )
        if not authorization_url:
            raise ManagedAuthError(
                "The managed-auth provider returned no sign-in address for "
                f"{toolkit}."
            )
        return ManagedConnection(
            connection_id=connection_id,
            authorization_url=authorization_url,
            state=STATE_PENDING,
            toolkit=toolkit,
        )

    async def connection_state(self, connection_id: str) -> ManagedConnection:
        """Report where one connection has got to."""
        answer = await self._call(
            "GET",
            ENDPOINTS["connected_account"].format(connection_id=connection_id),
        )
        answer = answer if isinstance(answer, dict) else {}
        return _connection_from_payload(connection_id, answer)

    async def list_tools(self, toolkit: str) -> list[dict[str, Any]]:
        """Describe the tools a connected account of this toolkit provides."""
        listing = await self._call(
            "GET", ENDPOINTS["tools"], params={"toolkit_slug": str(toolkit).lower()}
        )
        tools: list[dict[str, Any]] = []
        for item in _as_items(listing):
            slug = str(item.get("slug") or item.get("name") or "")
            if not slug:
                continue
            tools.append(
                {
                    "slug": slug,
                    "name": str(item.get("name") or slug),
                    "description": str(item.get("description") or ""),
                    "input_schema": item.get("input_parameters")
                    or item.get("parameters")
                    or {},
                }
            )
        return tools

    async def execute_tool(
        self, *, tool_slug: str, user_id: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one tool as the given user against their connected account."""
        answer = await self._call(
            "POST",
            ENDPOINTS["execute_tool"].format(tool_slug=tool_slug),
            json_body={"user_id": user_id, "arguments": dict(arguments or {})},
        )
        return answer if isinstance(answer, dict) else {"result": answer}

    async def disconnect(self, connection_id: str) -> None:
        """Forget one connected account at the vendor."""
        await self._call(
            "DELETE",
            ENDPOINTS["connected_account"].format(connection_id=connection_id),
        )

    async def verify_configuration(self) -> dict[str, Any]:
        """Exercise each endpoint and report which ones answered.

        Written for the moment the API key first arrives: one call says whether
        the key works and whether any path in :data:`ENDPOINTS` has drifted from
        the vendor's current API, naming the one that disagreed rather than
        failing later inside a connect card.
        """
        report: dict[str, Any] = {"base_url": self._base_url, "checks": []}
        for label, method, path, params in (
            ("auth_configs", "GET", ENDPOINTS["auth_configs"], {"limit": 1}),
            ("tools", "GET", ENDPOINTS["tools"], {"toolkit_slug": "gmail", "limit": 1}),
            (
                "connected_accounts",
                "GET",
                ENDPOINTS["connected_accounts"],
                {"limit": 1},
            ),
        ):
            try:
                await self._call(method, path, params=params)
            except ManagedAuthError as error:
                report["checks"].append(
                    {"endpoint": label, "path": path, "ok": False, "error": str(error)}
                )
                continue
            report["checks"].append({"endpoint": label, "path": path, "ok": True})
        report["ok"] = all(check["ok"] for check in report["checks"])
        return report


def _as_items(payload: Any) -> list[dict[str, Any]]:
    """Read a list out of whichever envelope the vendor used."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "results", "auth_configs", "tools"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _connection_from_payload(
    connection_id: str, payload: dict[str, Any]
) -> ManagedConnection:
    body = payload.get("connected_account")
    body = body if isinstance(body, dict) else payload
    status = str(body.get(FIELDS["status"]) or "").strip().upper()
    identifier = ""
    for key in ("user_email", "email", "account_id", "account_identifier"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            identifier = value.strip()
            break
    if not identifier:
        data = body.get("data")
        if isinstance(data, dict):
            for key in ("email", "user_email", "login"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    identifier = value.strip()
                    break
    return ManagedConnection(
        connection_id=str(body.get(FIELDS["connection_id"]) or connection_id),
        state=STATE_BY_VENDOR_STATUS.get(status, STATE_PENDING),
        account_identifier=identifier,
        display_label=identifier,
        toolkit=str(
            (body.get("toolkit") or {}).get("slug")
            if isinstance(body.get("toolkit"), dict)
            else body.get("toolkit_slug") or ""
        ),
        details={"vendor_status": status},
    )
