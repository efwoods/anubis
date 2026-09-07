"""Bank and card accounts through Plaid Link, opened from the connect card.

Plaid Link is the bank's own sign-in, hosted by Plaid: the owner picks the
institution and signs in there; Neural Nexus never sees the bank password.
The flow is three requests:

1. ``POST /connect_account/plaid/link_token`` (signed-in owner) creates a Plaid
   link token and returns the address of an API-hosted page that loads Plaid
   Link. The page address carries a signed login token because a popup cannot
   send an Authorization header.
2. ``GET /connect_account/plaid/link?t=…`` serves that page. Plaid Link runs in
   the popup; on success the page posts the short-lived ``public_token`` back.
3. ``POST /connect_account/plaid/exchange`` (login token) exchanges the public
   token for the item's access token, reads the institution and accounts, and
   stores the encrypted access token. The page then posts the same non-secret
   result contract every popup login uses and closes.

Plaid's own SDK is not used: three JSON endpoints over ``httpx`` are all this
needs, and one dependency fewer is one wheel fewer to keep compatible with the
Python 3.11 runtime image.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Any

from src.anubis.utils.connected_accounts.oauth_flow import LOGIN_RESULT_MESSAGE_TYPE
from src.anubis.utils.connected_accounts.oauth_state import (
    random_nonce,
    sign_state,
    state_secret,
)
from src.anubis.utils.connected_accounts.pending_logins import (
    MODE_PLAID,
    build_pending_row,
    get_pending_login_repository,
)

logger = logging.getLogger(__name__)

PLAID_LINK_PAGE_PATH = "/connect_account/plaid/link"
PLAID_EXCHANGE_PATH = "/connect_account/plaid/exchange"
PLAID_LINK_SCRIPT_URL = "https://cdn.plaid.com/link/v2/stable/link-initialize.js"


class PlaidLinkError(Exception):
    """Plaid refused a request; ``detail`` is owner-safe."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Carry the HTTP status the route should answer with, and why."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def plaid_base_url(context: Any) -> str:
    """Return the Plaid API base for the configured environment."""
    environment = str(getattr(context, "plaid_environment", None) or "sandbox").strip().lower()
    if environment not in ("sandbox", "development", "production"):
        environment = "sandbox"
    return f"https://{environment}.plaid.com"


def _credentials(context: Any) -> tuple[str, str]:
    client_id = str(getattr(context, "plaid_client_id", "") or "").strip()
    secret = str(getattr(context, "plaid_secret", "") or "").strip()
    if not client_id or not secret:
        raise PlaidLinkError(
            503,
            "Bank connections are not configured on this server "
            "(PLAID_CLIENT_ID and PLAID_SECRET are empty).",
        )
    return client_id, secret


async def plaid_post(
    context: Any, path: str, body: dict[str, Any], *, http_client: Any | None = None
) -> dict[str, Any]:
    """POST one Plaid request with the client credentials added."""
    import httpx

    client_id, secret = _credentials(context)
    timeout_seconds = float(getattr(context, "connect_oauth_http_timeout_seconds", None) or 15.0)
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        response = await client.post(
            plaid_base_url(context) + path,
            json={"client_id": client_id, "secret": secret, **body},
            headers={"Content-Type": "application/json"},
        )
    except Exception as request_error:
        raise PlaidLinkError(503, f"Plaid could not be reached: {request_error}") from request_error
    finally:
        if owned:
            await client.aclose()
    try:
        document = response.json()
    except Exception:
        document = {}
    if response.status_code >= 400:
        message = (document or {}).get("error_message") or f"Plaid answered {response.status_code}."
        raise PlaidLinkError(400 if response.status_code < 500 else 503, str(message))
    return document if isinstance(document, dict) else {}


async def start_plaid_link(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    provider: Any,
    repository: Any | None = None,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Create a link token and return ``{"link_url", "nonce", "expires_in"}``."""
    repository = repository or get_pending_login_repository()
    products = [
        entry.strip()
        for entry in str(getattr(context, "plaid_products", None) or "transactions").split(",")
        if entry.strip()
    ]
    country_codes = [
        entry.strip().upper()
        for entry in str(getattr(context, "plaid_country_codes", None) or "US").split(",")
        if entry.strip()
    ]
    nonce = random_nonce()
    document = await plaid_post(
        context,
        "/link/token/create",
        {
            "user": {"client_user_id": user_id},
            "client_name": str(getattr(context, "mcp_oauth_client_name", None) or "Neural Nexus"),
            "products": products,
            "country_codes": country_codes,
            "language": "en",
        },
        http_client=http_client,
    )
    link_token = str(document.get("link_token") or "")
    if not link_token:
        raise PlaidLinkError(503, "Plaid did not return a link token.")
    max_age = int(getattr(context, "connect_oauth_state_max_age_seconds", None) or 600)
    login_token = sign_state(
        {
            "nonce": nonce,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "provider": provider.name,
            "mode": MODE_PLAID,
        },
        state_secret(context),
        max_age,
    )
    await repository.create(
        build_pending_row(
            nonce=nonce,
            user_id=user_id,
            assistant_id=assistant_id,
            provider=provider.name,
            mode=MODE_PLAID,
            payload={"link_token": link_token},
            max_age_seconds=max_age,
        )
    )
    return {
        "link_url": f"{PLAID_LINK_PAGE_PATH}?t={login_token}",
        "nonce": nonce,
        "expires_in": max_age,
        "provider": provider.name,
    }


async def exchange_public_token(
    context: Any,
    *,
    public_token: str,
    pending: dict[str, Any],
    existing_records: list[dict[str, Any]],
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Exchange a public token, read institution and accounts, return the record."""
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
    )
    from src.anubis.utils.secret_store import encrypt_secret

    provider = get_provider(str(pending.get("provider") or "plaid"))
    exchanged = await plaid_post(
        context, "/item/public_token/exchange", {"public_token": public_token}, http_client=http_client
    )
    access_token = str(exchanged.get("access_token") or "")
    item_id = str(exchanged.get("item_id") or "")
    if not access_token or not item_id:
        raise PlaidLinkError(400, "Plaid did not return an access token for the item.")

    accounts_document = await plaid_post(
        context, "/accounts/get", {"access_token": access_token}, http_client=http_client
    )
    accounts = [
        {
            "account_id": entry.get("account_id"),
            "name": entry.get("name") or entry.get("official_name"),
            "mask": entry.get("mask"),
            "type": entry.get("type"),
            "subtype": entry.get("subtype"),
        }
        for entry in accounts_document.get("accounts") or []
        if isinstance(entry, dict)
    ]
    institution_id = str(((accounts_document.get("item") or {}).get("institution_id")) or "")
    institution_name = ""
    if institution_id:
        try:
            institution = await plaid_post(
                context,
                "/institutions/get_by_id",
                {
                    "institution_id": institution_id,
                    "country_codes": [
                        entry.strip().upper()
                        for entry in str(getattr(context, "plaid_country_codes", None) or "US").split(",")
                        if entry.strip()
                    ],
                },
                http_client=http_client,
            )
            institution_name = str(((institution.get("institution") or {}).get("name")) or "")
        except PlaidLinkError:
            institution_name = ""
    label_base = institution_name or "Bank"
    key = account_key(provider.name, item_id)
    label = deduplicate_label(label_base, existing_records, key)
    record = build_account_record(
        provider=provider,
        account_address=item_id,
        display_label=label,
        encrypted_secret=encrypt_secret(access_token, context),
        assistant_id=str(pending["assistant_id"]),
        transport={
            "item_id": item_id,
            "institution_id": institution_id,
            "institution_name": institution_name,
            "accounts": accounts,
            "environment": str(getattr(context, "plaid_environment", None) or "sandbox"),
        },
    )
    record["user_id"] = str(pending["user_id"])
    return record


def render_link_page_html(
    *, link_token: str, nonce: str, login_token: str, allowed_origins: list[str], exchange_path: str = PLAID_EXCHANGE_PATH
) -> str:
    """Render the popup page that runs Plaid Link and reports back."""
    result_template = {
        "type": LOGIN_RESULT_MESSAGE_TYPE,
        "ok": False,
        "nonce": nonce,
        "provider": "plaid",
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Connect your bank</title>
<style>body{{font-family:system-ui,sans-serif;background:#0b0b0d;color:#e8e8ea;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}}main{{text-align:center;max-width:28rem;padding:2rem}}
h1{{font-size:1.25rem;margin:0 0 .5rem}}p{{color:#a3a3a8;margin:0}}button{{margin-top:1rem;padding:.6rem 1.2rem;border-radius:999px;border:1px solid #444;background:#1a1a1f;color:#fff;cursor:pointer}}</style>
<script src="{PLAID_LINK_SCRIPT_URL}"></script></head>
<body><main><h1>Connect your bank</h1><p id="status">Opening your bank's sign-in…</p>
<button id="retry" hidden>Try again</button></main>
<script>
(function () {{
  var origins = {json.dumps(list(allowed_origins))};
  var base = {json.dumps(result_template)};
  var statusLine = document.getElementById('status');
  var retry = document.getElementById('retry');
  function post(result) {{
    var payload = Object.assign({{}}, base, result);
    try {{
      if (window.opener && !window.opener.closed) {{
        for (var i = 0; i < origins.length; i += 1) {{
          try {{ window.opener.postMessage(payload, origins[i]); }} catch (e) {{}}
        }}
      }}
    }} catch (e) {{}}
  }}
  function finish(result, closeAfter) {{
    post(result);
    statusLine.textContent = result.ok
      ? ((result.display_label || 'Your bank') + ' is connected. You can close this window.')
      : (result.error || 'The bank was not connected.');
    if (closeAfter) {{ setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 600); }}
    else {{ retry.hidden = false; }}
  }}
  function open() {{
    retry.hidden = true;
    var handler = Plaid.create({{
      token: {json.dumps(link_token)},
      onSuccess: function (publicToken) {{
        statusLine.textContent = 'Finishing the connection…';
        fetch({json.dumps(exchange_path)}, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json', 'X-Login-Token': {json.dumps(login_token)} }},
          body: JSON.stringify({{ public_token: publicToken, nonce: {json.dumps(nonce)} }})
        }}).then(function (response) {{ return response.json(); }})
          .then(function (body) {{ finish(body, !!body.ok); }})
          .catch(function () {{ finish({{ ok: false, error: 'The connection could not be finished.' }}, false); }});
      }},
      onExit: function (error) {{
        finish({{ ok: false, error: error ? (error.display_message || error.error_message || 'The sign-in was closed.') : 'The sign-in was closed.' }}, false);
      }}
    }});
    handler.open();
  }}
  retry.addEventListener('click', open);
  if (typeof Plaid === 'undefined') {{
    finish({{ ok: false, error: 'Plaid Link could not load. Check the network and try again.' }}, false);
  }} else {{ open(); }}
}})();
</script></body></html>"""


def escape_for_html(text: str) -> str:
    """Escape text for an HTML attribute or body."""
    return html.escape(str(text or ""))
