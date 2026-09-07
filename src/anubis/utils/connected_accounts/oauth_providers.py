"""The OAuth 2.0 authorization servers a connect card can open in a popup.

One row per vendor. A provider row in ``providers.py`` names which of these
configurations to use through ``oauth_config_key`` and may narrow the scopes;
several providers share one vendor (Gmail, Google Calendar, Google Analytics,
and YouTube all sign in through the same Google client with different scopes).

Every configuration answers the same questions: where to send the owner, where
to exchange the code, which scopes to request, how to learn who signed in
(``userinfo_url`` + ``identity_from_userinfo``), and which context fields hold
the client id and secret. Nothing here performs a request; ``oauth_flow.py``
does that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Vendor keys, spelled out so a provider row referencing a vendor that does not
# exist fails registry validation rather than a live sign-in.
VENDOR_GOOGLE = "google"
VENDOR_GITHUB = "github"
VENDOR_X = "x"
VENDOR_VERCEL = "vercel"

GOOGLE_MAIL_SCOPE = "https://mail.google.com/"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
GOOGLE_ANALYTICS_SCOPE = "https://www.googleapis.com/auth/analytics.readonly"
GOOGLE_YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
GOOGLE_IDENTITY_SCOPES: tuple[str, ...] = ("openid", "email")


def _google_identity(userinfo: dict[str, Any]) -> tuple[str, str]:
    email = str(userinfo.get("email") or "").strip().lower()
    return email, email.split("@", 1)[0] or email


def _github_identity(userinfo: dict[str, Any]) -> tuple[str, str]:
    login = str(userinfo.get("login") or "").strip()
    return login.lower(), login


def _x_identity(userinfo: dict[str, Any]) -> tuple[str, str]:
    data = userinfo.get("data") if isinstance(userinfo.get("data"), dict) else userinfo
    username = str(data.get("username") or "").strip()
    return username.lower(), f"@{username}" if username else "X account"


def _vercel_identity(userinfo: dict[str, Any]) -> tuple[str, str]:
    user = userinfo.get("user") if isinstance(userinfo.get("user"), dict) else userinfo
    username = str(user.get("username") or user.get("email") or "").strip()
    return username.lower(), username or "Vercel account"


@dataclass(frozen=True)
class OAuthProviderConfig:
    """Everything needed to run the authorization-code flow against one vendor.

    Attributes:
        key: One of the ``VENDOR_*`` constants.
        authorization_url: Where the popup is sent.
        token_url: Where the code (and later the refresh token) is exchanged.
        scopes: The full scope set this vendor may be asked for; a provider row
            narrows this through ``oauth_scopes``.
        userinfo_url: Where the signed-in identity is read after the exchange.
        identity_from_userinfo: Maps the userinfo document to
            ``(account_address, display_label)``.
        client_id_field / client_secret_field: The ``GlobalContext`` fields
            holding the vendor's client credentials.
        pkce: Whether to send a PKCE challenge. Always sent where supported;
            GitHub ignores the parameter, so the flag is only off where a
            vendor rejects unknown parameters.
        extra_authorize_params: Vendor-specific query parameters (Google's
            ``access_type=offline`` and ``prompt=consent`` are what make Google
            issue a refresh token at all).
        token_auth: ``"body"`` sends the client secret in the form body,
            ``"basic"`` as HTTP Basic (X requires Basic for confidential clients).
        userinfo_headers: Extra headers for the userinfo request.
    """

    key: str
    authorization_url: str
    token_url: str
    scopes: tuple[str, ...]
    userinfo_url: str
    identity_from_userinfo: Callable[[dict[str, Any]], tuple[str, str]]
    client_id_field: str
    client_secret_field: str
    pkce: bool = True
    extra_authorize_params: dict[str, str] = field(default_factory=dict)
    token_auth: str = "body"
    userinfo_headers: dict[str, str] = field(default_factory=dict)
    scope_separator: str = " "


OAUTH_PROVIDERS: dict[str, OAuthProviderConfig] = {
    VENDOR_GOOGLE: OAuthProviderConfig(
        key=VENDOR_GOOGLE,
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=(
            *GOOGLE_IDENTITY_SCOPES,
            GOOGLE_MAIL_SCOPE,
            GOOGLE_CALENDAR_SCOPE,
            GOOGLE_ANALYTICS_SCOPE,
            GOOGLE_YOUTUBE_SCOPE,
        ),
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        identity_from_userinfo=_google_identity,
        client_id_field="google_oauth_client_id",
        client_secret_field="google_oauth_client_secret",
        extra_authorize_params={
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
        },
    ),
    VENDOR_GITHUB: OAuthProviderConfig(
        key=VENDOR_GITHUB,
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=("repo", "read:user", "user:email", "read:org"),
        userinfo_url="https://api.github.com/user",
        identity_from_userinfo=_github_identity,
        client_id_field="github_oauth_client_id",
        client_secret_field="github_oauth_client_secret",
        pkce=False,
        userinfo_headers={"Accept": "application/vnd.github+json"},
    ),
    VENDOR_X: OAuthProviderConfig(
        key=VENDOR_X,
        authorization_url="https://x.com/i/oauth2/authorize",
        token_url="https://api.x.com/2/oauth2/token",
        scopes=("tweet.read", "tweet.write", "users.read", "offline.access"),
        userinfo_url="https://api.x.com/2/users/me",
        identity_from_userinfo=_x_identity,
        client_id_field="x_oauth_client_id",
        client_secret_field="x_oauth_client_secret",
        token_auth="basic",
    ),
    VENDOR_VERCEL: OAuthProviderConfig(
        key=VENDOR_VERCEL,
        authorization_url="https://vercel.com/oauth/authorize",
        token_url="https://api.vercel.com/login/oauth/token",
        scopes=(),
        userinfo_url="https://api.vercel.com/v2/user",
        identity_from_userinfo=_vercel_identity,
        client_id_field="vercel_oauth_client_id",
        client_secret_field="vercel_oauth_client_secret",
        pkce=False,
    ),
}


def get_oauth_provider(key: str) -> OAuthProviderConfig | None:
    """Look up a vendor configuration by key."""
    return OAUTH_PROVIDERS.get(str(key or "").strip().lower())


def client_credentials(config: OAuthProviderConfig, context: Any) -> tuple[str, str]:
    """Return ``(client_id, client_secret)`` from the context, or empty strings."""
    client_id = str(getattr(context, config.client_id_field, "") or "").strip()
    client_secret = str(getattr(context, config.client_secret_field, "") or "").strip()
    return client_id, client_secret
