"""The registry of external accounts a personal avatar can be connected to.

One row per provider, so adding Google Calendar or X/Twitter later is a table
entry plus whatever client that provider needs — not another parallel copy of
the connect/list/disconnect machinery.

Three fields carry the design decisions:

``kind`` is a SECURITY BOUNDARY, not a taxonomy
    ``social_providers()`` is what an identity-verification gate must read when
    deciding whether the human behind a likeness has been verified. If a mailbox
    were filed as ``"social"``, connecting an email account would satisfy "this
    person proved they own the account behind this likeness" — which it plainly
    does not, since anyone can own an email address. Gmail is therefore
    ``"mailbox"``. A future data source (a Drive, a bank export) is
    ``"data_source"`` for the same reason. Only accounts that actually evidence
    a public identity may be ``"social"``.

``credential_mechanism`` decides how a connection is established
    ``"app_password"`` collects a credential in a form and verifies it by
    logging in. ``"auth0_identity"`` links a secondary identity onto the Auth0
    account and reads the provider handle back. ``"oauth"`` runs an
    authorization-code redirect and stores a refresh token. The endpoints branch
    on this field, so a provider declares its flow instead of the flow being
    hard-coded per provider.

Why Gmail is ``"app_password"`` rather than ``"oauth"``
    Reading a mailbox through the Gmail API requires an OAuth scope Google
    classifies as *restricted* (``gmail.readonly``, ``gmail.compose``,
    ``gmail.modify``, ``https://mail.google.com/``). A published application
    requesting a restricted scope must pass OAuth verification AND an annual
    CASA security assessment. Routing the flow through Auth0 does not avoid
    this, because Google's requirement attaches to the OAuth client, not to the
    broker in front of it. The escapes are an app in Testing publishing status,
    whose refresh tokens expire after seven days, or Internal status, which
    requires every user to be inside one Google Workspace organization.

    IMAP with an app password needs no OAuth client at all, so no verification
    and no assessment ever apply, and the credential does not expire. That is
    why this provider is defined the way it is, and it is also why
    ``langchain_google_community.GmailToolkit`` is not used here: the toolkit
    builds on a ``googleapiclient`` resource constructed from OAuth credentials
    and cannot authenticate with an app password. The tool surface exposed to
    the model in ``mailbox_tools`` deliberately mirrors that toolkit's tools so
    the backend can be swapped if an OAuth client is ever verified.

    NOTE for anyone maintaining this: since 2025-03-14 Google no longer accepts
    a regular account password for IMAP/SMTP. Only OAuth 2.0 and app passwords
    work, and creating an app password requires 2-Step Verification on the
    account. The connect endpoint verifies by real login precisely so a user who
    pastes their account password is told this immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Account kinds. Spelled out as constants so a typo in a provider row fails the
# membership check in `validate_registry` rather than silently creating a fourth
# kind that no gate knows about.
KIND_MAILBOX = "mailbox"
KIND_SOCIAL = "social"
KIND_DATA_SOURCE = "data_source"
ALL_KINDS = frozenset({KIND_MAILBOX, KIND_SOCIAL, KIND_DATA_SOURCE})

# Credential mechanisms.
MECHANISM_APP_PASSWORD = "app_password"
MECHANISM_AUTH0_IDENTITY = "auth0_identity"
MECHANISM_OAUTH = "oauth"
ALL_MECHANISMS = frozenset(
    {MECHANISM_APP_PASSWORD, MECHANISM_AUTH0_IDENTITY, MECHANISM_OAUTH}
)


@dataclass(frozen=True)
class ConnectFieldSpec:
    """One input the connect card renders when establishing a connection.

    The card that collects a credential is drawn by the frontend but described
    here, so a new provider ships its own form by adding a row to this table
    rather than by editing a component. ``help_text`` is not decoration: for
    Gmail it is the only place the owner is told that their account password
    will not work, and a card that omits it produces a user who types the wrong
    secret, is rejected, and types the same wrong secret again.

    Attributes:
        name: Request-body key this input fills. Must match the name the
            connect endpoint reads.
        label: Field label shown above the input.
        input_type: HTML input type. ``"password"`` for anything secret, so the
            frontend masks it without having to know which field is the secret.
        placeholder: Placeholder text shown in the empty input.
        help_text: Explanation rendered beneath the input.
    """

    name: str
    label: str
    input_type: str = "text"
    placeholder: str = ""
    help_text: str = ""


@dataclass(frozen=True)
class ConnectedAccountProvider:
    """One external account type the personal avatar can connect to.

    Attributes:
        name: Stable identifier used in the store key and in every endpoint
            argument. Never rename one of these without a migration: it is half
            of the store key ``"{provider}:{account_address}"``.
        kind: One of :data:`ALL_KINDS`. See the module docstring — this gates
            identity verification and is not cosmetic.
        credential_mechanism: One of :data:`ALL_MECHANISMS`.
        display_name: Human-readable name used in messages to the owner.
        imap_host: IMAP server, for ``app_password`` providers only.
        imap_port: IMAP TLS port, for ``app_password`` providers only.
        smtp_host: SMTP submission server, for ``app_password`` providers only.
        smtp_port: SMTP submission port, for ``app_password`` providers only.
        drafts_mailbox: IMAP folder that holds drafts. Gmail exposes this as
            ``"[Gmail]/Drafts"`` rather than the ``"Drafts"`` most other servers
            use, which is exactly the sort of per-provider detail this table
            exists to hold.
        credential_help_url: Where the owner obtains the credential. Surfaced in
            the error message when verification fails, so a user who supplied
            the wrong kind of password is told where to get the right one.
        card_description: One line naming what connecting this account lets the
            avatar do, shown on the connect card beneath the provider name.
        icon_key: Stable key the frontend maps to its own icon asset. A key
            rather than a URL because where the frontend keeps its images is the
            frontend's business, and a URL here would break every client that
            stores assets somewhere else.
        connect_fields: The inputs the connect card renders, in display order.
            Empty for providers whose mechanism is a redirect rather than a form.
    """

    name: str
    kind: str
    credential_mechanism: str
    display_name: str
    imap_host: str | None = None
    imap_port: int = 993
    smtp_host: str | None = None
    smtp_port: int = 587
    drafts_mailbox: str = "Drafts"
    credential_help_url: str | None = None
    auth0_connection: str | None = field(default=None)
    card_description: str = ""
    icon_key: str = ""
    connect_fields: tuple[ConnectFieldSpec, ...] = ()

    @property
    def is_mailbox(self) -> bool:
        """Whether this provider exposes an email mailbox."""
        return self.kind == KIND_MAILBOX


GMAIL_PROVIDER = ConnectedAccountProvider(
    name="gmail",
    kind=KIND_MAILBOX,
    credential_mechanism=MECHANISM_APP_PASSWORD,
    display_name="Gmail",
    imap_host="imap.gmail.com",
    imap_port=993,
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    drafts_mailbox="[Gmail]/Drafts",
    credential_help_url="https://myaccount.google.com/apppasswords",
    # Deliberately narrower than "manage email": the tool surface reads and
    # drafts, and there is no send tool. A card that promised management would
    # be advertising a capability the tools withhold.
    card_description="Search, read, and draft email.",
    icon_key="gmail",
    connect_fields=(
        ConnectFieldSpec(
            name="email_address",
            label="Email address",
            input_type="email",
            placeholder="you@gmail.com",
            help_text="The Gmail address of the mailbox to connect.",
        ),
        ConnectFieldSpec(
            name="app_password",
            label="App password",
            input_type="password",
            placeholder="16-character app password",
            help_text=(
                "Not your Google account password. Google stopped accepting "
                "account passwords for mail access on 14 March 2025, so a "
                "16-character app password is required. Generating one needs "
                "2-Step Verification switched on for the account."
            ),
        ),
    ),
)

# The social providers named in _SOCIAL_MEDIA_ACCOUNT_CONNECTION.md. Declared
# now, with no connect flow implemented yet, for two reasons: the registry's
# shape is only proven by holding more than one kind of row, and
# `social_providers()` needs real members so the verification gate that will
# read it is written against actual data rather than an empty tuple. Attempting
# to connect one of these is rejected by the endpoint until the Auth0 identity
# linking lands.
YOUTUBE_PROVIDER = ConnectedAccountProvider(
    name="youtube",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="YouTube",
    auth0_connection="google-oauth2",
    card_description="Read your channel's videos, descriptions, and comments.",
    icon_key="youtube",
)

TWITTER_PROVIDER = ConnectedAccountProvider(
    name="twitter",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="X (Twitter)",
    auth0_connection="twitter",
    card_description="Read your posts, timeline, and bookmarks.",
    icon_key="twitter",
)

INSTAGRAM_PROVIDER = ConnectedAccountProvider(
    name="instagram",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="Instagram",
    auth0_connection="instagram",
    card_description="Read your posts, captions, and comments.",
    icon_key="instagram",
)

TWITCH_PROVIDER = ConnectedAccountProvider(
    name="twitch",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="Twitch",
    auth0_connection="twitch",
    card_description="Read your channel, streams, and chat history.",
    icon_key="twitch",
)

PROVIDER_REGISTRY: dict[str, ConnectedAccountProvider] = {
    provider.name: provider
    for provider in (
        GMAIL_PROVIDER,
        YOUTUBE_PROVIDER,
        TWITTER_PROVIDER,
        INSTAGRAM_PROVIDER,
        TWITCH_PROVIDER,
    )
}


def get_provider(name: str) -> ConnectedAccountProvider | None:
    """Look up a provider by name, case-insensitively.

    The name reaches this function from an endpoint argument or from the model
    echoing back something a human typed, so "Gmail" and "gmail" both resolve.
    """
    return PROVIDER_REGISTRY.get(str(name or "").strip().lower())


def mailbox_providers() -> tuple[ConnectedAccountProvider, ...]:
    """Return the providers that expose an email mailbox."""
    return tuple(
        provider
        for provider in PROVIDER_REGISTRY.values()
        if provider.kind == KIND_MAILBOX
    )


def social_providers() -> tuple[ConnectedAccountProvider, ...]:
    """Return only the providers that evidence a public social identity.

    SECURITY: this is the allow-list an avatar-sharing likeness check must
    consult. Sharing an avatar of one's own likeness is gated on having verified
    the account behind that likeness, and only a genuine social identity can
    discharge that requirement. A connected mailbox or data source must never
    satisfy it — owning an email address proves nothing about who a likeness
    depicts. Filter on this function rather than on "the user has any connected
    account", which would open exactly that hole.
    """
    return tuple(
        provider
        for provider in PROVIDER_REGISTRY.values()
        if provider.kind == KIND_SOCIAL
    )


def validate_registry() -> None:
    """Assert every row uses a known kind and mechanism.

    Called from the package import so a malformed row fails loudly at startup
    rather than at the moment a user tries to connect that provider.
    """
    for provider in PROVIDER_REGISTRY.values():
        if provider.kind not in ALL_KINDS:
            raise ValueError(
                f"Connected-account provider {provider.name!r} declares unknown "
                f"kind {provider.kind!r}; expected one of {sorted(ALL_KINDS)}."
            )
        if provider.credential_mechanism not in ALL_MECHANISMS:
            raise ValueError(
                f"Connected-account provider {provider.name!r} declares unknown "
                f"credential mechanism {provider.credential_mechanism!r}; "
                f"expected one of {sorted(ALL_MECHANISMS)}."
            )
        if provider.kind == KIND_MAILBOX and not provider.imap_host:
            raise ValueError(
                f"Mailbox provider {provider.name!r} must declare an imap_host."
            )
        # A form-based provider with no fields renders an empty connect card the
        # owner cannot complete, which fails at the one moment the feature is
        # supposed to work. Catch it at import instead.
        if (
            provider.credential_mechanism == MECHANISM_APP_PASSWORD
            and not provider.connect_fields
        ):
            raise ValueError(
                f"Provider {provider.name!r} collects a credential in a form and "
                "must declare connect_fields."
            )


validate_registry()
