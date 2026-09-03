"""The registry of external accounts a personal avatar can be connected to.

One row per provider. This table is THE extension point for account
connections: adding a provider whose ``kind`` already has a tool factory (see
``tool_factories.py``) is one row here plus one icon in the frontend, and adding
a provider with a brand-new kind is one row plus one factory module. Nothing
else — no new route, no new store, no new component — because every route
dispatches on the fields declared here rather than on the provider's name.

Fields that carry design decisions:

``kind`` is a SECURITY BOUNDARY, not a taxonomy
    ``social_providers()`` is what an identity-verification gate must read when
    deciding whether the human behind a likeness has been verified. If a mailbox
    were filed as ``"social"``, connecting an email account would satisfy "this
    person proved they own the account behind this likeness" — which it plainly
    does not, since anyone can own an email address. Gmail is therefore
    ``"mailbox"``. A connected machine or a custom Model Context Protocol server
    is ``"data_source"`` / ``"mcp_server"`` for the same reason. Only accounts
    that actually evidence a public identity may be ``"social"``.

``credential_mechanism`` decides how a connection is established
    ``"app_password"`` collects a credential in a form and verifies it by
    logging in. ``"mcp_url"`` collects a server address (and an optional bearer
    token) and verifies it by listing the server's tools. ``"auth0_identity"``
    links a secondary identity onto the account. ``"oauth"`` runs an
    authorization-code redirect and stores a refresh token.
    ``"device_pairing"`` is not a form at all: the connection is made by the
    Neural Nexus daemon registering itself, so the card carries instructions.
    The connect endpoint dispatches on this field (``connect_handlers.py``), so
    a provider declares its flow instead of the flow being hard-coded per name.

``availability`` keeps the catalog honest
    Every provider the product intends to support appears in the catalog so the
    owner sees the full set, but only ``"available"`` providers can be
    connected. A ``"coming_soon"`` row renders with a disabled action and its
    connect attempt is refused with a plain message rather than a broken form.

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
# membership check in `validate_registry` rather than silently creating a kind
# that no gate knows about.
KIND_MAILBOX = "mailbox"
KIND_SOCIAL = "social"
KIND_DATA_SOURCE = "data_source"
KIND_CALENDAR = "calendar"
KIND_MESSAGING = "messaging"
KIND_MCP_SERVER = "mcp_server"
ALL_KINDS = frozenset(
    {
        KIND_MAILBOX,
        KIND_SOCIAL,
        KIND_DATA_SOURCE,
        KIND_CALENDAR,
        KIND_MESSAGING,
        KIND_MCP_SERVER,
    }
)

# Credential mechanisms.
MECHANISM_APP_PASSWORD = "app_password"
MECHANISM_AUTH0_IDENTITY = "auth0_identity"
MECHANISM_OAUTH = "oauth"
MECHANISM_MCP_URL = "mcp_url"
MECHANISM_DEVICE_PAIRING = "device_pairing"
ALL_MECHANISMS = frozenset(
    {
        MECHANISM_APP_PASSWORD,
        MECHANISM_AUTH0_IDENTITY,
        MECHANISM_OAUTH,
        MECHANISM_MCP_URL,
        MECHANISM_DEVICE_PAIRING,
    }
)

# Mechanisms whose connect flow is a form the owner completes. Everything else
# is either a redirect (not yet implemented) or a daemon-side registration.
FORM_MECHANISMS = frozenset({MECHANISM_APP_PASSWORD, MECHANISM_MCP_URL})

# Catalog categories, in the order the manage-connections screen groups them.
CATEGORY_MAIL = "mail"
CATEGORY_CALENDAR = "calendar"
CATEGORY_SOCIAL = "social"
CATEGORY_MESSAGING = "messaging"
CATEGORY_DEVICE = "device"
CATEGORY_CUSTOM = "custom"
CATEGORY_ORDER: tuple[str, ...] = (
    CATEGORY_MAIL,
    CATEGORY_CALENDAR,
    CATEGORY_SOCIAL,
    CATEGORY_MESSAGING,
    CATEGORY_DEVICE,
    CATEGORY_CUSTOM,
)
ALL_CATEGORIES = frozenset(CATEGORY_ORDER)

# Availability.
AVAILABILITY_AVAILABLE = "available"
AVAILABILITY_COMING_SOON = "coming_soon"
ALL_AVAILABILITIES = frozenset({AVAILABILITY_AVAILABLE, AVAILABILITY_COMING_SOON})

# The generic connect endpoint every form-mechanism provider posts to. Named
# here, once, so the card payload and the route agree.
CONNECT_ACCOUNT_ENDPOINT = "/connect_account"

# Where the owner obtains the Neural Nexus daemon for a machine. The device
# provider's card points here because a device is connected by installing
# software rather than by filling in a form.
DAEMON_INSTALL_URL = "https://github.com/AfterlifeSystems/anubis-mcp-server-ubuntu"

COMING_SOON_MESSAGE = (
    "{display_name} connections are coming soon. The connector is listed so you "
    "can see what the avatar will be able to reach; it cannot be connected yet."
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
            connect handler reads.
        label: Field label shown above the input.
        input_type: HTML input type. ``"password"`` for anything secret, so the
            frontend masks it without having to know which field is the secret.
        placeholder: Placeholder text shown in the empty input.
        help_text: Explanation rendered beneath the input.
        required: Whether the handler refuses a connection without this field.
    """

    name: str
    label: str
    input_type: str = "text"
    placeholder: str = ""
    help_text: str = ""
    required: bool = True


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
        category: One of :data:`ALL_CATEGORIES`; groups the catalog screen.
        summary: One line for the catalog row ("Search and draft emails").
        featured: Whether the row appears in the Featured section.
        availability: One of :data:`ALL_AVAILABILITIES`.
        connect_endpoint: The route a form-mechanism card posts its fields to.
        imap_host: IMAP server, for ``app_password`` providers only.
        imap_port: IMAP TLS port, for ``app_password`` providers only.
        smtp_host: SMTP submission server, for ``app_password`` providers only.
        smtp_port: SMTP submission port, for ``app_password`` providers only.
        drafts_mailbox: IMAP folder that holds drafts. Gmail exposes this as
            ``"[Gmail]/Drafts"`` rather than the ``"Drafts"`` most other servers
            use, which is exactly the sort of per-provider detail this table
            exists to hold.
        sent_mailbox: IMAP folder that holds the owner's sent messages — the
            owner's own writing, read when matching the owner's voice.
        send_supported: Whether the provider's tools may transmit a message.
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
            Empty for providers whose mechanism is a redirect or a pairing.
        pairing_instructions: For ``device_pairing`` providers, the text the
            card shows instead of a form.
        install_url: For ``device_pairing`` providers, where the daemon lives.
    """

    name: str
    kind: str
    credential_mechanism: str
    display_name: str
    category: str = CATEGORY_CUSTOM
    summary: str = ""
    featured: bool = True
    availability: str = AVAILABILITY_AVAILABLE
    connect_endpoint: str = CONNECT_ACCOUNT_ENDPOINT
    imap_host: str | None = None
    imap_port: int = 993
    smtp_host: str | None = None
    smtp_port: int = 587
    drafts_mailbox: str = "Drafts"
    sent_mailbox: str | None = None
    send_supported: bool = False
    credential_help_url: str | None = None
    auth0_connection: str | None = field(default=None)
    card_description: str = ""
    icon_key: str = ""
    connect_fields: tuple[ConnectFieldSpec, ...] = ()
    pairing_instructions: str = ""
    install_url: str | None = None

    @property
    def is_mailbox(self) -> bool:
        """Whether this provider exposes an email mailbox."""
        return self.kind == KIND_MAILBOX

    @property
    def is_available(self) -> bool:
        """Whether the provider can be connected today."""
        return self.availability == AVAILABILITY_AVAILABLE

    @property
    def uses_form(self) -> bool:
        """Whether the connect flow is a form the owner completes on the card."""
        return self.credential_mechanism in FORM_MECHANISMS

    def coming_soon_message(self) -> str:
        """Return the refusal an unavailable provider's connect attempt gets."""
        return COMING_SOON_MESSAGE.format(display_name=self.display_name)


GMAIL_PROVIDER = ConnectedAccountProvider(
    name="gmail",
    kind=KIND_MAILBOX,
    credential_mechanism=MECHANISM_APP_PASSWORD,
    display_name="Gmail",
    category=CATEGORY_MAIL,
    summary="Search, read, draft, and send email",
    imap_host="imap.gmail.com",
    imap_port=993,
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    drafts_mailbox="[Gmail]/Drafts",
    sent_mailbox="[Gmail]/Sent Mail",
    send_supported=True,
    credential_help_url="https://myaccount.google.com/apppasswords",
    card_description="Search, read, draft, and send email.",
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

CUSTOM_MCP_PROVIDER = ConnectedAccountProvider(
    name="custom_mcp",
    kind=KIND_MCP_SERVER,
    credential_mechanism=MECHANISM_MCP_URL,
    display_name="Custom connector",
    category=CATEGORY_CUSTOM,
    summary="Add your own Model Context Protocol server",
    featured=False,
    card_description=(
        "Give the avatar the tools of any Model Context Protocol server you run."
    ),
    icon_key="custom",
    connect_fields=(
        ConnectFieldSpec(
            name="name",
            label="Name",
            placeholder="My Connector",
            help_text="How the avatar refers to this connector in conversation.",
        ),
        ConnectFieldSpec(
            name="server_url",
            label="Server URL",
            input_type="url",
            placeholder="https://mcp.example.com/sse",
            help_text=(
                "The server's Streamable HTTP or SSE endpoint. The connector is "
                "verified by listing the server's tools before it is saved."
            ),
        ),
        ConnectFieldSpec(
            name="bearer_token",
            label="Access token",
            input_type="password",
            placeholder="Optional bearer token",
            help_text=(
                "Sent as an Authorization header when the server requires one. "
                "Stored encrypted; never shown again."
            ),
            required=False,
        ),
    ),
)

DESKTOP_MCP_PROVIDER = ConnectedAccountProvider(
    name="desktop_mcp",
    kind=KIND_DATA_SOURCE,
    credential_mechanism=MECHANISM_DEVICE_PAIRING,
    display_name="Your machines",
    category=CATEGORY_DEVICE,
    summary="Ubuntu, macOS, Windows, and mobile devices running Neural Nexus",
    card_description=(
        "Let the avatar read and analyze the files a machine of yours shares."
    ),
    icon_key="mcp",
    pairing_instructions=(
        "Install the Neural Nexus daemon on the machine, sign in with your API "
        "key, and choose a folder to share. The machine appears here on its own "
        "and the avatar connects to the machine automatically."
    ),
    install_url=DAEMON_INSTALL_URL,
)

GOOGLE_CALENDAR_PROVIDER = ConnectedAccountProvider(
    name="google_calendar",
    kind=KIND_CALENDAR,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="Google Calendar",
    category=CATEGORY_CALENDAR,
    summary="Check your schedule and book meetings",
    availability=AVAILABILITY_COMING_SOON,
    card_description="Read your calendar and schedule meetings on your behalf.",
    icon_key="google_calendar",
)

# The social providers named in _SOCIAL_MEDIA_ACCOUNT_CONNECTION.md. Declared,
# with no connect flow implemented yet, for two reasons: the registry's shape is
# only proven by holding more than one kind of row, and `social_providers()`
# needs real members so the verification gate that reads it is written against
# actual data rather than an empty tuple. They are marked coming soon, so the
# catalog shows them with a disabled action and the connect endpoint refuses
# them plainly until the identity-linking flow lands.
YOUTUBE_PROVIDER = ConnectedAccountProvider(
    name="youtube",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="YouTube",
    category=CATEGORY_SOCIAL,
    summary="Your channel's videos, descriptions, and comments",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="google-oauth2",
    card_description="Read your channel's videos, descriptions, and comments.",
    icon_key="youtube",
)

TWITTER_PROVIDER = ConnectedAccountProvider(
    name="twitter",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="X",
    category=CATEGORY_SOCIAL,
    summary="Your posts, timeline, and bookmarks",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="twitter",
    card_description="Read your posts, timeline, and bookmarks; post as you.",
    icon_key="twitter",
)

INSTAGRAM_PROVIDER = ConnectedAccountProvider(
    name="instagram",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="Instagram",
    category=CATEGORY_SOCIAL,
    summary="Your posts, captions, and comments",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="instagram",
    card_description="Read your posts, captions, and comments.",
    icon_key="instagram",
)

TWITCH_PROVIDER = ConnectedAccountProvider(
    name="twitch",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="Twitch",
    category=CATEGORY_SOCIAL,
    summary="Your channel, streams, and chat history",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="twitch",
    card_description="Read your channel, streams, and chat history.",
    icon_key="twitch",
)

FACEBOOK_PROVIDER = ConnectedAccountProvider(
    name="facebook",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="Facebook",
    category=CATEGORY_SOCIAL,
    summary="Your posts and pages",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="facebook",
    card_description="Read your posts and pages.",
    icon_key="facebook",
)

LINKEDIN_PROVIDER = ConnectedAccountProvider(
    name="linkedin",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_AUTH0_IDENTITY,
    display_name="LinkedIn",
    category=CATEGORY_SOCIAL,
    summary="Your profile, posts, and messages",
    availability=AVAILABILITY_COMING_SOON,
    auth0_connection="linkedin",
    card_description="Read your profile, posts, and messages.",
    icon_key="linkedin",
)

DISCORD_PROVIDER = ConnectedAccountProvider(
    name="discord",
    kind=KIND_MESSAGING,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="Discord",
    category=CATEGORY_MESSAGING,
    summary="Your servers and direct messages",
    availability=AVAILABILITY_COMING_SOON,
    card_description="Read and reply in your servers and direct messages.",
    icon_key="discord",
)

SLACK_PROVIDER = ConnectedAccountProvider(
    name="slack",
    kind=KIND_MESSAGING,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="Slack",
    category=CATEGORY_MESSAGING,
    summary="Your workspaces and channels",
    availability=AVAILABILITY_COMING_SOON,
    card_description="Read and reply in your workspaces and channels.",
    icon_key="slack",
)

PROVIDER_REGISTRY: dict[str, ConnectedAccountProvider] = {
    provider.name: provider
    for provider in (
        GMAIL_PROVIDER,
        GOOGLE_CALENDAR_PROVIDER,
        YOUTUBE_PROVIDER,
        TWITTER_PROVIDER,
        INSTAGRAM_PROVIDER,
        TWITCH_PROVIDER,
        FACEBOOK_PROVIDER,
        LINKEDIN_PROVIDER,
        DISCORD_PROVIDER,
        SLACK_PROVIDER,
        DESKTOP_MCP_PROVIDER,
        CUSTOM_MCP_PROVIDER,
    )
}


def get_provider(name: str) -> ConnectedAccountProvider | None:
    """Look up a provider by name, case-insensitively.

    The name reaches this function from an endpoint argument or from the model
    echoing back something a human typed, so "Gmail" and "gmail" both resolve.
    """
    return PROVIDER_REGISTRY.get(str(name or "").strip().lower())


def catalog_providers() -> tuple[ConnectedAccountProvider, ...]:
    """Every provider in catalog order: featured first, then by category.

    The manage-connections screen and the New Connector picker both render this
    order, so the two surfaces agree on where a provider sits.
    """
    return tuple(
        sorted(
            PROVIDER_REGISTRY.values(),
            key=lambda provider: (
                0 if provider.featured else 1,
                CATEGORY_ORDER.index(provider.category),
                provider.display_name.lower(),
            ),
        )
    )


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
    discharge that requirement. A connected mailbox, machine, or data source
    must never satisfy it — owning an email address proves nothing about who a
    likeness depicts. Filter on this function rather than on "the user has any
    connected account", which would open exactly that hole.
    """
    return tuple(
        provider
        for provider in PROVIDER_REGISTRY.values()
        if provider.kind == KIND_SOCIAL
    )


def validate_registry() -> None:
    """Assert every row is internally consistent.

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
        if provider.category not in ALL_CATEGORIES:
            raise ValueError(
                f"Connected-account provider {provider.name!r} declares unknown "
                f"category {provider.category!r}; expected one of "
                f"{sorted(ALL_CATEGORIES)}."
            )
        if provider.availability not in ALL_AVAILABILITIES:
            raise ValueError(
                f"Connected-account provider {provider.name!r} declares unknown "
                f"availability {provider.availability!r}; expected one of "
                f"{sorted(ALL_AVAILABILITIES)}."
            )
        if provider.kind == KIND_MAILBOX and not provider.imap_host:
            raise ValueError(
                f"Mailbox provider {provider.name!r} must declare an imap_host."
            )
        if provider.send_supported and not provider.smtp_host:
            raise ValueError(
                f"Provider {provider.name!r} supports sending but declares no "
                "smtp_host."
            )
        # A form-based provider with no fields renders an empty connect card the
        # owner cannot complete, which fails at the one moment the feature is
        # supposed to work. Catch it at import instead.
        if provider.uses_form and not provider.connect_fields:
            raise ValueError(
                f"Provider {provider.name!r} collects its connection in a form "
                "and must declare connect_fields."
            )
        if provider.credential_mechanism == MECHANISM_MCP_URL and not any(
            field_spec.name == "server_url" for field_spec in provider.connect_fields
        ):
            raise ValueError(
                f"Provider {provider.name!r} connects by server address and must "
                "declare a 'server_url' field."
            )
        if (
            provider.credential_mechanism == MECHANISM_DEVICE_PAIRING
            and not provider.pairing_instructions
        ):
            raise ValueError(
                f"Provider {provider.name!r} is connected by pairing a device and "
                "must declare pairing_instructions."
            )


validate_registry()
