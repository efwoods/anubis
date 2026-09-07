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

How an owner signs in (``login_mode``)
    The owner never types a credential into Neural Nexus. The connect card's
    button opens the vendor's OWN sign-in page in a popup: Google's consent
    screen for Gmail, Calendar, Analytics, and YouTube; GitHub's, X's, and
    Vercel's authorization pages; Plaid Link for a bank; and, for any site
    with no OAuth at all, a live browser the API hosts, where the owner signs
    in on the site's real login page and the signed-in session is kept. The
    ``login_mode`` property derives the popup kind from the mechanism so the
    card, the routes, and the tools agree.

Gmail through Google sign-in (an accepted trade-off)
    Reading a mailbox needs the restricted scope ``https://mail.google.com/``.
    Until the OAuth client passes Google verification and the CASA assessment,
    the consent screen stays in Testing status: only listed test users may sign
    in and refresh tokens expire after seven days. An expired token is surfaced
    as a ``needs_reconnect`` status and the card is raised again; nothing
    silently breaks. App passwords are deliberately NOT offered: the owner asked
    for the official Google login and nothing else, and since 2025-03-14 Google
    accepts only OAuth 2.0 or an app password over IMAP anyway. Existing
    app-password records keep working through ``mailbox_credentials_for``.
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
KIND_BANK = "bank"
KIND_DEVELOPER = "developer"
KIND_WEBSITE = "website"
KIND_ANALYTICS = "analytics"
KIND_HOSTING = "hosting"
ALL_KINDS = frozenset(
    {
        KIND_MAILBOX,
        KIND_SOCIAL,
        KIND_DATA_SOURCE,
        KIND_CALENDAR,
        KIND_MESSAGING,
        KIND_MCP_SERVER,
        KIND_BANK,
        KIND_DEVELOPER,
        KIND_WEBSITE,
        KIND_ANALYTICS,
        KIND_HOSTING,
    }
)

# Credential mechanisms.
MECHANISM_APP_PASSWORD = "app_password"
MECHANISM_AUTH0_IDENTITY = "auth0_identity"
MECHANISM_OAUTH = "oauth"
MECHANISM_MCP_URL = "mcp_url"
MECHANISM_DEVICE_PAIRING = "device_pairing"
MECHANISM_PLAID_LINK = "plaid_link"
MECHANISM_BROWSER_SESSION = "browser_session"
MECHANISM_URL_ONLY = "url_only"
ALL_MECHANISMS = frozenset(
    {
        MECHANISM_APP_PASSWORD,
        MECHANISM_AUTH0_IDENTITY,
        MECHANISM_OAUTH,
        MECHANISM_MCP_URL,
        MECHANISM_DEVICE_PAIRING,
        MECHANISM_PLAID_LINK,
        MECHANISM_BROWSER_SESSION,
        MECHANISM_URL_ONLY,
    }
)

# Mechanisms whose connect flow is a form the owner completes on the card.
# ``url_only`` is a form too (a website address, no credential).
FORM_MECHANISMS = frozenset(
    {MECHANISM_APP_PASSWORD, MECHANISM_MCP_URL, MECHANISM_URL_ONLY}
)

# How the card signs the owner in. Derived from the mechanism so every surface
# (the in-chat card, the "+" menu, the settings picker) opens the same thing.
LOGIN_MODE_FORM = "form"
LOGIN_MODE_OAUTH_POPUP = "oauth_popup"
LOGIN_MODE_PLAID_LINK = "plaid_link"
LOGIN_MODE_BROWSER_SESSION = "browser_session"
LOGIN_MODE_NONE = "none"
LOGIN_MODES_BY_MECHANISM: dict[str, str] = {
    MECHANISM_APP_PASSWORD: LOGIN_MODE_FORM,
    MECHANISM_MCP_URL: LOGIN_MODE_FORM,
    MECHANISM_URL_ONLY: LOGIN_MODE_FORM,
    MECHANISM_OAUTH: LOGIN_MODE_OAUTH_POPUP,
    MECHANISM_PLAID_LINK: LOGIN_MODE_PLAID_LINK,
    MECHANISM_BROWSER_SESSION: LOGIN_MODE_BROWSER_SESSION,
    MECHANISM_AUTH0_IDENTITY: LOGIN_MODE_NONE,
    MECHANISM_DEVICE_PAIRING: LOGIN_MODE_NONE,
}

# The login endpoints each popup mode starts from. Named once so the card
# payload and the routes agree.
OAUTH_LOGIN_ENDPOINT = "/connect_account/oauth/start"
PLAID_LOGIN_ENDPOINT = "/connect_account/plaid/link_token"
BROWSER_LOGIN_ENDPOINT = "/connect_account/browser/start"
LOGIN_ENDPOINTS_BY_MODE: dict[str, str] = {
    LOGIN_MODE_OAUTH_POPUP: OAUTH_LOGIN_ENDPOINT,
    LOGIN_MODE_PLAID_LINK: PLAID_LOGIN_ENDPOINT,
    LOGIN_MODE_BROWSER_SESSION: BROWSER_LOGIN_ENDPOINT,
}

# Catalog categories, in the order the manage-connections screen groups them.
CATEGORY_MAIL = "mail"
CATEGORY_CALENDAR = "calendar"
CATEGORY_SOCIAL = "social"
CATEGORY_MESSAGING = "messaging"
CATEGORY_DEVICE = "device"
CATEGORY_CUSTOM = "custom"
CATEGORY_FINANCE = "finance"
CATEGORY_DEVELOPMENT = "development"
CATEGORY_VENDOR = "vendor"
CATEGORY_WEB = "web"
CATEGORY_ANALYTICS = "analytics"
CATEGORY_HOSTING = "hosting"
CATEGORY_ORDER: tuple[str, ...] = (
    CATEGORY_MAIL,
    CATEGORY_FINANCE,
    CATEGORY_DEVELOPMENT,
    CATEGORY_VENDOR,
    CATEGORY_ANALYTICS,
    CATEGORY_WEB,
    CATEGORY_HOSTING,
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
    # Popup sign-in details. ``oauth_config_key`` names a row of
    # ``oauth_providers.OAUTH_PROVIDERS``; ``oauth_scopes`` narrows that vendor's
    # scopes to what this provider needs. ``login_url`` is the page a live
    # browser sign-in opens; ``home_url`` is what the keepalive revisits and
    # where connected-site tools start; ``recipe_key`` names the vendor recipe
    # (``recipes.py``) that knows the site's usage pages.
    oauth_config_key: str | None = None
    oauth_scopes: tuple[str, ...] = ()
    login_url: str | None = None
    home_url: str | None = None
    recipe_key: str | None = None
    # A device-bound provider is connected through a machine running the
    # daemon rather than through a credential of its own.
    device_bound: bool = False

    @property
    def login_mode(self) -> str:
        """How the card signs the owner in (see ``LOGIN_MODES_BY_MECHANISM``)."""
        return LOGIN_MODES_BY_MECHANISM.get(self.credential_mechanism, LOGIN_MODE_NONE)

    @property
    def login_endpoint(self) -> str | None:
        """The route a popup sign-in starts from, or ``None`` for forms/devices."""
        return LOGIN_ENDPOINTS_BY_MODE.get(self.login_mode)

    @property
    def uses_popup(self) -> bool:
        """Whether the card's button opens a sign-in window."""
        return self.login_mode in LOGIN_ENDPOINTS_BY_MODE

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
    credential_mechanism=MECHANISM_OAUTH,
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
    credential_help_url="https://myaccount.google.com/permissions",
    card_description="Search, read, draft, and send email. Sign in with Google.",
    icon_key="gmail",
    oauth_config_key="google",
    oauth_scopes=("openid", "email", "https://mail.google.com/"),
    login_url="https://accounts.google.com/ServiceLogin?continue=https://mail.google.com/mail/",
    home_url="https://mail.google.com/mail/",
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
    summary="Check your schedule",
    card_description="Read your calendar so the avatar knows your schedule.",
    icon_key="google_calendar",
    login_url="https://accounts.google.com/ServiceLogin?continue=https://calendar.google.com/",
    home_url="https://calendar.google.com/",
    oauth_config_key="google",
    oauth_scopes=(
        "openid",
        "email",
        "https://www.googleapis.com/auth/calendar.readonly",
    ),
)

GOOGLE_ANALYTICS_PROVIDER = ConnectedAccountProvider(
    name="google_analytics",
    kind=KIND_ANALYTICS,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="Google Analytics",
    category=CATEGORY_ANALYTICS,
    summary="Visitors, sessions, and pages for your websites",
    card_description="Read traffic reports for the websites you own.",
    icon_key="google_analytics",
    login_url="https://accounts.google.com/ServiceLogin?continue=https://analytics.google.com/",
    home_url="https://analytics.google.com/",
    oauth_config_key="google",
    oauth_scopes=(
        "openid",
        "email",
        "https://www.googleapis.com/auth/analytics.readonly",
    ),
)

YOUTUBE_PROVIDER = ConnectedAccountProvider(
    name="youtube",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="YouTube",
    category=CATEGORY_SOCIAL,
    summary="Your channel's videos, statistics, and comments",
    card_description="Read your channel's videos, statistics, and comments.",
    icon_key="youtube",
    login_url="https://accounts.google.com/ServiceLogin?continue=https://studio.youtube.com/",
    home_url="https://studio.youtube.com/",
    oauth_config_key="google",
    oauth_scopes=(
        "openid",
        "email",
        "https://www.googleapis.com/auth/youtube.readonly",
    ),
)

GITHUB_PROVIDER = ConnectedAccountProvider(
    name="github",
    kind=KIND_DEVELOPER,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="GitHub",
    category=CATEGORY_DEVELOPMENT,
    summary="Commits, pull requests, and issues",
    card_description=(
        "Read your repositories' commits, pull requests, and issues to report on "
        "development."
    ),
    icon_key="github",
    oauth_config_key="github",
    login_url="https://github.com/login",
    home_url="https://github.com/notifications",
)

X_PROVIDER = ConnectedAccountProvider(
    name="twitter",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="X",
    category=CATEGORY_SOCIAL,
    summary="Your posts and replies",
    send_supported=True,
    card_description="Read your posts and post replies as you.",
    icon_key="twitter",
    oauth_config_key="x",
    login_url="https://x.com/i/flow/login",
    home_url="https://x.com/notifications",
)

VERCEL_PROVIDER = ConnectedAccountProvider(
    name="vercel",
    kind=KIND_HOSTING,
    credential_mechanism=MECHANISM_OAUTH,
    display_name="Vercel",
    category=CATEGORY_HOSTING,
    summary="Deployments and usage of your projects",
    card_description="Read your projects' deployments and usage.",
    icon_key="vercel",
    oauth_config_key="vercel",
    login_url="https://vercel.com/login",
    home_url="https://vercel.com/dashboard",
)

PLAID_PROVIDER = ConnectedAccountProvider(
    name="plaid",
    kind=KIND_BANK,
    credential_mechanism=MECHANISM_PLAID_LINK,
    display_name="Finance",
    category=CATEGORY_FINANCE,
    summary="Bank and card accounts through Plaid",
    card_description=(
        "Connect a bank or card so the avatar can report spending, burn rate, "
        "and cost per customer."
    ),
    icon_key="bank",
)

LANGSMITH_PROVIDER = ConnectedAccountProvider(
    name="langsmith",
    kind=KIND_ANALYTICS,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="LangSmith",
    category=CATEGORY_VENDOR,
    summary="Traces, runs, and usage of your LangSmith organization",
    card_description="Sign in to LangSmith so the avatar can read usage and cost.",
    icon_key="langsmith",
    login_url="https://smith.langchain.com/",
    home_url="https://smith.langchain.com/",
    recipe_key="langsmith",
)

OPENAI_PROVIDER = ConnectedAccountProvider(
    name="openai",
    kind=KIND_ANALYTICS,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="OpenAI",
    category=CATEGORY_VENDOR,
    summary="Usage and costs of your OpenAI organization",
    card_description="Sign in to the OpenAI platform so the avatar can read usage.",
    icon_key="openai",
    login_url="https://platform.openai.com/login",
    home_url="https://platform.openai.com/usage",
    recipe_key="openai",
)

ANTHROPIC_PROVIDER = ConnectedAccountProvider(
    name="anthropic",
    kind=KIND_ANALYTICS,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Claude (Anthropic)",
    category=CATEGORY_VENDOR,
    summary="Usage and costs of your Anthropic console",
    card_description="Sign in to the Anthropic console so the avatar can read usage.",
    icon_key="anthropic",
    login_url="https://console.anthropic.com/login",
    home_url="https://console.anthropic.com/settings/usage",
    recipe_key="anthropic",
)

WEBSITE_PROVIDER = ConnectedAccountProvider(
    name="website",
    kind=KIND_WEBSITE,
    credential_mechanism=MECHANISM_URL_ONLY,
    display_name="Website",
    category=CATEGORY_WEB,
    summary="Crawl, audit, and report on a website",
    card_description=(
        "Add a website by address. The avatar crawls the site and reports on "
        "content, search visibility, links, accessibility, and changes."
    ),
    icon_key="website",
    connect_fields=(
        ConnectFieldSpec(
            name="name",
            label="Name",
            placeholder="My website",
            help_text="How the avatar refers to this site in conversation.",
            required=False,
        ),
        ConnectFieldSpec(
            name="site_url",
            label="Website address",
            input_type="url",
            placeholder="https://example.com",
            help_text="The site's home page. The address is checked before it is saved.",
        ),
    ),
)

CUSTOM_SITE_PROVIDER = ConnectedAccountProvider(
    name="custom_site",
    kind=KIND_ANALYTICS,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Custom site",
    category=CATEGORY_CUSTOM,
    summary="Sign in to any website and let the avatar use your account",
    featured=False,
    card_description=(
        "Sign in to any website on its own login page. The avatar keeps the "
        "signed-in session and can read and act on your account there."
    ),
    icon_key="url",
    connect_fields=(
        ConnectFieldSpec(
            name="name",
            label="Name",
            placeholder="My dashboard",
            help_text="How the avatar refers to this site in conversation.",
        ),
        ConnectFieldSpec(
            name="site_url",
            label="Sign-in page address",
            input_type="url",
            placeholder="https://example.com/login",
            help_text="The page where you sign in. Opens in a window for you to sign in on.",
        ),
    ),
)

CLAUDE_CODE_PROVIDER = ConnectedAccountProvider(
    name="claude_code",
    kind=KIND_DEVELOPER,
    credential_mechanism=MECHANISM_DEVICE_PAIRING,
    display_name="Claude Code",
    category=CATEGORY_DEVELOPMENT,
    summary="Coding sessions and repositories on your machines",
    card_description=(
        "Read the Claude Code sessions and git repositories on a machine running "
        "Neural Nexus, to report on what was built and how long it took."
    ),
    icon_key="claude_code",
    pairing_instructions=(
        "Claude Code sessions and repositories are read through a machine running "
        "the Neural Nexus daemon. Install the daemon on the machine where you code, "
        "and it appears here on its own."
    ),
    install_url=DAEMON_INSTALL_URL,
    device_bound=True,
)

# Social and messaging accounts without an official app yet: the owner signs in
# on the site's own page in the live browser, and the avatar reads through the
# signed-in session. When a developer app is created, the row switches to OAuth.
INSTAGRAM_PROVIDER = ConnectedAccountProvider(
    name="instagram",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Instagram",
    category=CATEGORY_SOCIAL,
    summary="Your posts, captions, and comments",
    card_description="Sign in to Instagram so the avatar can read your posts.",
    icon_key="instagram",
    login_url="https://www.instagram.com/accounts/login/",
    home_url="https://www.instagram.com/",
    recipe_key="instagram",
)

TWITCH_PROVIDER = ConnectedAccountProvider(
    name="twitch",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Twitch",
    category=CATEGORY_SOCIAL,
    summary="Your channel, streams, and chat history",
    card_description="Sign in to Twitch so the avatar can read your channel.",
    icon_key="twitch",
    login_url="https://www.twitch.tv/login",
    home_url="https://www.twitch.tv/",
    recipe_key="twitch",
)

FACEBOOK_PROVIDER = ConnectedAccountProvider(
    name="facebook",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Facebook",
    category=CATEGORY_SOCIAL,
    summary="Your posts and pages",
    card_description="Sign in to Facebook so the avatar can read your posts and pages.",
    icon_key="facebook",
    login_url="https://www.facebook.com/login/",
    home_url="https://www.facebook.com/",
    recipe_key="facebook",
)

LINKEDIN_PROVIDER = ConnectedAccountProvider(
    name="linkedin",
    kind=KIND_SOCIAL,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="LinkedIn",
    category=CATEGORY_SOCIAL,
    summary="Your profile, posts, and messages",
    card_description="Sign in to LinkedIn so the avatar can read your profile and posts.",
    icon_key="linkedin",
    login_url="https://www.linkedin.com/login",
    home_url="https://www.linkedin.com/feed/",
    recipe_key="linkedin",
)

DISCORD_PROVIDER = ConnectedAccountProvider(
    name="discord",
    kind=KIND_MESSAGING,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Discord",
    category=CATEGORY_MESSAGING,
    summary="Your servers and direct messages",
    card_description="Sign in to Discord so the avatar can read your servers.",
    icon_key="discord",
    login_url="https://discord.com/login",
    home_url="https://discord.com/channels/@me",
    recipe_key="discord",
)

SLACK_PROVIDER = ConnectedAccountProvider(
    name="slack",
    kind=KIND_MESSAGING,
    credential_mechanism=MECHANISM_BROWSER_SESSION,
    display_name="Slack",
    category=CATEGORY_MESSAGING,
    summary="Your workspaces and channels",
    card_description="Sign in to Slack so the avatar can read your workspaces.",
    icon_key="slack",
    login_url="https://slack.com/signin",
    home_url="https://app.slack.com/",
    recipe_key="slack",
)

# The row keeps the historical name "twitter" (icons, welcome page, stored
# records) while presenting as X; ``get_provider`` accepts "x" as an alias.
TWITTER_PROVIDER = X_PROVIDER

# The pre-OAuth Gmail row, kept UNREGISTERED: records connected with an app
# password before Google sign-in landed still carry ``credential_mechanism
# "app_password"`` and keep working through ``mailbox_credentials_for``; tests of
# the app-password path register this row under the gmail name. No catalog
# surface offers an app password to an owner.
GMAIL_APP_PASSWORD_PROVIDER = ConnectedAccountProvider(
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
            help_text="A 16-character Google app password (legacy path).",
        ),
    ),
)

PROVIDER_REGISTRY: dict[str, ConnectedAccountProvider] = {
    provider.name: provider
    for provider in (
        GMAIL_PROVIDER,
        PLAID_PROVIDER,
        GITHUB_PROVIDER,
        CLAUDE_CODE_PROVIDER,
        LANGSMITH_PROVIDER,
        OPENAI_PROVIDER,
        ANTHROPIC_PROVIDER,
        GOOGLE_ANALYTICS_PROVIDER,
        WEBSITE_PROVIDER,
        VERCEL_PROVIDER,
        GOOGLE_CALENDAR_PROVIDER,
        YOUTUBE_PROVIDER,
        X_PROVIDER,
        INSTAGRAM_PROVIDER,
        TWITCH_PROVIDER,
        FACEBOOK_PROVIDER,
        LINKEDIN_PROVIDER,
        DISCORD_PROVIDER,
        SLACK_PROVIDER,
        DESKTOP_MCP_PROVIDER,
        CUSTOM_MCP_PROVIDER,
        CUSTOM_SITE_PROVIDER,
    )
}


# Names a person (or the model) may use for a provider that is registered
# under another name.
PROVIDER_NAME_ALIASES: dict[str, str] = {
    "x": "twitter",
    "x.com": "twitter",
    "bank": "plaid",
    "finance": "plaid",
    "bank_account": "plaid",
    "claude": "anthropic",
    "chatgpt": "openai",
    "google_mail": "gmail",
}


def get_provider(name: str) -> ConnectedAccountProvider | None:
    """Look up a provider by name, case-insensitively.

    The name reaches this function from an endpoint argument or from the model
    echoing back something a human typed, so "Gmail" and "gmail" both resolve.
    """
    key = str(name or "").strip().lower()
    return PROVIDER_REGISTRY.get(PROVIDER_NAME_ALIASES.get(key, key))


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
        if (
            provider.kind == KIND_MAILBOX
            and provider.send_supported
            and not provider.smtp_host
        ):
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
        if provider.credential_mechanism == MECHANISM_OAUTH:
            from src.anubis.utils.connected_accounts.oauth_providers import (
                get_oauth_provider,
            )

            if get_oauth_provider(provider.oauth_config_key or "") is None:
                raise ValueError(
                    f"Provider {provider.name!r} signs in with OAuth and must name "
                    "a known oauth_config_key."
                )
        if provider.credential_mechanism == MECHANISM_BROWSER_SESSION and not (
            provider.login_url or provider.connect_fields
        ):
            raise ValueError(
                f"Provider {provider.name!r} signs in through a live browser and "
                "must declare a login_url or a site_url field."
            )
        if provider.credential_mechanism == MECHANISM_URL_ONLY and not any(
            field_spec.name == "site_url" for field_spec in provider.connect_fields
        ):
            raise ValueError(
                f"Provider {provider.name!r} is a website and must declare a "
                "'site_url' field."
            )


validate_registry()
