"""Provisioning and resolution of the one PERSONAL_AVATAR_OF_THE_CREATOR per user.

Every signed-up user always has exactly one personal avatar — the single avatar
carrying ``is_personal_avatar_of_creator`` in its LangGraph assistant metadata.
That avatar is the only one allowed to reach the creator's private capabilities
(the desktop Model Context Protocol data servers, the connected mailbox, personal
and business analytics), and it is the avatar whose adapter is trained from the
creator's conversations. Because the personal avatar is guaranteed to exist, no
feature ever answers "create a personal avatar first": every caller resolves the
personal avatar and self-heals by provisioning one when none is found.

Provisioning runs the first time an email-verified account is seen on the API-key
authentication path (see ``get_user_with_api_key`` in ``src/security/auth.py``),
mirroring how ``ensure_initial_subscription_after_verification`` enrolls a newly
verified account into its initial subscription.

Why this module exists separately from both ``src/security/auth.py`` and
``src/api/webapp.py``: the provisioning helpers are needed by the authentication
path, and ``src/api/webapp.py`` already imports ``src/security/auth.py``, so
placing these helpers in the web application module would make the import
circular. ``src/security/auth.py`` imports this module lazily inside the calling
function, matching the cold-start lazy-import convention used elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


PERSONAL_AVATAR_METADATA_FLAG = "is_personal_avatar_of_creator"
"""Assistant-metadata key marking the user's one personal avatar."""

PERSONAL_AVATAR_PROVISIONED_MARKER = "personal_avatar_provisioned"
"""Auth0 ``app_metadata`` key that keeps provisioning off the authentication hot path."""

PERSONAL_AVATAR_IDENTIFIER_FIELD = "personal_avatar_id"
"""Auth0 ``app_metadata`` key holding the provisioned personal avatar's identifier."""

DEFAULT_PERSONAL_AVATAR_DESCRIPTION = (
    "The personal avatar of the account owner. This avatar reconstructs the "
    "owner's identity and is the only avatar of this account permitted to reach "
    "the owner's private capabilities."
)


@dataclass(frozen=True)
class PersonalAvatarCapability:
    """One exclusive capability of the personal avatar, for display to the owner.

    ``status_key`` names the field that ``GET /personal_avatar`` fills in with the
    capability's live state (for example the list of connected data-server
    devices). Capabilities whose implementation has not landed yet report a status
    of ``"not_configured"`` rather than being hidden, so the owner can see the full
    set of what the personal avatar is for.
    """

    name: str
    summary: str
    status_key: str


PERSONAL_AVATAR_CAPABILITIES: tuple[PersonalAvatarCapability, ...] = (
    PersonalAvatarCapability(
        name="desktop_data_servers",
        summary=(
            "Connect to the owner's own Model Context Protocol data servers "
            "(desktop and mobile machines) and analyze the files those servers "
            "expose."
        ),
        status_key="connected_data_servers",
    ),
    PersonalAvatarCapability(
        name="mailbox",
        summary=(
            "Read the owner's connected mailbox, triage each message, and act on "
            "a message by following the links the message contains."
        ),
        status_key="connected_mailboxes",
    ),
    PersonalAvatarCapability(
        name="social_accounts",
        summary=(
            "Link the owner's social media accounts to verify the owner's "
            "identity and to collect the owner's own published content."
        ),
        status_key="connected_social_accounts",
    ),
    PersonalAvatarCapability(
        name="browser_analytics",
        summary=(
            "Sign in to the owner's own accounts in a browser and report the "
            "metrics those accounts hold."
        ),
        status_key="browser_sessions",
    ),
    PersonalAvatarCapability(
        name="adapter_training_from_conversations",
        summary=(
            "Use the owner's messages from conversations with any avatar as "
            "training data for the personal avatar's adapter."
        ),
        status_key="adapter_training",
    ),
)


# User identifiers whose provisioning is currently in flight, guarding against
# re-entrancy. Creating an avatar calls the LangGraph server over HTTP, and that
# server authenticates the call through ``get_user_with_api_key`` — the very
# function that triggers provisioning. The API-key cache is populated before
# provisioning starts, so the nested authentication normally returns from cache
# without re-entering; this set is the second line of defense for the case where
# the cache entry is evicted while a creation is in flight.
_user_identifiers_being_provisioned: set[str] = set()
_provisioning_guard_lock: asyncio.Lock = asyncio.Lock()


def bare_user_identifier(user: dict) -> str | None:
    """Return the identifier used as ``user_id`` in avatar metadata.

    Avatar ownership is recorded with ``identities[0].user_id`` — the Auth0
    identifier WITHOUT the provider prefix — because that is what
    ``/create_avatar`` writes and what every ownership check compares against.
    The provider-prefixed ``user_id`` is a different string and is used only for
    Auth0 Management API calls.
    """
    identities = user.get("identities") or []
    if not identities:
        return None
    return identities[0].get("user_id")


def is_personal_avatar(avatar: dict) -> bool:
    """Report whether an avatar record carries the personal-avatar flag."""
    metadata = avatar.get("metadata") or {}
    return metadata.get(PERSONAL_AVATAR_METADATA_FLAG) is True


async def demote_other_personal_avatars(
    client: Any, user_id: str, keep_assistant_id: str | None
) -> None:
    """Enforce "at most one personal avatar per user" by demoting the rest.

    A user may flag exactly one avatar as their ``PERSONAL_AVATAR_OF_THE_CREATOR``
    (the only avatar that can reach their desktop data servers, mailbox, and
    personal analytics). When a new avatar is flagged, any *other* avatar of the
    same user that still holds the flag is cleared, so the newest choice wins
    without an error. ``keep_assistant_id`` is the avatar just flagged (never
    demoted).

    Shared by ``/create_avatar``, ``/modify_avatar``, and post-verification
    provisioning so all three enforce the invariant identically.
    """
    from src.anubis.utils.avatar_deletion import search_all_avatars_for_user

    try:
        owned_avatars = await search_all_avatars_for_user(client, user_id)
    except Exception:
        logger.warning(
            "Could not enumerate avatars to demote prior personal avatar for user %s",
            user_id,
            exc_info=True,
        )
        return

    for avatar in owned_avatars or []:
        avatar_id = avatar.get("assistant_id")
        if avatar_id == keep_assistant_id:
            continue
        if not is_personal_avatar(avatar):
            continue
        try:
            await client.assistants.update(
                assistant_id=avatar_id,
                metadata={PERSONAL_AVATAR_METADATA_FLAG: False},
            )
        except Exception:
            logger.warning(
                "Failed to demote prior personal avatar %s for user %s",
                avatar_id,
                user_id,
                exc_info=True,
            )


async def find_personal_avatar(client: Any, user_id: str) -> dict | None:
    """Return the user's flagged personal avatar, or ``None`` when none exists.

    Enumeration pages through ``search_all_avatars_for_user`` rather than calling
    ``assistants.search`` directly, because that search defaults to ``limit=10``
    and would silently hide a personal avatar created after the tenth avatar.
    When more than one avatar carries the flag (possible only if a demotion call
    failed earlier), the first is returned and the caller's demotion pass clears
    the remainder.
    """
    from src.anubis.utils.avatar_deletion import search_all_avatars_for_user

    owned_avatars = await search_all_avatars_for_user(client, user_id)
    for avatar in owned_avatars or []:
        if is_personal_avatar(avatar):
            return avatar
    return None


def default_personal_avatar_name(user: dict) -> str:
    """Choose the name for an auto-provisioned personal avatar.

    Prefers the account's own name, falls back to the local part of the email
    address, and finally to a fixed label — the name is cosmetic and the owner may
    rename the avatar at any time through ``/modify_avatar``.
    """
    account_name = (user.get("name") or "").strip()
    if account_name and "@" not in account_name:
        return account_name
    email_address = (user.get("email") or "").strip()
    if email_address and "@" in email_address:
        local_part = email_address.split("@", 1)[0].strip()
        if local_part:
            return local_part
    return "Personal Avatar"


async def create_personal_avatar(
    client: Any, user_id: str, *, name: str, description: str
) -> dict:
    """Create one flagged personal avatar and record its creator, then demote others.

    Performs the same two writes ``/create_avatar`` performs — the assistant
    itself and the ``creator_id`` store item — so every downstream reader of
    ``creator_id`` treats an auto-provisioned avatar exactly like a manually
    created one.

    Note what this deliberately does NOT do. ``/create_avatar`` starts a
    deep-research job for the avatar it creates, and this function does not,
    because it runs on the API-key authentication path the first time an account
    is seen and the name it has to work with at that moment is the local part of
    an email address (see ``default_personal_avatar_name``). Researching
    "j.smith" on the web would spend money learning about whoever that string
    happens to match and write a stranger's facts, face and voice into the
    account holder's own avatar. Research for a personal avatar belongs to a
    flow where the owner has given a real name.
    """
    assistant_id = str(uuid4())
    created_avatar = await client.assistants.create(
        graph_id="Anubis",
        description=description,
        name=name,
        assistant_id=assistant_id,
        metadata={
            "user_id": user_id,
            "is_public": False,
            PERSONAL_AVATAR_METADATA_FLAG: True,
        },
    )

    # Store the creator of the assistant. The langgraph_sdk StoreClient exposes
    # put_item (the HTTP store API), not the BaseStore aput method used on
    # in-process store objects.
    await client.store.put_item(
        (assistant_id, "creator_id"), key="creator_id", value={"value": user_id}
    )

    await demote_other_personal_avatars(client, user_id, keep_assistant_id=assistant_id)
    # Readers with no authenticated session find this avatar through the pointer.
    await record_personal_avatar_pointer(client, user_id, assistant_id)
    return created_avatar


async def ensure_personal_avatar_for_user(
    request: Any, user: dict, api_key: str
) -> dict | None:
    """Guarantee the verified account owns exactly one personal avatar.

    Called the first time an email-verified user is seen on the API-key path. The
    Auth0 ``app_metadata`` marker keeps this off the LangGraph server on cache
    misses, exactly as ``initial_subscription_provisioned`` keeps the subscription
    enrollment off Stripe.

    The sequence is:

    1. Return immediately when the marker is already present.
    2. Adopt an existing flagged avatar when the account already has one (an
       account that created and flagged its personal avatar by hand before
       auto-provisioning existed), writing the marker so this never runs again.
    3. Otherwise create the avatar, record its creator, and demote any other
       flagged avatar.
    4. Write the marker and the avatar identifier into ``app_metadata``.

    Best-effort and idempotent, exactly like the subscription enrollment: any
    failure logs and returns WITHOUT writing the marker, so the next cache-miss
    request retries, and a failure here never blocks authentication.

    Known gap, mirroring the one documented on the ``/login`` route: creating an
    avatar requires the LangGraph software development kit authenticated as the
    user, which needs the account's API key. The customer portal authenticates
    with email and password and never holds an API key (only its hash is stored),
    so a portal-only session cannot provision. Such an account receives its
    personal avatar on the first request made with its API key after
    verification.
    """
    from langgraph_sdk import get_client

    from src.security.auth import update_user_app_metadata_fields

    app_metadata = user.get("app_metadata") or {}
    if app_metadata.get(PERSONAL_AVATAR_PROVISIONED_MARKER):
        return None

    auth0_user_id = user.get("user_id")
    user_id = bare_user_identifier(user)
    if not auth0_user_id or not user_id or not api_key:
        return None

    # Re-entrancy guard: never start a second provisioning pass for a user while
    # one is in flight (see ``_user_identifiers_being_provisioned``).
    async with _provisioning_guard_lock:
        if user_id in _user_identifiers_being_provisioned:
            return None
        _user_identifiers_being_provisioned.add(user_id)

    try:
        client = get_client(headers={"API-KEY": f"{api_key}"})

        personal_avatar = await find_personal_avatar(client, user_id)
        if personal_avatar is None:
            # ``create_personal_avatar`` demotes any other flagged avatar itself.
            personal_avatar = await create_personal_avatar(
                client,
                user_id,
                name=default_personal_avatar_name(user),
                description=DEFAULT_PERSONAL_AVATAR_DESCRIPTION,
            )
            logger.info(
                "Provisioned the personal avatar %s for user %s after verification.",
                personal_avatar.get("assistant_id"),
                user_id,
            )
        else:
            # Adoption still has to restore exactly-one: an account can reach this
            # branch holding more than one flagged avatar when an earlier demotion
            # failed, and adopting the first without clearing the rest would leave
            # the invariant broken for good, since the marker stops this from
            # running again.
            await demote_other_personal_avatars(
                client, user_id, keep_assistant_id=personal_avatar.get("assistant_id")
            )
            await record_personal_avatar_pointer(
                client, user_id, str(personal_avatar.get("assistant_id") or "")
            )

        provisioned_fields = {
            PERSONAL_AVATAR_PROVISIONED_MARKER: True,
            PERSONAL_AVATAR_IDENTIFIER_FIELD: personal_avatar.get("assistant_id"),
        }
        await update_user_app_metadata_fields(
            request, auth0_user_id, provisioned_fields
        )
        user.setdefault("app_metadata", {}).update(provisioned_fields)
        return personal_avatar
    finally:
        async with _provisioning_guard_lock:
            _user_identifiers_being_provisioned.discard(user_id)


async def resolve_personal_avatar(
    client: Any, request: Any, user: dict, api_key: str
) -> dict | None:
    """Return the user's personal avatar, provisioning one when none is found.

    The self-healing resolution every caller should use. The personal avatar is
    guaranteed to exist, so a missing avatar is a provisioning gap to close
    silently rather than an error to report back to the owner.
    """
    user_id = bare_user_identifier(user)
    if not user_id:
        return None

    personal_avatar = await find_personal_avatar(client, user_id)
    if personal_avatar is not None:
        # Heal the pointer for an account provisioned before the pointer existed.
        # Deliberately here rather than in ``load_consciousness``: every route
        # that needs the personal avatar already passes through this function,
        # so the pointer is repaired without a single store read being added to
        # the path a reply travels.
        await record_personal_avatar_pointer(
            client, user_id, str(personal_avatar.get("assistant_id") or "")
        )
        return personal_avatar

    # A missing avatar means either provisioning never ran for this account or a
    # previous run failed after the marker was written. Clear the marker so the
    # provisioning pass below is not short-circuited by a stale one.
    user.setdefault("app_metadata", {}).pop(PERSONAL_AVATAR_PROVISIONED_MARKER, None)
    return await ensure_personal_avatar_for_user(request, user, api_key)


PERSONAL_AVATAR_POINTER_NAMESPACE_ROOT = "personal_avatar_of_user"
"""Namespace root of the per-user pointer naming that user's personal avatar."""

PERSONAL_AVATAR_POINTER_KEY = "personal_avatar"


def personal_avatar_pointer_namespace(user_id: str) -> tuple[str, str]:
    """Return the store namespace holding one user's personal-avatar pointer."""
    return (PERSONAL_AVATAR_POINTER_NAMESPACE_ROOT, user_id)


async def record_personal_avatar_pointer(
    store_or_client: Any, user_id: str, assistant_id: str
) -> None:
    """Note which avatar is this user's personal avatar, for readers with no session.

    ``find_personal_avatar`` needs a LangGraph software development kit client
    authenticated as the user, which a background sweeper and a graph node both
    lack — the sweeper holds only the store, and the graph holds only the store
    and the identifiers on the turn. This pointer is how those readers answer
    "which avatar is this speaker's own" without an authenticated call.

    Accepts either flavour of store: the in-process ``BaseStore`` (``aput``) that
    graph nodes hold, or the software development kit's ``StoreClient``
    (``put_item``) that the provisioning path holds. Best effort — a missing
    pointer costs a reader nothing except skipping that user, which is why no
    caller waits on the result.
    """
    if store_or_client is None or not user_id or not assistant_id:
        return
    namespace = personal_avatar_pointer_namespace(user_id)
    value = {
        "value": {
            "assistant_id": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    }
    try:
        if hasattr(store_or_client, "aput"):
            await store_or_client.aput(
                namespace, key=PERSONAL_AVATAR_POINTER_KEY, value=value
            )
            return
        store_api = getattr(store_or_client, "store", None)
        if store_api is not None and hasattr(store_api, "put_item"):
            await store_api.put_item(
                list(namespace), key=PERSONAL_AVATAR_POINTER_KEY, value=value
            )
            return
        if hasattr(store_or_client, "put_item"):
            await store_or_client.put_item(
                list(namespace), key=PERSONAL_AVATAR_POINTER_KEY, value=value
            )
    except Exception as write_error:  # noqa: BLE001 - never block the caller
        logger.debug(
            "Could not record the personal avatar pointer for %s: %s",
            user_id,
            write_error,
        )


async def read_personal_avatar_id(store: Any, user_id: str) -> str | None:
    """Return the identifier of this user's personal avatar, or ``None``.

    ``None`` means only "no pointer has been written for this user yet", never
    "this user has no personal avatar" — every account has one. A reader that
    gets ``None`` should skip the user and try again later, by which time a turn
    on the personal avatar will have healed the pointer.
    """
    if store is None or not user_id:
        return None
    try:
        item = await store.aget(
            personal_avatar_pointer_namespace(user_id), PERSONAL_AVATAR_POINTER_KEY
        )
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Personal avatar pointer lookup failed: %s", read_error)
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    inner = (value or {}).get("value") if isinstance(value, dict) else None
    if isinstance(inner, dict):
        return str(inner.get("assistant_id") or "") or None
    if isinstance(value, dict) and value.get("assistant_id"):
        return str(value["assistant_id"])
    return None


PERSONAL_AVATAR_RESEARCH_CLAIM_CATEGORY = "personal_avatar_research_claim"
"""Store-namespace category recording that research has run for a personal avatar."""


def personal_avatar_research_claim_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Return the store namespace holding this avatar's research claim."""
    return (creator_id, assistant_id, PERSONAL_AVATAR_RESEARCH_CLAIM_CATEGORY)


def _normalized_name_text(value: str) -> str:
    """Lower-case ``value`` and reduce the separators used in addresses to spaces.

    ``j.smith``, ``j_smith`` and ``J Smith`` all normalise to ``j smith``, which
    is what lets :func:`looks_like_a_real_person_name` recognise an email local
    part that has merely been prettied up.
    """
    lowered = (value or "").strip().lower()
    for separator in (".", "_", "-", "+"):
        lowered = lowered.replace(separator, " ")
    return " ".join(lowered.split())


def looks_like_a_real_person_name(
    candidate: str, *, email_address: str | None = None
) -> bool:
    """Report whether ``candidate`` is a real person's name the owner supplied.

    ``create_personal_avatar`` refuses to research the avatar it provisions
    because the only name available at that moment is the local part of an email
    address, and researching ``j.smith`` on the web would spend money learning
    about whoever that string happens to match. This predicate is the guard that
    lets a LATER rename start research: research may run once the owner has
    replaced the placeholder with a name a person would actually be called.

    A candidate qualifies only when every one of these holds:

    - two or more whitespace-separated parts, at least one of which is a word
      rather than an initial, so a single word is refused while "J. R. R.
      Tolkien" is allowed;
    - letters, hyphens, apostrophes and periods only, with no digits and no ``@``;
    - the candidate is not the ``"Personal Avatar"`` placeholder;
    - and, when ``email_address`` is given, the candidate does not normalise to
      the address's local part or to the whole address. This is the load-bearing
      rule: ``J Smith`` for ``j.smith@example.com`` is the placeholder wearing a
      space, not a name the owner chose.
    """
    cleaned = (candidate or "").strip()
    if not cleaned or "@" in cleaned:
        return False
    if cleaned.casefold() == "personal avatar":
        return False
    if any(character.isdigit() for character in cleaned):
        return False

    parts = cleaned.split()
    if len(parts) < 2:
        return False
    if not all(
        all(character.isalpha() or character in "-'." for character in part)
        for part in parts
    ):
        return False
    # At least one part has to be a name rather than an initial. Initials are
    # perfectly real — "J. R. R. Tolkien" is a name a person goes by — so the
    # rule is about the whole name, not about each part. The placeholder case
    # this used to catch ("J Smith" for j.smith@example.com) is caught by the
    # address comparison below, which is the check that actually knows what the
    # placeholder was.
    if not any(len(part.strip(".'-")) >= 2 for part in parts):
        return False

    if email_address and "@" in email_address:
        local_part = email_address.split("@", 1)[0]
        normalized_candidate = _normalized_name_text(cleaned)
        if normalized_candidate in (
            _normalized_name_text(local_part),
            _normalized_name_text(email_address),
        ):
            return False
    return True


async def claim_personal_avatar_research(
    store: Any, *, creator_id: str, assistant_id: str
) -> bool:
    """Record that research has started for this avatar; report whether this call won.

    Returns ``False`` when a claim already exists, which is what limits the naming
    trigger to one research run per personal avatar. The claim lives in the store
    rather than in memory so that renaming the avatar again after a restart still
    does not re-research the owner. Mirrors ``claim_bootstrap`` in
    ``src/anubis/utils/research/asset_bootstrap.py``.

    Pressing "Research & verify facts" by hand is unaffected: that route never
    consults this claim, because a deliberate press is the owner asking for
    exactly the run this guard exists to prevent happening by accident.
    """
    namespace = personal_avatar_research_claim_namespace(creator_id, assistant_id)
    try:
        existing_claim = await store.aget(namespace, assistant_id)
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug(
            "Personal avatar research claim lookup failed (continuing): %s", read_error
        )
        existing_claim = None
    if existing_claim is not None:
        return False
    try:
        await store.aput(
            namespace,
            key=assistant_id,
            value={"status": "claimed", "claimed_at": datetime.now(UTC).isoformat()},
        )
    except Exception as write_error:  # noqa: BLE001 - never block the rename
        logger.warning(
            "Could not record the personal avatar research claim for %s: %s",
            assistant_id,
            write_error,
        )
        return False
    return True


async def personal_avatar_id_for_owner(pool: Any, user_id: str) -> str | None:
    """Return the id of ``user_id``'s personal avatar, read straight from Postgres.

    ``resolve_personal_avatar`` is the API-layer answer and needs a LangGraph
    client, a request and an API key. The graph has none of those in the middle
    of a turn, but it does have the pool, and the personal avatar is only ever a
    row in ``assistant`` carrying this owner's id and the personal flag.

    Returns ``None`` when the owner has no personal avatar yet. That is a real
    state — provisioning happens on the first API-key authentication — and the
    caller treats it as "no brokered connections", never as an error.
    """
    if pool is None or not user_id:
        return None
    query = (
        "SELECT assistant_id FROM assistant "
        "WHERE metadata->>'user_id' = %s "
        "AND metadata->>'is_personal_avatar_of_creator' = 'true' "
        "ORDER BY created_at ASC LIMIT 1"
    )
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, (user_id,))
                row = await cursor.fetchone()
    except Exception:  # noqa: BLE001 - a lookup failure means no brokering, not a broken turn
        logger.debug(
            "Could not resolve the personal avatar for brokering", exc_info=True
        )
        return None
    if not row:
        return None
    return str(row[0])


__all__ = [
    "DEFAULT_PERSONAL_AVATAR_DESCRIPTION",
    "PERSONAL_AVATAR_CAPABILITIES",
    "PERSONAL_AVATAR_IDENTIFIER_FIELD",
    "PERSONAL_AVATAR_METADATA_FLAG",
    "PERSONAL_AVATAR_POINTER_KEY",
    "PERSONAL_AVATAR_POINTER_NAMESPACE_ROOT",
    "PERSONAL_AVATAR_PROVISIONED_MARKER",
    "PERSONAL_AVATAR_RESEARCH_CLAIM_CATEGORY",
    "PersonalAvatarCapability",
    "bare_user_identifier",
    "claim_personal_avatar_research",
    "create_personal_avatar",
    "default_personal_avatar_name",
    "demote_other_personal_avatars",
    "ensure_personal_avatar_for_user",
    "find_personal_avatar",
    "is_personal_avatar",
    "looks_like_a_real_person_name",
    "personal_avatar_id_for_owner",
    "personal_avatar_pointer_namespace",
    "personal_avatar_research_claim_namespace",
    "read_personal_avatar_id",
    "record_personal_avatar_pointer",
    "resolve_personal_avatar",
]
