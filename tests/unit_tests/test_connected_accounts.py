"""Unit tests for connecting an external account to the personal avatar.

Three properties are pinned down here, each because getting it wrong would be
expensive rather than merely untidy:

1. **A credential is proved before it is stored.** The whole reason the connect
   endpoint dials a real mail server is so a user who supplies their Google
   account password — which has not worked for mail since 2025-03-14 — is told
   so while the form is still in front of them. A regression that stored first
   and verified later would replace that with a mailbox that silently never
   works.
2. **Secrets never leave the process.** The listing projection is a whitelist,
   so a field added to the record later stays out of an API response by default.
3. **One account is one record.** The Model Context Protocol namespaces
   originally kept a single record under a constant key, which let a second
   machine overwrite the first and let one daemon delete another's record. These
   tests assert the analogous mistakes are impossible here: a second mailbox
   coexists with the first, and disconnecting names exactly one account.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import (
    account_key,
    build_account_record,
    deduplicate_label,
    derive_display_label,
    get_provider,
    public_account_view,
    social_providers,
)
from src.api import webapp as webapp_module

USER_ID = "auth0-user-abc"
ASSISTANT_ID = "assistant-personal"
ADDRESS = "evan@example.com"
APP_PASSWORD = "abcd efgh ijkl mnop"


def _current_user(user_id=USER_ID):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


def _context():
    """A context carrying a real Fernet key, so encryption is genuinely exercised."""
    return SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        max_connected_accounts_per_user=10,
        mailbox_request_timeout_seconds=5.0,
        mailbox_fetch_max_messages=25,
    )


class _StoreAPI:
    """Records store calls the way the SDK client would receive them."""

    def __init__(self, items=None):
        self.items = dict(items or {})
        self.deleted = []

    async def search_items(self, namespace, limit=100):
        return {
            "items": [{"key": key, "value": value} for key, value in self.items.items()]
        }

    async def put_item(self, namespace, key, value):
        self.items[key] = value

    async def get_item(self, namespace, key):
        value = self.items.get(key)
        return {"key": key, "value": value} if value is not None else None

    async def delete_item(self, namespace, key):
        self.deleted.append(key)
        self.items.pop(key, None)


def _install(monkeypatch, store_api, context=None):
    """Point the endpoints at a fake SDK client and a resolved personal avatar."""
    monkeypatch.setattr(
        webapp_module, "get_client", lambda **kwargs: SimpleNamespace(store=store_api)
    )
    monkeypatch.setattr(
        webapp_module.app, "state", SimpleNamespace(context=context or _context())
    )

    async def _resolve(client, request, user, api_key):
        return {"assistant_id": ASSISTANT_ID}

    monkeypatch.setattr(
        webapp_module,
        "_resolve_personal_avatar_for_connection",
        _resolve,
    )


def _accept_credentials(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    monkeypatch.setattr(imap_client, "verify_credentials", lambda credentials: None)


def _reject_credentials(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    def _raise(credentials):
        raise imap_client.MailboxAuthenticationError("AUTHENTICATIONFAILED")

    monkeypatch.setattr(imap_client, "verify_credentials", _raise)


def _body(**overrides):
    payload = {
        "provider": "gmail",
        "email_address": ADDRESS,
        "app_password": APP_PASSWORD,
    }
    payload.update(overrides)
    return SimpleNamespace(json=_async_returning(payload))


def _async_returning(value):
    async def _call():
        return value

    return _call


# --------------------------------------------------------------------------
# The credential is proved before anything is written
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_password_stores_nothing(monkeypatch):
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    _reject_credentials(monkeypatch)

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_mailbox(
            request=_body(), current_user=_current_user()
        )

    assert raised.value.status_code == 400
    # The message has to send the owner to the right place, because "wrong
    # password" alone makes people retype the same account password.
    assert "app password" in raised.value.detail.lower()
    assert "apppasswords" in raised.value.detail
    assert store_api.items == {}, "nothing may be stored when verification fails"


@pytest.mark.asyncio
async def test_an_unreachable_server_is_not_reported_as_a_bad_password(monkeypatch):
    """A network problem and a bad credential need different remedies."""
    from src.anubis.utils.tools.email import imap_client

    store_api = _StoreAPI()
    _install(monkeypatch, store_api)

    def _raise(credentials):
        raise imap_client.MailboxUnreachableError("connection timed out")

    monkeypatch.setattr(imap_client, "verify_credentials", _raise)

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_mailbox(
            request=_body(), current_user=_current_user()
        )

    assert raised.value.status_code == 503
    assert store_api.items == {}


@pytest.mark.asyncio
async def test_a_missing_encryption_key_fails_the_request_not_the_boot(monkeypatch):
    store_api = _StoreAPI()
    _install(
        monkeypatch,
        store_api,
        context=SimpleNamespace(
            connected_account_encryption_key=None,
            max_connected_accounts_per_user=10,
            mailbox_request_timeout_seconds=5.0,
        ),
    )
    _accept_credentials(monkeypatch)

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_mailbox(
            request=_body(), current_user=_current_user()
        )

    assert raised.value.status_code == 503
    assert "CONNECTED_ACCOUNT_ENCRYPTION_KEY" in raised.value.detail
    assert store_api.items == {}


# --------------------------------------------------------------------------
# Secrets never leave the process
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connecting_stores_ciphertext_and_returns_neither_secret(monkeypatch):
    store_api = _StoreAPI()
    context = _context()
    _install(monkeypatch, store_api, context=context)
    _accept_credentials(monkeypatch)

    response = await webapp_module.connect_mailbox(
        request=_body(), current_user=_current_user()
    )
    assert response.status_code == 200

    key = account_key("gmail", ADDRESS)
    stored = store_api.items[key]
    assert stored["encrypted_secret"] != APP_PASSWORD, "the secret must be encrypted"
    assert (
        secret_store.decrypt_secret(stored["encrypted_secret"], context) == APP_PASSWORD
    ), "and must round-trip, or the mailbox could never be opened"

    body = response.body.decode("utf-8")
    assert APP_PASSWORD not in body
    assert stored["encrypted_secret"] not in body


@pytest.mark.asyncio
async def test_listing_never_returns_the_password_or_the_ciphertext(monkeypatch):
    context = _context()
    record = build_account_record(
        provider=get_provider("gmail"),
        account_address=ADDRESS,
        display_label="evan",
        encrypted_secret=secret_store.encrypt_secret(APP_PASSWORD, context),
        assistant_id=ASSISTANT_ID,
    )
    store_api = _StoreAPI({record["account_key"]: record})
    _install(monkeypatch, store_api, context=context)

    response = await webapp_module.list_connected_accounts(
        request=SimpleNamespace(), current_user=_current_user()
    )
    body = response.body.decode("utf-8")

    assert ADDRESS in body, "the owner must be able to tell their accounts apart"
    assert APP_PASSWORD not in body
    assert record["encrypted_secret"] not in body
    assert "encrypted_secret" not in body


def test_the_public_projection_is_a_whitelist():
    """A field added to the record later must not leak by default."""
    record = build_account_record(
        provider=get_provider("gmail"),
        account_address=ADDRESS,
        display_label="evan",
        encrypted_secret="ciphertext",
        assistant_id=ASSISTANT_ID,
    )
    record["some_future_secret_field"] = "must not appear"

    view = public_account_view(record)

    assert "some_future_secret_field" not in view
    assert "encrypted_secret" not in view
    assert view["account_address"] == ADDRESS


# --------------------------------------------------------------------------
# One account is one record
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_mailbox_coexists_with_the_first(monkeypatch):
    """The bug the MCP namespaces had: a second thing overwriting the first."""
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    _accept_credentials(monkeypatch)

    await webapp_module.connect_mailbox(
        request=_body(email_address="evan@personal.com"),
        current_user=_current_user(),
    )
    await webapp_module.connect_mailbox(
        request=_body(email_address="evan@work.com"), current_user=_current_user()
    )

    assert len(store_api.items) == 2
    labels = sorted(record["display_label"] for record in store_api.items.values())
    assert labels == ["evan", "evan 2"], "same local part must still be distinguishable"


@pytest.mark.asyncio
async def test_reconnecting_refreshes_one_record_and_keeps_its_label(monkeypatch):
    """Rotating an app password must not create a duplicate or climb the label."""
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    _accept_credentials(monkeypatch)

    await webapp_module.connect_mailbox(request=_body(), current_user=_current_user())
    await webapp_module.connect_mailbox(
        request=_body(app_password="rotated password"), current_user=_current_user()
    )

    assert len(store_api.items) == 1
    assert list(store_api.items.values())[0]["display_label"] == "evan"


@pytest.mark.asyncio
async def test_the_account_cap_is_enforced_for_new_accounts_only(monkeypatch):
    context = _context()
    context.max_connected_accounts_per_user = 1
    store_api = _StoreAPI()
    _install(monkeypatch, store_api, context=context)
    _accept_credentials(monkeypatch)

    await webapp_module.connect_mailbox(request=_body(), current_user=_current_user())
    # Reconnecting the SAME account must still work at the cap, or rotating a
    # password would lock the owner out of their own mailbox.
    await webapp_module.connect_mailbox(
        request=_body(app_password="rotated"), current_user=_current_user()
    )
    assert len(store_api.items) == 1

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_mailbox(
            request=_body(email_address="second@example.com"),
            current_user=_current_user(),
        )
    assert raised.value.status_code == 409


@pytest.mark.asyncio
async def test_disconnecting_removes_exactly_one_account(monkeypatch):
    context = _context()
    first = build_account_record(
        provider=get_provider("gmail"),
        account_address="evan@personal.com",
        display_label="evan",
        encrypted_secret="c1",
        assistant_id=ASSISTANT_ID,
    )
    second = build_account_record(
        provider=get_provider("gmail"),
        account_address="evan@work.com",
        display_label="evan 2",
        encrypted_secret="c2",
        assistant_id=ASSISTANT_ID,
    )
    store_api = _StoreAPI({first["account_key"]: first, second["account_key"]: second})
    _install(monkeypatch, store_api, context=context)

    await webapp_module.disconnect_account(
        request=SimpleNamespace(),
        account_key=first["account_key"],
        current_user=_current_user(),
    )

    assert store_api.deleted == [first["account_key"]]
    assert second["account_key"] in store_api.items


@pytest.mark.asyncio
async def test_disconnecting_without_an_account_key_deletes_nothing(monkeypatch):
    """The /disconnect_mcp defect: an omitted identifier meaning 'delete all'."""
    context = _context()
    record = build_account_record(
        provider=get_provider("gmail"),
        account_address=ADDRESS,
        display_label="evan",
        encrypted_secret="c1",
        assistant_id=ASSISTANT_ID,
    )
    store_api = _StoreAPI({record["account_key"]: record})
    _install(monkeypatch, store_api, context=context)

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.disconnect_account(
            request=SimpleNamespace(), account_key="", current_user=_current_user()
        )

    assert raised.value.status_code == 400
    assert store_api.deleted == []
    assert store_api.items, "every account must survive a malformed disconnect"


@pytest.mark.asyncio
async def test_disconnecting_an_unknown_account_reports_404(monkeypatch):
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.disconnect_account(
            request=SimpleNamespace(),
            account_key="gmail:nobody@example.com",
            current_user=_current_user(),
        )
    assert raised.value.status_code == 404


# --------------------------------------------------------------------------
# Registry invariants
# --------------------------------------------------------------------------


def test_a_mailbox_is_never_a_social_account():
    """SECURITY: a mailbox must not satisfy an identity-verification gate.

    ``social_providers()`` is the allow-list a likeness-verification check reads.
    Owning an email address proves nothing about whose likeness an avatar
    depicts, so Gmail appearing in that tuple would open a verification hole.
    """
    social_names = {provider.name for provider in social_providers()}

    assert "gmail" not in social_names
    assert get_provider("gmail").kind == "mailbox"
    assert social_names, "the gate must be written against real members"
    assert all(provider.kind == "social" for provider in social_providers())


def test_a_non_mailbox_provider_cannot_be_connected_with_a_password(monkeypatch):
    """A social provider has no IMAP server; the endpoint must refuse, not crash."""

    async def _run():
        store_api = _StoreAPI()
        _install(monkeypatch, store_api)
        with pytest.raises(webapp_module.HTTPException) as raised:
            await webapp_module.connect_mailbox(
                request=_body(provider="youtube"), current_user=_current_user()
            )
        assert raised.value.status_code == 400
        assert store_api.items == {}

    import asyncio

    asyncio.run(_run())


def test_account_keys_are_stable_across_capitalisation():
    """A key that varied by capitalisation would duplicate the same mailbox."""
    assert account_key("Gmail", "Evan@Example.com") == account_key(
        "gmail", "evan@example.com"
    )


def test_labels_are_derived_and_deduplicated():
    assert derive_display_label("evan@example.com") == "evan"
    existing = [{"account_key": "gmail:a@b.c", "display_label": "evan"}]
    assert deduplicate_label("evan", existing, "gmail:other@b.c") == "evan 2"
    # Reconnecting the same account keeps its label rather than climbing.
    assert deduplicate_label("evan", existing, "gmail:a@b.c") == "evan"


def test_a_rotated_encryption_key_fails_closed():
    """A wrong key must raise, never return plausible-looking plaintext."""
    context = _context()
    ciphertext = secret_store.encrypt_secret(APP_PASSWORD, context)

    rotated = SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key()
    )
    with pytest.raises(secret_store.SecretDecryptionError):
        secret_store.decrypt_secret(ciphertext, rotated)


def test_an_absent_encryption_key_is_reported_as_configuration():
    with pytest.raises(secret_store.SecretEncryptionNotConfiguredError):
        secret_store.encrypt_secret(
            "x", SimpleNamespace(connected_account_encryption_key=None)
        )


# --------------------------------------------------------------------------
# The per-avatar gate
# --------------------------------------------------------------------------


class _InMemoryStore:
    """Minimal async store with the four methods this layer uses."""

    def __init__(self):
        self.data = {}

    async def aput(self, namespace, key, value):
        self.data.setdefault(tuple(namespace), {})[key] = value

    async def asearch(self, namespace, limit=100):
        items = self.data.get(tuple(namespace), {})
        return [SimpleNamespace(key=key, value=value) for key, value in items.items()]

    async def aget(self, namespace, key):
        value = self.data.get(tuple(namespace), {}).get(key)
        return SimpleNamespace(key=key, value=value) if value is not None else None

    async def adelete(self, namespace, key):
        self.data.get(tuple(namespace), {}).pop(key, None)


def _stored(address, assistant_id, status="connected"):
    record = build_account_record(
        provider=get_provider("gmail"),
        account_address=address,
        display_label=derive_display_label(address),
        encrypted_secret="ciphertext",
        assistant_id=assistant_id,
    )
    record["status"] = status
    return record


@pytest.mark.asyncio
async def test_only_the_bound_avatar_reaches_a_connected_account():
    """A second avatar of the SAME owner must not reach the owner's mail.

    Binding to the avatar rather than to the user is what makes a demoted
    personal avatar — or a shared one — lose access without any record being
    rewritten.
    """
    from src.anubis.utils.connected_accounts import (
        bound_accounts_for,
        connected_account_namespace,
        save_connected_account,
    )

    store = _InMemoryStore()
    mine = _stored("evan@example.com", ASSISTANT_ID)
    theirs = _stored("evan@work.com", "assistant-other")
    await save_connected_account(store, USER_ID, mine)
    await save_connected_account(store, USER_ID, theirs)

    bound = await bound_accounts_for(store, USER_ID, ASSISTANT_ID)

    assert [record["account_address"] for record in bound] == ["evan@example.com"]
    # Both records still exist; only visibility differs.
    assert len(store.data[connected_account_namespace(USER_ID)]) == 2


@pytest.mark.asyncio
async def test_an_account_needing_reconnection_is_withheld_from_the_tools():
    """A dead credential must not be handed to the tool layer as if it worked."""
    from src.anubis.utils.connected_accounts import (
        bound_accounts_for,
        mark_account_needs_reconnect,
        read_connected_accounts,
        save_connected_account,
    )

    store = _InMemoryStore()
    record = _stored("evan@example.com", ASSISTANT_ID)
    await save_connected_account(store, USER_ID, record)

    await mark_account_needs_reconnect(store, USER_ID, record["account_key"])

    assert await bound_accounts_for(store, USER_ID, ASSISTANT_ID) == []
    # It stays readable so the status block can tell the owner what to fix.
    still_listed = await read_connected_accounts(store, USER_ID)
    assert still_listed[0]["status"] == "needs_reconnect"


@pytest.mark.asyncio
async def test_an_unreachable_store_yields_no_accounts_rather_than_failing():
    """A store hiccup must never fail a turn whose real work is something else."""
    from src.anubis.utils.connected_accounts import read_connected_accounts

    class _BrokenStore:
        async def asearch(self, namespace, limit=100):
            raise RuntimeError("store is down")

    assert await read_connected_accounts(_BrokenStore(), USER_ID) == []
    assert await read_connected_accounts(None, USER_ID) == []


@pytest.mark.asyncio
async def test_saving_without_an_account_key_is_refused():
    """The key is the record's identity; a keyless write would be a singleton."""
    from src.anubis.utils.connected_accounts import save_connected_account

    with pytest.raises(ValueError):
        await save_connected_account(_InMemoryStore(), USER_ID, {"provider": "gmail"})
