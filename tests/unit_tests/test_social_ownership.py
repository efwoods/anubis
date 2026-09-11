"""Proving a social account belongs to the avatar's person, and refusing when it does not.

The ownership gate is the prerequisite the whole subscription feature rests on,
so these tests are mostly about the refusals: a record with no proof, a record
belonging to another avatar, and a mailbox offered as if it were a likeness.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.connected_accounts import ownership


def _social_record(**overrides):
    record = {
        "account_key": "youtube:person@example.com",
        "provider": "youtube",
        "kind": "social",
        "credential_mechanism": "oauth",
        "account_address": "person@example.com",
        "display_label": "YouTube",
        "assistant_id": "avatar-1",
        "user_id": "owner-1",
        "status": "connected",
        "transport": {},
    }
    record.update(overrides)
    return record


def test_a_record_without_a_proof_is_unproven() -> None:
    """A record written before ownership existed must not be trusted by default.

    The safe direction for a gate deciding whose words become the avatar's is
    to refuse what it cannot vouch for.
    """
    record = _social_record()
    assert ownership.is_ownership_proven(record) is False
    assert ownership.ownership_of(record)["state"] == ownership.OWNERSHIP_UNPROVEN


def test_a_proven_record_passes_the_gate() -> None:
    record = _social_record(
        ownership=ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN,
            method=ownership.METHOD_OAUTH_IDENTITY,
            handle="person",
        )
    )
    assert ownership.is_ownership_proven(record) is True
    assert (
        ownership.is_owned_by_personal_avatar(record, personal_avatar_id="avatar-1")
        is True
    )
    assert ownership.refusal_reason(record, personal_avatar_id="avatar-1") is None


def test_a_proven_account_of_another_avatar_is_refused() -> None:
    """Proof is about a person, but binding is about an avatar; both must hold."""
    record = _social_record(
        ownership=ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN, method=ownership.METHOD_OAUTH_IDENTITY
        )
    )
    assert (
        ownership.is_owned_by_personal_avatar(record, personal_avatar_id="avatar-2")
        is False
    )
    reason = ownership.refusal_reason(record, personal_avatar_id="avatar-2")
    assert reason is not None
    assert "not connected to this personal avatar" in reason


def test_a_mailbox_can_never_discharge_a_likeness_claim() -> None:
    """Owning an email address proves nothing about who a likeness depicts."""
    record = _social_record(
        kind="mailbox",
        ownership=ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN, method=ownership.METHOD_OAUTH_IDENTITY
        ),
    )
    assert (
        ownership.is_owned_by_personal_avatar(record, personal_avatar_id="avatar-1")
        is False
    )


def test_an_unbound_record_is_refused() -> None:
    """A record naming no avatar cannot be matched to the avatar being updated."""
    record = _social_record(
        assistant_id="",
        ownership=ownership.build_ownership(state=ownership.OWNERSHIP_PROVEN),
    )
    assert (
        ownership.is_owned_by_personal_avatar(record, personal_avatar_id="avatar-1")
        is False
    )


@pytest.mark.asyncio
async def test_oauth_sign_in_is_itself_the_proof() -> None:
    """The vendor already stated who signed in; nothing further is needed."""
    record = _social_record(provider="twitter", account_address="evan")
    result = await ownership.prove_ownership(object(), None, record)
    assert result["state"] == ownership.OWNERSHIP_PROVEN
    assert result["method"] == ownership.METHOD_OAUTH_IDENTITY
    assert result["handle"] == "evan"
    assert result["profile_url"] == "https://x.com/evan"


@pytest.mark.asyncio
async def test_a_feed_with_no_evidence_stays_unproven(monkeypatch) -> None:
    """Anyone can name anyone's blog, so a bare address proves nothing."""
    record = _social_record(
        provider="podcast_feed",
        credential_mechanism="url_only",
        transport={"site_url": "https://example.com/feed.xml"},
    )

    async def _fetch(url, **kwargs):
        return "<rss><channel><title>Some show</title></channel></rss>"

    monkeypatch.setattr(ownership, "_fetch_text", _fetch)
    result = await ownership.prove_ownership(object(), None, record)
    assert result["state"] == ownership.OWNERSHIP_UNPROVEN
    assert "verification token" in (result["detail"] or "")


@pytest.mark.asyncio
async def test_a_feed_carrying_the_verification_token_is_proven(monkeypatch) -> None:
    """Placing the token requires write access to the thing being claimed."""
    token = ownership.new_verification_token()
    record = _social_record(
        provider="podcast_feed",
        credential_mechanism="url_only",
        transport={
            "site_url": "https://example.com/feed.xml",
            "verification_token": token,
        },
    )

    async def _fetch(url, **kwargs):
        return f"<rss><channel><description>{token}</description></channel></rss>"

    monkeypatch.setattr(ownership, "_fetch_text", _fetch)
    result = await ownership.prove_ownership(object(), None, record)
    assert result["state"] == ownership.OWNERSHIP_PROVEN
    assert result["method"] == ownership.METHOD_VERIFICATION_TOKEN


@pytest.mark.asyncio
async def test_a_feed_linking_back_to_a_proven_account_is_proven(monkeypatch) -> None:
    """A rel="me" link back is the IndieAuth convention and requires the same access."""
    record = _social_record(
        provider="profile_url",
        credential_mechanism="url_only",
        transport={"site_url": "https://example.com"},
    )

    async def _fetch(url, **kwargs):
        return '<html><a rel="me" href="https://x.com/evan">me</a></html>'

    monkeypatch.setattr(ownership, "_fetch_text", _fetch)
    result = await ownership.prove_ownership(object(), None, record, identity_hint="evan")
    assert result["state"] == ownership.OWNERSHIP_PROVEN
    assert result["method"] == ownership.METHOD_LINKED_FROM_PROVEN


@pytest.mark.asyncio
async def test_a_failed_proof_leaves_the_account_connected_but_unproven(
    monkeypatch,
) -> None:
    """A proof that crashes is a state the owner can fix, not a failed connection."""
    record = _social_record(provider="instagram", credential_mechanism="browser_session")

    async def _explode(*args, **kwargs):
        raise RuntimeError("the browser is unavailable")

    monkeypatch.setattr(ownership, "_read_page_html", _explode)
    result = await ownership.prove_ownership(object(), None, record)
    assert result["state"] == ownership.OWNERSHIP_UNPROVEN


def test_the_handle_is_read_out_of_a_signed_in_page() -> None:
    """The handle is read rather than asserted, so the record names the real account."""
    assert (
        ownership._extract_handle('{"username":"evan.woods"}', "instagram")
        == "evan.woods"
    )
    assert ownership._extract_handle('{"login":"evanwoods"}', "twitch") == "evanwoods"
    assert ownership._extract_handle("nothing useful here", "instagram") is None
