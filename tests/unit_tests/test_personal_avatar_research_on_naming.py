"""Research the account holder only once the account holder has given a real name.

``create_personal_avatar`` refuses to research the avatar it provisions, and the
reason is in its docstring: at provisioning time the only name available is the
local part of an email address, and researching "j.smith" would spend money
learning about whoever that string happens to match and write a stranger's facts,
face and voice into the account holder's own avatar. That docstring says research
"belongs to a flow where the owner has given a real name" — and until now no flow
did. What is pinned down:

- **The placeholder never triggers research**, including the placeholder wearing a
  space, which is the case a naive "does it contain a space" check would let past.
- **A real name triggers research exactly once**, and a later rename does not
  research again, because the claim is in the store and survives a restart.
- **The kill switch stops only this trigger**, leaving the manual button alone.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.personal_avatar import (
    claim_personal_avatar_research,
    looks_like_a_real_person_name,
    personal_avatar_research_claim_namespace,
)

ADDRESS = "j.smith@example.com"


class _Store:
    def __init__(self) -> None:
        self.rows: dict = {}

    async def aget(self, namespace, key):
        if (namespace, key) not in self.rows:
            return None
        return type("Item", (), {"value": self.rows[(namespace, key)]})()

    async def aput(self, namespace, key, value):
        self.rows[(namespace, key)] = value


@pytest.mark.parametrize(
    "candidate",
    [
        "j.smith",
        "jsmith",
        "J Smith",
        "j smith",
        "J.Smith",
        "Personal Avatar",
        "Evan",
        "",
        "   ",
        "evan@example.com",
        "Agent 47",
    ],
)
def test_names_that_must_not_start_research(candidate: str) -> None:
    assert not looks_like_a_real_person_name(candidate, email_address=ADDRESS)


@pytest.mark.parametrize(
    "candidate",
    [
        "Jane Smith",
        "Mary-Jane O'Brien",
        "Jean-Luc Picard",
        "Ada Lovelace",
        "J. R. R. Tolkien",
    ],
)
def test_names_that_may_start_research(candidate: str) -> None:
    assert looks_like_a_real_person_name(candidate, email_address=ADDRESS)


def test_the_local_part_is_refused_however_it_is_dressed_up() -> None:
    # The load-bearing case: an owner who "renames" the avatar to a prettied-up
    # version of their email address has not given a real name.
    for candidate in ("E Woods Business", "e.woods.business", "e_woods_business"):
        assert not looks_like_a_real_person_name(
            candidate, email_address="e.woods.business@icloud.com"
        )
    assert looks_like_a_real_person_name(
        "Evan Woods", email_address="e.woods.business@icloud.com"
    )


def test_a_name_is_judged_on_its_own_when_no_address_is_known() -> None:
    assert looks_like_a_real_person_name("Jane Smith")
    assert not looks_like_a_real_person_name("Jane")


@pytest.mark.asyncio
async def test_research_is_claimed_once_and_only_once() -> None:
    store = _Store()
    assert await claim_personal_avatar_research(
        store, creator_id="creator", assistant_id="avatar-1"
    )
    # A second rename, a restart, a retry: the claim is in the store, so no.
    assert not await claim_personal_avatar_research(
        store, creator_id="creator", assistant_id="avatar-1"
    )
    # A different avatar of the same owner is a separate claim.
    assert await claim_personal_avatar_research(
        store, creator_id="creator", assistant_id="avatar-2"
    )
    assert (
        personal_avatar_research_claim_namespace("creator", "avatar-1"),
        "avatar-1",
    ) in store.rows


@pytest.mark.asyncio
async def test_a_store_that_cannot_write_does_not_claim() -> None:
    # Better to research later than to record a claim that was never honoured.
    class _Unwritable(_Store):
        async def aput(self, namespace, key, value):
            raise RuntimeError("the store said no")

    assert not await claim_personal_avatar_research(
        _Unwritable(), creator_id="creator", assistant_id="avatar-1"
    )
