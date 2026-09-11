"""The accounts deep research turns up, and the question each one becomes.

Research already reads the subject's own pages. When one of those pages lives at
a host the connector catalog covers, the owner is one sign-in away from the avatar
reading that account directly instead of inferring the subject from whatever the
open web happened to publish. What is pinned down:

- **Evidence outranks coincidence.** A page that supported a fact the research
  verified is better evidence than a page the search merely returned.
- **One question per provider**, never two because two of a vendor's pages were read.
- **A question that cannot be answered is never asked**: an account already
  connected, or a provider the catalog cannot yet connect.
- **An answered question never reopens.** A second research run must not ask the
  owner again about an account they already connected or declined.
- **Connecting one account answers one question**, leaving the others open.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.research.discovered_accounts import (
    ACCOUNT_STATUS_CONNECTED,
    ACCOUNT_STATUS_DECLINED,
    candidate_accounts_from_research,
    discovered_accounts_namespace,
    mark_discovered_account_resolved,
    open_accounts,
    read_discovered_accounts,
    record_discovered_accounts,
)

CREATOR = "auth0|creator"
AVATAR = "avatar-1"


def _summary() -> dict:
    return {
        # The channel supported a verified fact; the rest were merely read.
        "media_source_urls": ["https://www.youtube.com/@evanwoods/videos"],
        "bootstrap_media_urls": [],
        "sources": [
            {"url": "https://twitter.com/evanwoods", "title": "Evan Woods"},
            {"url": "https://github.com/evdev3", "title": "evdev3"},
            {"url": "https://en.wikipedia.org/wiki/Bridges", "title": "Bridges"},
            {"url": "https://www.youtube.com/@evanwoods", "title": "the same channel"},
        ],
    }


class _Store:
    def __init__(self) -> None:
        self.rows: dict = {}

    async def aget(self, namespace, key):
        if (namespace, key) not in self.rows:
            return None
        value = self.rows[(namespace, key)]
        return type("Item", (), {"value": value})()

    async def aput(self, namespace, key, value):
        self.rows[(namespace, key)] = value


def test_verified_evidence_is_ranked_first() -> None:
    found = candidate_accounts_from_research(
        _summary(), connected_provider_names=set(), maximum=3
    )
    assert [account.provider for account in found] == ["youtube", "twitter", "github"]
    assert found[0].supported_a_verified_fact is True
    assert found[1].supported_a_verified_fact is False
    # The page that suggested each account is carried, so the question can name it.
    assert found[0].evidence_url.startswith("https://www.youtube.com/@evanwoods")


def test_one_question_per_provider() -> None:
    found = candidate_accounts_from_research(
        _summary(), connected_provider_names=set(), maximum=5
    )
    providers = [account.provider for account in found]
    assert len(providers) == len(set(providers))


def test_an_account_already_connected_is_never_asked_about() -> None:
    found = candidate_accounts_from_research(
        _summary(), connected_provider_names={"youtube", "github"}, maximum=3
    )
    assert [account.provider for account in found] == ["twitter"]


def test_the_ceiling_is_about_the_owners_attention() -> None:
    found = candidate_accounts_from_research(
        _summary(), connected_provider_names=set(), maximum=1
    )
    assert len(found) == 1
    assert candidate_accounts_from_research(_summary(), maximum=0) == []


def test_a_page_that_is_not_an_account_suggests_nothing() -> None:
    assert (
        candidate_accounts_from_research(
            {"sources": [{"url": "https://en.wikipedia.org/wiki/Bridges"}]}
        )
        == []
    )
    assert candidate_accounts_from_research({}) == []


def test_a_malformed_address_suggests_nothing() -> None:
    assert (
        candidate_accounts_from_research({"sources": [{"url": "not a url at all"}]})
        == []
    )


@pytest.mark.asyncio
async def test_the_record_round_trips_and_names_the_namespace() -> None:
    store = _Store()
    found = candidate_accounts_from_research(_summary(), maximum=3)
    await record_discovered_accounts(
        store,
        creator_id=CREATOR,
        assistant_id=AVATAR,
        job_id="job-1",
        subject_name="Evan Woods",
        accounts=found,
    )
    assert (discovered_accounts_namespace(CREATOR, AVATAR), AVATAR) in store.rows
    assert discovered_accounts_namespace(CREATOR, AVATAR) == (
        CREATOR,
        AVATAR,
        "research_discovered_accounts",
    )
    record = await read_discovered_accounts(
        store, creator_id=CREATOR, assistant_id=AVATAR
    )
    assert len(open_accounts(record)) == 3
    assert record["subject_name"] == "Evan Woods"


@pytest.mark.asyncio
async def test_connecting_one_account_answers_only_that_question() -> None:
    store = _Store()
    await record_discovered_accounts(
        store,
        creator_id=CREATOR,
        assistant_id=AVATAR,
        job_id="job-1",
        subject_name="Evan Woods",
        accounts=candidate_accounts_from_research(_summary(), maximum=3),
    )
    await mark_discovered_account_resolved(
        store, creator_id=CREATOR, assistant_id=AVATAR, provider="twitter"
    )
    record = await read_discovered_accounts(
        store, creator_id=CREATOR, assistant_id=AVATAR
    )
    statuses = {item["provider"]: item["status"] for item in record["accounts"]}
    assert statuses == {
        "youtube": "open",
        "twitter": ACCOUNT_STATUS_CONNECTED,
        "github": "open",
    }
    assert {account["provider"] for account in open_accounts(record)} == {
        "youtube",
        "github",
    }


@pytest.mark.asyncio
async def test_a_second_research_run_does_not_reopen_an_answered_question() -> None:
    store = _Store()
    found = candidate_accounts_from_research(_summary(), maximum=3)
    await record_discovered_accounts(
        store,
        creator_id=CREATOR,
        assistant_id=AVATAR,
        job_id="job-1",
        subject_name="Evan Woods",
        accounts=found,
    )
    await mark_discovered_account_resolved(
        store,
        creator_id=CREATOR,
        assistant_id=AVATAR,
        provider="twitter",
        status=ACCOUNT_STATUS_DECLINED,
    )
    # The same run again: the declined account must stay declined.
    await record_discovered_accounts(
        store,
        creator_id=CREATOR,
        assistant_id=AVATAR,
        job_id="job-2",
        subject_name="Evan Woods",
        accounts=found,
    )
    record = await read_discovered_accounts(
        store, creator_id=CREATOR, assistant_id=AVATAR
    )
    statuses = {item["provider"]: item["status"] for item in record["accounts"]}
    assert statuses["twitter"] == ACCOUNT_STATUS_DECLINED
    assert statuses["youtube"] == "open"


@pytest.mark.asyncio
async def test_reading_an_avatar_that_was_never_researched() -> None:
    store = _Store()
    assert (
        await read_discovered_accounts(store, creator_id=CREATOR, assistant_id=AVATAR)
        is None
    )
    assert open_accounts(None) == []
    # Nothing found means nothing written, so no empty row is left behind.
    assert (
        await record_discovered_accounts(
            store,
            creator_id=CREATOR,
            assistant_id=AVATAR,
            job_id="job-1",
            subject_name="Evan Woods",
            accounts=[],
        )
        is None
    )
    assert store.rows == {}
