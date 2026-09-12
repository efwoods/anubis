"""The initial crawl: breadth-first, bounded, and pruned to the avatar's own person.

The pruning rule is the load-bearing behaviour. A page that is not about the
person is not merely skipped — its links are dropped, because they lead further
away. Without that, a crawl of a profile becomes a crawl of the web.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.connected_accounts import ownership
from src.anubis.utils.subscriptions import crawl


def _context(**overrides):
    settings = {
        "social_crawl_max_items_free": 5,
        "social_crawl_max_items_pro": 50,
        "social_crawl_max_items_premium": 300,
        "social_crawl_max_depth": 3,
        "social_crawl_max_nodes": 200,
        "social_crawl_relevance_minimum_score": 0.5,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def _proven_record(**overrides):
    record = {
        "account_key": "instagram:evan",
        "provider": "instagram",
        "kind": "social",
        "display_label": "Instagram",
        "assistant_id": "avatar-1",
        "user_id": "owner-1",
        "transport": {},
        "ownership": ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN,
            method=ownership.METHOD_SIGNED_IN_SESSION,
            handle="evan",
            profile_url="https://www.instagram.com/evan/",
        ),
    }
    record.update(overrides)
    return record


@pytest.fixture
def _pipeline(monkeypatch):
    """Record what the crawl ingested without touching the media pipeline."""
    ingested = []

    async def _fake_ingest(**kwargs):
        ingested.append(kwargs["url"])
        return {"status": "started", "job_id": "job-1"}

    monkeypatch.setattr(
        "src.anubis.utils.subscriptions.intake.ingest_content_url", _fake_ingest
    )
    return ingested


def _judge(verdicts):
    """Answer the relevance question from a table, defaulting to irrelevant."""

    async def _judge_relevance(context, *, url, avatar_name, avatar_description, page_text=None):
        return verdicts.get(
            url,
            {
                "is_about_target": False,
                "is_targets_own_words": False,
                "relevance": 0.0,
                "worth_following": False,
                "reasoning": "not this person",
            },
        )

    return _judge_relevance


def _links(mapping):
    async def _discover_links(url):
        return list(mapping.get(url, []))

    return _discover_links


@pytest.mark.asyncio
async def test_an_unproven_account_is_never_crawled(monkeypatch, _pipeline):
    record = _proven_record(ownership=ownership.build_ownership(state="unproven"))
    report = await crawl.crawl_connected_account(
        _context(),
        record=record,
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
    )
    assert report["status"] == "refused"
    assert _pipeline == []


@pytest.mark.asyncio
async def test_a_branch_with_no_content_about_the_person_stops_expanding(
    monkeypatch, _pipeline
):
    """The pruning rule: an unrelated page's links are dropped, not queued."""
    profile = "https://www.instagram.com/evan/"
    own_post = "https://www.instagram.com/p/mine/"
    stranger = "https://example.com/unrelated"
    deep_behind_stranger = "https://example.com/unrelated/deeper"

    monkeypatch.setattr(
        crawl,
        "judge_relevance",
        _judge(
            {
                profile: {
                    "is_about_target": True,
                    "is_targets_own_words": True,
                    "relevance": 1.0,
                    "worth_following": True,
                    "reasoning": "the person's own profile",
                },
                own_post: {
                    "is_about_target": True,
                    "is_targets_own_words": True,
                    "relevance": 0.9,
                    "worth_following": False,
                    "reasoning": "their own post",
                },
            }
        ),
    )
    monkeypatch.setattr(
        crawl,
        "discover_links",
        _links(
            {
                profile: [own_post, stranger],
                stranger: [deep_behind_stranger],
            }
        ),
    )

    report = await crawl.crawl_connected_account(
        _context(),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
    )

    assert report["status"] == "completed"
    assert profile in _pipeline
    assert own_post in _pipeline
    # The unrelated page is judged and refused...
    assert stranger not in _pipeline
    assert any(entry["url"] == stranger for entry in report["pruned"])
    # ...and crucially, nothing behind it is ever visited.
    assert deep_behind_stranger not in report["visited"]
    assert deep_behind_stranger not in _pipeline


@pytest.mark.asyncio
async def test_a_relevant_page_that_is_not_worth_following_is_a_leaf(
    monkeypatch, _pipeline
):
    """Being the person's own content does not by itself justify descending."""
    profile = "https://www.instagram.com/evan/"
    child = "https://www.instagram.com/p/child/"
    monkeypatch.setattr(
        crawl,
        "judge_relevance",
        _judge(
            {
                profile: {
                    "is_about_target": True,
                    "is_targets_own_words": True,
                    "relevance": 1.0,
                    "worth_following": False,
                    "reasoning": "a single post",
                }
            }
        ),
    )
    monkeypatch.setattr(crawl, "discover_links", _links({profile: [child]}))

    report = await crawl.crawl_connected_account(
        _context(),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
    )
    assert _pipeline == [profile]
    assert child not in report["visited"]


@pytest.mark.asyncio
async def test_a_weak_relevance_score_is_pruned(monkeypatch, _pipeline):
    """Scoring below the floor prunes even when the model says it is about them."""
    profile = "https://www.instagram.com/evan/"
    monkeypatch.setattr(
        crawl,
        "judge_relevance",
        _judge(
            {
                profile: {
                    "is_about_target": True,
                    "is_targets_own_words": False,
                    "relevance": 0.2,
                    "worth_following": True,
                    "reasoning": "a passing mention",
                }
            }
        ),
    )
    monkeypatch.setattr(crawl, "discover_links", _links({}))
    report = await crawl.crawl_connected_account(
        _context(social_crawl_relevance_minimum_score=0.5),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
    )
    assert _pipeline == []
    assert report["pruned"]


@pytest.mark.asyncio
async def test_the_free_tier_cap_bounds_what_one_crawl_spends(monkeypatch, _pipeline):
    """Every item costs a transcription, so the cap is a spending limit."""
    urls = [f"https://www.instagram.com/p/{index}/" for index in range(10)]
    profile = "https://www.instagram.com/evan/"
    relevant = {
        url: {
            "is_about_target": True,
            "is_targets_own_words": True,
            "relevance": 1.0,
            "worth_following": True,
            "reasoning": "theirs",
        }
        for url in [profile, *urls]
    }
    monkeypatch.setattr(crawl, "judge_relevance", _judge(relevant))
    monkeypatch.setattr(crawl, "discover_links", _links({profile: urls}))

    await crawl.crawl_connected_account(
        _context(social_crawl_max_items_free=3),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="free",
    )
    assert len(_pipeline) <= 3


@pytest.mark.asyncio
async def test_a_billing_refusal_ends_the_crawl_rather_than_repeating(monkeypatch):
    """One allotment refusal means every later item would be refused too."""
    calls = []

    async def _refusing_ingest(**kwargs):
        calls.append(kwargs["url"])
        return {"status": "refused", "detail": "monthly allotment reached"}

    monkeypatch.setattr(
        "src.anubis.utils.subscriptions.intake.ingest_content_url", _refusing_ingest
    )
    profile = "https://www.instagram.com/evan/"
    others = [f"https://www.instagram.com/p/{index}/" for index in range(5)]
    monkeypatch.setattr(
        crawl,
        "judge_relevance",
        _judge(
            {
                url: {
                    "is_about_target": True,
                    "is_targets_own_words": True,
                    "relevance": 1.0,
                    "worth_following": True,
                    "reasoning": "theirs",
                }
                for url in [profile, *others]
            }
        ),
    )
    monkeypatch.setattr(crawl, "discover_links", _links({profile: others}))

    report = await crawl.crawl_connected_account(
        _context(),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
    )
    assert report["status"] == "budget_reached"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_resumed_pull_skips_what_the_first_pass_already_took(
    monkeypatch, _pipeline
):
    """"Pull more" must not pay twice for the same pages."""
    profile = "https://www.instagram.com/evan/"
    monkeypatch.setattr(
        crawl,
        "judge_relevance",
        _judge(
            {
                profile: {
                    "is_about_target": True,
                    "is_targets_own_words": True,
                    "relevance": 1.0,
                    "worth_following": False,
                    "reasoning": "theirs",
                }
            }
        ),
    )
    monkeypatch.setattr(crawl, "discover_links", _links({}))
    report = await crawl.crawl_connected_account(
        _context(),
        record=_proven_record(),
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        avatar_name="Evan",
        avatar_description=None,
        tier_name="premium",
        already_seen={profile},
    )
    assert _pipeline == []
    assert report["status"] == "nothing_to_crawl"


def test_a_youtube_seed_prefers_the_uploads_playlist():
    """The uploads playlist is canonical; a handle-built address is a guess."""
    record = _proven_record(
        provider="youtube",
        transport={"youtube_uploads_playlist_id": "UU123"},
        ownership=ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN,
            handle="evan",
            profile_url="https://www.youtube.com/@evan",
        ),
    )
    seeds = crawl.seed_urls_for(record)
    assert seeds[0] == "https://www.youtube.com/playlist?list=UU123"


@pytest.mark.asyncio
async def test_an_unreadable_page_is_pruned_not_ingested(monkeypatch):
    """Failing open toward ingesting would spend money on an unconfirmable page."""

    async def _no_text(url):
        return ""

    monkeypatch.setattr(crawl, "_read_page_text", _no_text)
    verdict = await crawl.judge_relevance(
        _context(),
        url="https://example.com/x",
        avatar_name="Evan",
        avatar_description=None,
    )
    assert verdict["is_about_target"] is False
    assert verdict["worth_following"] is False


def test_navigation_and_store_links_are_never_followed():
    """Distribution links are not the person's work and cost a judgement each."""
    assert "apps.apple.com" in crawl._NEVER_FOLLOW_HOSTS
    assert "accounts.google.com" in crawl._NEVER_FOLLOW_HOSTS
