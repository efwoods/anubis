"""The initial pull: walk what a proven account has published, breadth-first.

Connecting an account is a promise that the avatar will know what this person
has already said, not only what they say next. This module keeps that promise
by walking outward from the account's own profile — newest first, one level at
a time — and ingesting what it finds through the ordinary media pipeline.

**Why breadth-first, and why it prunes.** A person's profile links to their
posts, and their posts link to the whole internet. A depth-first walk spends
its budget on whatever the first link happened to be; a breadth-first walk
spends it on the things closest to the person, which is where their own
material actually is. And because every level multiplies, the walk has to stop
descending branches that are not about this person at all: a page that is not
theirs is not merely uninteresting, its links lead further away, so the subtree
is dropped rather than explored. That pruning rule is the difference between
crawling a channel and crawling the web.

**Every medium is collected individually.** A video, a photograph, a recording
and a written post are four different kinds of evidence about one person, and
the pipeline already knows how to turn each into text — diarized transcript,
image description, article body. The crawl's job is to hand each item over
separately, so each is classified, analyzed and indexed on its own terms,
rather than flattening a page into one blob.

**The budget is real money.** Every item is transcribed or described by a
model, so the walk is bounded three ways at once: a per-tier item cap, a depth
limit, and a node ceiling. The caps live in configuration rather than in code
because the right number depends on what the owner is paying.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from src.anubis.utils.connected_accounts.ownership import (
    is_owned_by_personal_avatar,
    ownership_of,
    refusal_reason,
)
from src.anubis.utils.connected_accounts.providers import get_provider

logger = logging.getLogger(__name__)

# Hosts whose links are never worth following out of a crawled page: they are
# navigation and distribution, not the person's own work. Listing them saves a
# relevance call each, which is the cheapest possible pruning.
_NEVER_FOLLOW_HOSTS = frozenset(
    {
        "accounts.google.com",
        "policies.google.com",
        "support.google.com",
        "www.google.com",
        "play.google.com",
        "apps.apple.com",
        "itunes.apple.com",
        "www.apple.com",
        "consent.youtube.com",
        "about.instagram.com",
        "help.instagram.com",
        "privacy.twitter.com",
        "help.twitter.com",
        "t.co",
        "bit.ly",
        "amazon.com",
        "www.amazon.com",
        "patreon.com",
        "www.patreon.com",
    }
)


class CrawlBudget:
    """What one crawl is allowed to spend, and what it has spent so far.

    Kept as an object rather than as loose counters because every stopping rule
    has to be checked in several places — before expanding a level, before
    ingesting an item, before following a link — and a budget that is easy to
    consult is a budget that actually gets consulted.
    """

    def __init__(self, *, max_items: int, max_depth: int, max_nodes: int) -> None:
        """Bound one crawl by items ingested, depth reached, and nodes visited."""
        self.max_items = max(0, int(max_items))
        self.max_depth = max(0, int(max_depth))
        self.max_nodes = max(0, int(max_nodes))
        self.items_ingested = 0
        self.nodes_visited = 0

    @property
    def exhausted(self) -> bool:
        """Whether the crawl must stop now."""
        return (
            self.items_ingested >= self.max_items
            or self.nodes_visited >= self.max_nodes
        )

    def remaining_items(self) -> int:
        """How many more items may still be ingested."""
        return max(0, self.max_items - self.items_ingested)


def crawl_budget_for(context: Any, tier_name: str) -> CrawlBudget:
    """Return the budget this owner's tier allows.

    Read from configuration per tier, because the cost of a crawl is real and
    an owner on the free tier must not be able to start a hundred-video
    transcription run by connecting a channel.
    """
    per_tier = {
        "free": getattr(context, "social_crawl_max_items_free", None),
        "pro": getattr(context, "social_crawl_max_items_pro", None),
        "premium": getattr(context, "social_crawl_max_items_premium", None),
    }
    default_by_tier = {"free": 5, "pro": 50, "premium": 300}
    raw = per_tier.get(tier_name)
    try:
        max_items = int(raw) if raw not in (None, "") else default_by_tier.get(tier_name, 5)
    except (TypeError, ValueError):
        max_items = default_by_tier.get(tier_name, 5)
    return CrawlBudget(
        max_items=max_items,
        max_depth=_int_setting(context, "social_crawl_max_depth", 3),
        max_nodes=_int_setting(context, "social_crawl_max_nodes", 200),
    )


def _int_setting(context: Any, name: str, fallback: int) -> int:
    try:
        value = getattr(context, name, None)
        return int(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def _float_setting(context: Any, name: str, fallback: float) -> float:
    try:
        value = getattr(context, name, None)
        return float(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def seed_urls_for(record: dict[str, Any]) -> list[str]:
    """Return where the crawl starts for one proven account.

    Prefers what the account itself told us at connect time — YouTube's uploads
    playlist, read from the API — over a profile address assembled from a
    handle, because the former is the canonical list of the person's work and
    the latter is a guess that a platform can change the shape of.
    """
    provider = get_provider(str(record.get("provider") or ""))
    if provider is None:
        return []
    transport = record.get("transport") or {}
    ownership = ownership_of(record)
    seeds: list[str] = []

    uploads_playlist_id = str(transport.get("youtube_uploads_playlist_id") or "")
    if uploads_playlist_id:
        seeds.append(
            f"https://www.youtube.com/playlist?list={uploads_playlist_id}"
        )

    feed_url = str(transport.get("feed_url") or "")
    if feed_url:
        seeds.append(feed_url)

    profile_url = str(ownership.get("profile_url") or "")
    if profile_url:
        seeds.append(profile_url)

    site_url = str(transport.get("site_url") or "")
    if site_url and site_url not in seeds:
        seeds.append(site_url)

    handle = str(ownership.get("handle") or "")
    if handle and not seeds:
        built = provider.profile_url_for(handle)
        if built:
            seeds.append(built)

    # Preserve order while dropping repeats: the first seed is the best one.
    ordered: list[str] = []
    for url in seeds:
        if url and url not in ordered:
            ordered.append(url)
    return ordered


async def crawl_connected_account(
    context: Any,
    *,
    record: dict[str, Any],
    personal_avatar_id: str,
    user_id: str,
    avatar_name: str | None,
    avatar_description: str | None,
    tier_name: str,
    budget: CrawlBudget | None = None,
    already_seen: set[str] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Walk one proven account's published work and ingest what belongs to the person.

    Returns a report rather than raising. ``already_seen`` lets a later
    "pull more" resume without revisiting what the first pass covered.
    """
    refusal = refusal_reason(record, personal_avatar_id=personal_avatar_id)
    if refusal or not is_owned_by_personal_avatar(
        record, personal_avatar_id=personal_avatar_id
    ):
        return {
            "status": "refused",
            "detail": refusal or "This account is not proven to be yours.",
            "ingested": 0,
        }

    budget = budget or crawl_budget_for(context, tier_name)
    seen: set[str] = set(already_seen or set())
    frontier = [url for url in seed_urls_for(record) if url not in seen]
    if not frontier:
        return {
            "status": "nothing_to_crawl",
            "detail": (
                "This account did not say where its published work lives, so "
                "there is nothing to walk."
            ),
            "ingested": 0,
        }

    ingested_urls: list[str] = []
    pruned: list[dict[str, str]] = []
    depth = 0

    while frontier and depth < budget.max_depth and not budget.exhausted:
        level = frontier[: budget.remaining_items() or None]
        frontier = []
        for url in level:
            if budget.exhausted:
                break
            if url in seen:
                continue
            seen.add(url)
            budget.nodes_visited += 1

            verdict = await judge_relevance(
                context,
                url=url,
                avatar_name=avatar_name,
                avatar_description=avatar_description,
            )
            minimum = _float_setting(
                context, "social_crawl_relevance_minimum_score", 0.5
            )
            if not verdict.get("is_about_target") or float(
                verdict.get("relevance") or 0.0
            ) < minimum:
                # The pruning rule: a page that is not this person's is a dead
                # branch, and its links are dropped rather than queued.
                pruned.append(
                    {"url": url, "reason": str(verdict.get("reasoning") or "")[:200]}
                )
                continue

            outcome = await _ingest_one(
                url=url,
                user_id=user_id,
                personal_avatar_id=personal_avatar_id,
                avatar_name=avatar_name,
                avatar_description=avatar_description,
            )
            if outcome.get("status") in {"started", "accepted", "ok"}:
                budget.items_ingested += 1
                ingested_urls.append(url)
            elif outcome.get("status") == "refused":
                # An allotment or tier refusal is the end of the crawl, not of
                # this one item: continuing would repeat the same refusal for
                # every remaining node and waste the relevance calls doing it.
                return _report(
                    "budget_reached",
                    ingested_urls,
                    pruned,
                    seen,
                    detail=str(outcome.get("detail") or ""),
                )

            if depth + 1 < budget.max_depth and verdict.get("worth_following"):
                for child in await discover_links(url):
                    if child not in seen:
                        frontier.append(child)
        depth += 1

    return _report("completed", ingested_urls, pruned, seen)


def _report(
    status: str,
    ingested_urls: list[str],
    pruned: list[dict[str, str]],
    seen: set[str],
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    report = {
        "status": status,
        "ingested": len(ingested_urls),
        "ingested_urls": ingested_urls,
        "pruned": pruned,
        "visited": sorted(seen),
    }
    if detail:
        report["detail"] = detail
    return report


async def _ingest_one(
    *,
    url: str,
    user_id: str,
    personal_avatar_id: str,
    avatar_name: str | None,
    avatar_description: str | None,
) -> dict[str, Any]:
    """Hand one address to the metered media pipeline."""
    from src.anubis.utils.subscriptions.intake import ingest_content_url

    return await ingest_content_url(
        user_id=user_id,
        personal_avatar_id=personal_avatar_id,
        url=url,
        avatar_name=avatar_name,
        avatar_description=avatar_description,
    )


async def judge_relevance(
    context: Any,
    *,
    url: str,
    avatar_name: str | None,
    avatar_description: str | None,
    page_text: str | None = None,
) -> dict[str, Any]:
    """Decide whether one address holds content of or by the avatar's person.

    Fails **open toward pruning**, not toward ingesting: when the page cannot be
    read or the model cannot answer, the branch is dropped. Ingesting on an
    unreadable page would spend real money on something nobody could confirm is
    the right person, which is the expensive direction to be wrong in.
    """
    if not avatar_name:
        # Without a name there is nothing to match against, so the only honest
        # answer is to take the seeds and follow nothing.
        return {
            "is_about_target": True,
            "is_targets_own_words": False,
            "relevance": 1.0,
            "worth_following": False,
            "reasoning": "No avatar name was available to judge against.",
        }

    text = page_text
    if text is None:
        text = await _read_page_text(url)
    if not text:
        return {
            "is_about_target": False,
            "is_targets_own_words": False,
            "relevance": 0.0,
            "worth_following": False,
            "reasoning": f"{url} could not be read.",
        }

    from src.anubis.utils.model import init_model
    from src.anubis.utils.schema import (
        PERSONAL_AVATAR_RELEVANCE_SYSTEM_PROMPT,
        PersonalAvatarRelevance,
    )

    model = init_model(
        model_without_tools=True, response_format=PersonalAvatarRelevance
    )
    described = f" ({avatar_description})" if avatar_description else ""
    prompt = (
        f"The person is {avatar_name}{described}.\n\n"
        f"The address is {url}.\n\n"
        f"The content follows.\n\n{text[:12000]}"
    )
    try:
        answer = await model.ainvoke(
            [
                {"role": "system", "content": PERSONAL_AVATAR_RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as judge_error:  # noqa: BLE001 - prune on failure
        logger.warning("Relevance judgement failed for %s: %s", url, judge_error)
        return {
            "is_about_target": False,
            "is_targets_own_words": False,
            "relevance": 0.0,
            "worth_following": False,
            "reasoning": f"The relevance check failed: {judge_error}",
        }

    parsed = getattr(answer, "parsed", None) or answer
    if hasattr(parsed, "model_dump"):
        return dict(parsed.model_dump())
    if isinstance(parsed, dict):
        return parsed
    return {
        "is_about_target": False,
        "is_targets_own_words": False,
        "relevance": 0.0,
        "worth_following": False,
        "reasoning": "The relevance check returned nothing usable.",
    }


async def _read_page_text(url: str) -> str:
    """Read a page as visible text, reusing the loader the pipeline already has."""
    try:
        from src.anubis.utils.classes.URLDocumentLoaderClass import (
            _httpx_fallback_text,
        )

        return await _httpx_fallback_text(url)
    except Exception as read_error:  # noqa: BLE001 - an unread page prunes
        logger.info("Could not read %s for a relevance check: %s", url, read_error)
        return ""


async def discover_links(url: str) -> list[str]:
    """Return the outbound links of a page, minus the ones never worth following.

    Deliberately the same shape as the Linktree expansion the media pipeline
    already performs, because a profile page full of links to a person's own
    work is the same thing a link-in-bio page is.
    """
    try:
        from src.anubis.utils.classes.URLDocumentLoaderClass import (
            _httpx_fallback_text,
        )

        html = await _httpx_fallback_text(url, return_html=True)
    except Exception as fetch_error:  # noqa: BLE001
        logger.info("Could not expand %s: %s", url, fetch_error)
        return []
    if not html:
        return []

    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover - dependency is installed
        return []

    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        if not href.startswith("http"):
            continue
        try:
            host = (urlparse(href).hostname or "").lower()
        except Exception:  # noqa: BLE001
            continue
        if not host or host in _NEVER_FOLLOW_HOSTS:
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
    return links


async def crawl_accounts_for_avatar(
    context: Any,
    *,
    records: list[dict[str, Any]],
    personal_avatar_id: str,
    user_id: str,
    avatar_name: str | None,
    avatar_description: str | None,
    tier_name: str,
    store: Any = None,
) -> list[dict[str, Any]]:
    """Crawl every proven account of one avatar, one after another.

    Sequential on purpose. The accounts share one budget in spirit and one
    Stripe meter in fact, and running them concurrently would race the
    allotment check so that two crawls each believe there is room for the work
    the other is about to do.
    """
    reports: list[dict[str, Any]] = []
    for record in records:
        if not is_owned_by_personal_avatar(
            record, personal_avatar_id=personal_avatar_id
        ):
            continue
        report = await crawl_connected_account(
            context,
            record=record,
            personal_avatar_id=personal_avatar_id,
            user_id=user_id,
            avatar_name=avatar_name,
            avatar_description=avatar_description,
            tier_name=tier_name,
            store=store,
        )
        reports.append({"account_key": record.get("account_key"), **report})
        await asyncio.sleep(0)
    return reports
