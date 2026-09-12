"""Finding real, named places on the web so a plan can become an appointment.

The avatar could already read the owner's free time and write an appointment to
the owner's calendar. What it could not do was answer "where": a plan agreed as
"a quiet museum, then somewhere near calm water" stayed a description, and an
appointment whose location reads "Museum — TBC" is not an appointment. This
module closes that gap, and it is the last piece of the loop:

    the owner asks for a plan
        -> the calendar says when the owner is free
        -> the web says which real places match what was described
        -> the avatar proposes ONE time and ONE named place
        -> the owner agrees
        -> the appointment is written, with the venue's real address on it

Searching is the existing research search (``research/web_search.py``), reused
rather than rebuilt: the same providers, the same merging, the same failure
behaviour. What is added here is reading a search result as a PLACE — a name, an
address, and the page the address came from — and refusing to invent any of the
three. A place the search did not actually return is never offered, because the
cost of a confident wrong address is the owner standing outside the wrong
building.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PLACE_RESULT_LIMIT = 8
DEFAULT_PLACES_OFFERED = 3

# Hosts that aggregate or review places rather than being one. A result from one
# of these is still useful — the page names real venues — but the page itself is
# not the place, so its title must never be offered as a venue name.
DIRECTORY_HOSTS = frozenset(
    {
        "tripadvisor.com",
        "yelp.com",
        "timeout.com",
        "reddit.com",
        "wikipedia.org",
        "facebook.com",
        "eventbrite.com",
    }
)

# Reading an address off a page is the one place here that can quietly be wrong,
# so it is split into two rules with different burdens of proof.
#
# The first needs no context because it is unambiguous on its own: a number, some
# words, and a street type. The second has no street type — plenty of the world
# writes "18 Museumpark" — so it is trusted ONLY directly after a phrase that
# says an address is coming, and only when a town follows the street. Without
# that second rule every non-English address comes back empty; without its
# conditions, "10 Best Museums" reads as a street address.
_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl"
    r"|Square|Sq|Park|Parkway|Pkwy|Terrace|Court|Ct|Highway|Hwy|Quay|Wharf"
    r"|Strasse|Stra\u00dfe|Rue|Via|Plaza|Piazza|Gracht|Kade|Laan)"
)
_HOUSE_NUMBER = r"\d{1,5}(?:-\d{1,5})?"
_STREET_WORD = r"(?:[A-Z][\w'.\-]*|\d{1,3}(?:st|nd|rd|th))"
_TOWN = r"[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*){0,3}"

_ADDRESS_WITH_STREET_TYPE = re.compile(
    rf"{_HOUSE_NUMBER}\s+{_STREET_WORD}(?:\s+{_STREET_WORD}){{0,4}}\s+"
    rf"{_STREET_SUFFIX}\b(?:,\s*{_TOWN})?"
)

_ADDRESS_AFTER_CUE = re.compile(
    rf"(?:located at|situated at|address(?:\s+is)?:?|find us at|at)\s+"
    rf"({_HOUSE_NUMBER}\s+{_STREET_WORD}(?:\s+{_STREET_WORD}){{0,3}},\s*{_TOWN})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidatePlace:
    """One real place the web search actually returned.

    ``address`` is empty when no address could be read off the page rather than
    guessed at. An empty address is reported honestly to the owner — "I could not
    find the address" — because a plausible invented one is worse than none.
    """

    name: str
    address: str
    url: str
    why: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the tool-result form of this place."""
        return {
            "name": self.name,
            "address": self.address,
            "url": self.url,
            "why": self.why,
        }


def _hostname_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(str(url or "").strip()).hostname or "").lower()
    except Exception:  # noqa: BLE001 - a malformed address names nothing
        return ""


def _registrable_domain(hostname: str) -> str:
    labels = [label for label in (hostname or "").split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else (hostname or "")


def is_directory_page(url: str) -> bool:
    """Whether this result reviews or lists places rather than being one."""
    return _registrable_domain(_hostname_of(url)) in DIRECTORY_HOSTS


def place_name_from_title(title: str) -> str:
    """Return the venue's name from a page title, or an empty string.

    Page titles are written for search engines, not for people: "The Frick
    Collection | New York City's Finest Art Museum | Visit Today". The venue is
    almost always the first segment, so the tail is dropped rather than offered
    to the owner as part of the name.
    """
    cleaned = str(title or "").strip()
    if not cleaned:
        return ""
    for separator in ("|", " - ", " – ", " — ", " :: "):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip()
            break
    # A trailing city qualifier reads as part of the name and is not.
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()
    return cleaned


def address_from_text(text: str) -> str:
    """Return the first street address in ``text``, or an empty string.

    Deliberately conservative. A missed address costs a sentence saying the
    address was not found; an invented one sends the owner to the wrong place.
    """
    text = str(text or "")
    match = _ADDRESS_WITH_STREET_TYPE.search(text)
    found = match.group(0) if match else ""
    if not found:
        cued = _ADDRESS_AFTER_CUE.search(text)
        found = cued.group(1) if cued else ""
    # A town may legitimately carry a period ("St. Louis"), so only trailing
    # sentence punctuation is trimmed.
    return found.strip().rstrip(" ,.;")


def places_from_results(
    results: list[Any], *, limit: int = DEFAULT_PLACES_OFFERED
) -> list[CandidatePlace]:
    """Read search results as places, keeping only those that name a real venue."""
    places: list[CandidatePlace] = []
    seen_names: set[str] = set()
    for result in results or []:
        url = str(getattr(result, "url", "") or "")
        if not url or is_directory_page(url):
            continue
        name = place_name_from_title(getattr(result, "title", "") or "")
        if not name or name.casefold() in seen_names:
            continue
        body = " ".join(
            part
            for part in (
                getattr(result, "snippet", "") or "",
                getattr(result, "content", "") or "",
            )
            if part
        )
        seen_names.add(name.casefold())
        places.append(
            CandidatePlace(
                name=name,
                address=address_from_text(body),
                url=url,
                why=(getattr(result, "snippet", "") or "")[:240],
            )
        )
        if len(places) >= limit:
            break
    return places


async def find_places(
    description: str,
    near: str,
    *,
    context: Any = None,
    limit: int = DEFAULT_PLACES_OFFERED,
) -> list[CandidatePlace]:
    """Return real, named places matching ``description`` around ``near``.

    ``near`` is required and is not guessed at. "A quiet museum" is answerable
    anywhere on earth, so without a place to search around the honest answer is
    to ask the owner where, not to pick a city on the owner's behalf.
    """
    from src.anubis.utils.context import GlobalContext
    from src.anubis.utils.research.web_search import search_web

    description = str(description or "").strip()
    near = str(near or "").strip()
    if not description or not near:
        return []
    context = context or GlobalContext()
    query = f"{description} in {near}"
    try:
        results = await search_web(
            query, limit=DEFAULT_PLACE_RESULT_LIMIT, context=context
        )
    except Exception as search_error:  # noqa: BLE001 - reported, never raised
        logger.warning("Place search failed for %r: %s", query, search_error)
        return []
    return places_from_results(results, limit=limit)


def build_place_tools(context: Any) -> list[Any]:
    """Return the place-finding tool, for an avatar that may schedule."""
    from langchain_core.tools import tool

    @tool
    async def find_places_to_go(
        description: str, near: str, limit: int = DEFAULT_PLACES_OFFERED
    ) -> dict[str, Any]:
        """Find real, named places on the web that match a described outing.

        Call this whenever a plan needs somewhere to happen and no specific venue
        has been named yet — "a quiet museum", "somewhere near calm water", "a
        good coffee shop to talk in". The result holds real places with real
        addresses, which is what an appointment needs; a plan whose location
        reads "museum to be decided" is not something the conversation partner
        can turn up to.

        Args:
            description: What kind of place, in the conversation partner's own
                words. Pass "quiet museum" rather than "museum" when the
                conversation partner said quiet; the word changes the answer.
            near: The town, city, or neighbourhood to search around. Required.
                When the conversation partner has not said where, ask them
                rather than choosing a city for them.
            limit: How many places to bring back. Three is usually plenty.

        Never present a place this tool did not return, and never invent or
        complete an address. When a place comes back with no address, say that
        the address still needs checking rather than supplying a likely one.
        """
        places = await find_places(description, near, context=context, limit=limit)
        if not places:
            return {
                "status": "no_places_found",
                "description": description,
                "near": near,
                "message": (
                    f"No specific places matching {description!r} near {near!r} came "
                    "back. Say so plainly, and either try a different description or "
                    "ask the conversation partner to name the area more precisely."
                ),
            }
        return {
            "status": "ok",
            "description": description,
            "near": near,
            "places": [place.as_dict() for place in places],
        }

    return [find_places_to_go]


__all__ = [
    "DEFAULT_PLACES_OFFERED",
    "DIRECTORY_HOSTS",
    "CandidatePlace",
    "address_from_text",
    "build_place_tools",
    "find_places",
    "is_directory_page",
    "place_name_from_title",
    "places_from_results",
]
