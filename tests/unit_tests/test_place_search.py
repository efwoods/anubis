"""Finding a real, named place so a plan can become an appointment.

The avatar could already read the owner's free time and write to the owner's
calendar. What it could not do was answer "where" — a plan agreed as "a quiet
museum, then somewhere near calm water" stayed a description, and an appointment
whose location reads "Museum — TBC" is not one. What is pinned down:

- a place is only ever offered if the search actually returned it;
- an address is read off the page or left empty, never completed or guessed,
  because a confident wrong address puts somebody outside the wrong building;
- a review site or a listing page is not a place, even though it names places;
- a search failure answers "no places found", never an invented one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.anubis.utils.scheduling.place_search as place_search
from src.anubis.utils.scheduling.place_search import (
    CandidatePlace,
    address_from_text,
    build_place_tools,
    find_places,
    is_directory_page,
    place_name_from_title,
    places_from_results,
)


def _result(url: str, title: str, snippet: str = "", content: str = ""):
    return SimpleNamespace(url=url, title=title, snippet=snippet, content=content)


def test_a_venue_name_is_the_name_not_the_page_title() -> None:
    # Titles are written for search engines, not for people.
    assert (
        place_name_from_title("The Frick Collection | New York's Finest Art Museum")
        == "The Frick Collection"
    )
    assert place_name_from_title("Kelvingrove Art Gallery - Visit Glasgow") == (
        "Kelvingrove Art Gallery"
    )
    assert place_name_from_title("Dia Beacon (Beacon, NY)") == "Dia Beacon"
    assert place_name_from_title("") == ""


def test_an_address_is_read_or_left_empty_never_invented() -> None:
    # A street type makes an address unambiguous on its own.
    assert (
        address_from_text("Open daily. Located at 945 Madison Avenue, New York.")
        == "945 Madison Avenue, New York"
    )
    # Plenty of the world writes no street type, so a phrase announcing an
    # address plus a following town is accepted too.
    assert (
        address_from_text("A quiet collection at 18 Museumpark, Rotterdam.")
        == "18 Museumpark, Rotterdam"
    )
    assert (
        address_from_text("The garden is at 9-01 33rd Road, Long Island City.")
        == "9-01 33rd Road, Long Island City"
    )
    # And the false positives that rule would otherwise let in.
    assert address_from_text("10 Best Museums you must visit") == ""
    assert address_from_text("Open at 9 Monday through Friday.") == ""
    assert address_from_text("A lovely quiet museum by the water.") == ""
    assert address_from_text("") == ""


def test_a_listing_site_is_not_a_place() -> None:
    assert is_directory_page("https://www.tripadvisor.com/Attractions-museums")
    assert is_directory_page("https://en.wikipedia.org/wiki/List_of_museums")
    assert not is_directory_page("https://www.frick.org/visit")


def test_only_real_venues_are_offered() -> None:
    places = places_from_results(
        [
            _result("https://www.tripadvisor.com/best-museums", "10 Best Museums"),
            _result(
                "https://www.frick.org/visit",
                "The Frick Collection | Art Museum",
                "Open Wednesday to Sunday at 945 Madison Avenue, New York.",
            ),
            _result(
                "https://www.noguchi.org/",
                "The Noguchi Museum",
                "A quiet garden museum at 9-01 33rd Road, Long Island City.",
            ),
        ],
        limit=3,
    )
    assert [place.name for place in places] == [
        "The Frick Collection",
        "The Noguchi Museum",
    ]
    assert places[0].address == "945 Madison Avenue, New York"
    assert places[0].url == "https://www.frick.org/visit"


def test_the_same_venue_is_never_offered_twice() -> None:
    places = places_from_results(
        [
            _result("https://www.frick.org/visit", "The Frick Collection | Museum"),
            _result("https://www.frick.org/exhibitions", "The Frick Collection | Now"),
        ]
    )
    assert len(places) == 1


def test_a_place_with_no_address_still_comes_back_honestly() -> None:
    (place,) = places_from_results(
        [_result("https://www.example-museum.org/", "The Example Museum", "Quiet.")]
    )
    assert place.name == "The Example Museum"
    # Empty rather than plausible: the prompt tells the avatar to say so.
    assert place.address == ""


@pytest.mark.asyncio
async def test_where_in_the_world_is_never_guessed(monkeypatch) -> None:
    async def _must_not_search(*_args, **_kwargs):
        raise AssertionError("a city must not be chosen on the owner's behalf")

    monkeypatch.setattr(place_search, "find_places", find_places)
    import src.anubis.utils.research.web_search as web_search

    monkeypatch.setattr(web_search, "search_web", _must_not_search)
    assert await find_places("a quiet museum", "", context=object()) == []
    assert await find_places("", "Rotterdam", context=object()) == []


@pytest.mark.asyncio
async def test_a_search_failure_answers_plainly(monkeypatch) -> None:
    async def _failing(*_args, **_kwargs):
        raise RuntimeError("the search provider is down")

    import src.anubis.utils.research.web_search as web_search

    monkeypatch.setattr(web_search, "search_web", _failing)
    assert await find_places("a quiet museum", "Rotterdam", context=object()) == []


@pytest.mark.asyncio
async def test_the_tool_reports_places_or_says_there_were_none(monkeypatch) -> None:
    import src.anubis.utils.research.web_search as web_search

    async def _search(query, *, limit, context=None):
        assert "quiet museum" in query and "Rotterdam" in query
        return [
            _result(
                "https://www.boijmans.nl/",
                "Museum Boijmans Van Beuningen",
                "A quiet collection at 18 Museumpark, Rotterdam.",
            )
        ]

    monkeypatch.setattr(web_search, "search_web", _search)
    (tool,) = build_place_tools(object())
    result = await tool.ainvoke(
        {"description": "quiet museum", "near": "Rotterdam", "limit": 3}
    )
    assert result["status"] == "ok"
    assert result["places"][0]["name"] == "Museum Boijmans Van Beuningen"
    assert result["places"][0]["address"] == "18 Museumpark, Rotterdam"

    async def _nothing(query, *, limit, context=None):
        return []

    monkeypatch.setattr(web_search, "search_web", _nothing)
    empty = await tool.ainvoke({"description": "quiet museum", "near": "Rotterdam"})
    assert empty["status"] == "no_places_found"
    assert "No specific places" in empty["message"]


def test_the_candidate_is_reported_whole() -> None:
    place = CandidatePlace(
        name="The Noguchi Museum",
        address="9-01 33rd Road",
        url="https://www.noguchi.org/",
        why="A quiet garden museum.",
    )
    assert place.as_dict() == {
        "name": "The Noguchi Museum",
        "address": "9-01 33rd Road",
        "url": "https://www.noguchi.org/",
        "why": "A quiet garden museum.",
    }
