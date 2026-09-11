"""Deep research hands its verified media to the media pipeline, under a cap.

Research used to read a source page's text and nothing more: a video it verified
a fact from contributed only whatever sentences the page exposed, was never
transcribed, and never appeared in the avatar's uploaded material. The sources
behind VERIFIED facts now go through the same pipeline an uploaded link takes.

Transcription is the expensive step, so one run can only send
``DEEP_RESEARCH_MAX_MEDIA_ITEMS`` sources; the sources that supported the most
verified facts are the ones kept when the cap bites.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.research.deep_research import verified_source_urls
from src.anubis.utils.tools.identity.identity_tools import (
    _research_proposal_preview,
    review_researched_facts,
)


def _fact(*urls: str) -> dict:
    return {"supporting_source_urls": list(urls)}


""" Which media a run sends through the pipeline """


def test_only_sources_behind_verified_facts_are_sent():
    assert verified_source_urls([], limit=10) == []
    assert verified_source_urls([_fact()], limit=10) == []
    assert verified_source_urls([_fact("https://a")], limit=10) == ["https://a"]


def test_the_most_corroborating_sources_survive_the_cap():
    facts = [
        _fact("https://a", "https://b"),
        _fact("https://b"),
        _fact("https://b", "https://c"),
    ]
    # b supported three facts, c one, a one; ties break by url so the order is stable.
    assert verified_source_urls(facts, limit=3) == [
        "https://b",
        "https://a",
        "https://c",
    ]
    assert verified_source_urls(facts, limit=1) == ["https://b"]


def test_a_zero_cap_switches_the_media_hand_off_off():
    assert verified_source_urls([_fact("https://a")], limit=0) == []
    assert verified_source_urls([_fact("https://a")], limit=-1) == []


def test_blank_and_repeated_urls_are_not_sent_twice():
    facts = [_fact("https://a", "", "  "), _fact("https://a")]
    assert verified_source_urls(facts, limit=10) == ["https://a"]


def test_the_cap_has_a_configured_default():
    from src.anubis.utils.context import GlobalContext

    assert GlobalContext().deep_research_max_media_items == 24


""" Resolving a contradiction in the conversation """


def test_the_review_tool_is_offered_to_the_avatar():
    from src.anubis.utils.deep_agent import IDENTITY_TOOL_NAMES

    assert review_researched_facts.name == "review_researched_facts"
    # Accepting a researched fact changes the identity, so the tool must also
    # trigger the consciousness refresh the other identity tools trigger.
    assert "review_researched_facts" in IDENTITY_TOOL_NAMES


def test_a_contradiction_renders_in_the_correction_panel_shape():
    """The owner resolves these with the control they already know from chat."""
    preview = _research_proposal_preview(
        0,
        {
            "fact_id": "fact-1",
            "fact": "I was born in Ottawa.",
            "existing_fact": "I was born in Toronto.",
            "fact_context": "Place of birth.",
            "reasoning": "Two sources say Ottawa.",
            "conflicting_statements": ["born in Ottawa"],
            "supporting_source_urls": ["https://example.com/a"],
            "verification_status": "inconsistent",
        },
    )
    # The fields the existing panel reads.
    for field in (
        "index",
        "key",
        "current_fact_content",
        "current_fact_context",
        "document_excerpt",
        "suggested_edit_fact_content",
        "default_action",
        "recommended_action",
    ):
        assert field in preview, field
    assert preview["current_fact_content"] == "I was born in Toronto."
    assert preview["has_stored_fact"] is True
    assert preview["suggested_edit_fact_content"] == "I was born in Ottawa."
    # Nobody but the owner can say which version is true, so nothing is
    # pre-selected and a dismissed panel changes nothing.
    assert preview["recommended_action"] == "skip"
    assert preview["default_action"] == "skip"
    assert "Two sources say Ottawa." in preview["document_excerpt"]
    assert "https://example.com/a" in preview["document_excerpt"]


def test_a_proposal_with_nothing_stored_still_reads_sensibly():
    preview = _research_proposal_preview(1, {"fact_id": "f", "fact": "I sail."})
    assert preview["current_fact_content"] == "(nothing stored yet on this point)"
    assert preview["suggested_edit_fact_content"] == "I sail."
    # The panel words a contradiction against a stored fact differently from a
    # contradiction among the sources alone, so it is told which this is.
    assert preview["has_stored_fact"] is False


""" What the acquisition already ingested is not ingested a second time """


@pytest.mark.asyncio
async def test_a_recording_the_acquisition_took_is_not_transcribed_twice(monkeypatch):
    """The chosen video reaches the media pipeline once, not once per path.

    The acquisition hands its winning video straight to the media pipeline, and
    that same video is very often also one of the pages a verified fact came
    from. Transcribing it twice would download, diarize and transcribe the whole
    recording a second time, which is the single most expensive thing this
    pipeline does.
    """
    from types import SimpleNamespace

    from langgraph.store.memory import InMemoryStore

    from src.anubis.utils.research import deep_research

    chosen_video = "https://www.youtube.com/watch?v=abcdefghijk"
    other_page = "https://example.com/a-profile"

    async def _no_facts(*args, **kwargs):
        return []

    async def _brief(*args, **kwargs):
        return deep_research.ResearchBrief(
            subject_summary="A mathematician.", open_questions=[], topics=[]
        )

    async def _bootstrap(*args, **kwargs):
        return {"media_urls": [chosen_video]}

    monkeypatch.setattr(deep_research, "load_existing_identity_facts", _no_facts)
    monkeypatch.setattr(deep_research, "build_research_brief", _brief)
    monkeypatch.setattr(
        deep_research,
        "verified_source_urls",
        lambda to_apply, limit: [chosen_video, other_page],
    )
    monkeypatch.setattr(
        "src.anubis.utils.research.asset_bootstrap.run_asset_bootstrap", _bootstrap
    )

    summary = await deep_research.run_deep_research(
        InMemoryStore(),
        SimpleNamespace(
            deep_research_max_queries=1,
            deep_research_max_sources=1,
            deep_research_max_topics=1,
            deep_research_concurrency=1,
            deep_research_follow_up_rounds=0,
            deep_research_max_media_items=24,
        ),
        creator_id="auth0|creator",
        assistant_id="assistant-1",
        subject_name="Ada Lovelace",
        subject_description=None,
        research_hint=None,
        emit=lambda payload: None,
        bootstrap=object(),
    )

    assert summary["bootstrap_media_urls"] == [chosen_video]
    # The page the acquisition did not take still goes through; the video does not.
    assert summary["media_source_urls"] == [other_page]
