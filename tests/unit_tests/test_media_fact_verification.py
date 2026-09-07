"""Facts learned from media are verified the way researched facts are.

Uploaded media used to be indexed and believed. Nothing compared a new claim
against what the avatar already held, so a transcript saying one thing sat
beside a stored fact saying another and the avatar held both.

The verification stages deep research uses — extract, cluster the claims that
make the same claim alongside the facts the avatar holds, judge each cluster,
apply what agrees and hold back what contradicts — now run over what a media
batch indexed. Contradictions land in the same store the researched ones use, so
the owner settles both through one review.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.research.deep_research import (
    MEDIA_FACT_SOURCE,
    RESEARCH_FACT_SOURCE,
    build_identity_document,
)
from src.anubis.utils.research.media_fact_verification import (
    media_documents_as_sources,
    verify_facts_from_media,
)
from src.api.media_jobs import _newest_indexed_documents


class _Context:
    media_fact_verification_enabled = "true"
    media_fact_verification_max_documents = 40
    deep_research_concurrency = 2


def _document(text: str, *, document_id: str = "d1", filename: str = "clip.mp4") -> dict:
    return {
        "page_content": text,
        "metadata": {"document_id": document_id, "filename": filename},
    }


""" Reading what the batch indexed """


def test_a_document_becomes_a_source_that_names_its_upload():
    sources = media_documents_as_sources([_document("She loved Yosemite.")], limit=10)
    assert len(sources) == 1
    # A verified fact can point back at the file it came from, the way a
    # researched fact points at a web page.
    assert sources[0].url == "media://d1"
    assert sources[0].title == "clip.mp4"
    assert sources[0].content == "She loved Yosemite."
    assert sources[0].provider == "media"


def test_empty_documents_are_not_sent_to_the_extractor():
    assert media_documents_as_sources([_document("   ")], limit=10) == []
    assert media_documents_as_sources([], limit=10) == []


def test_the_cap_bounds_what_one_batch_fact_checks():
    documents = [_document(f"fact {n}", document_id=str(n)) for n in range(10)]
    assert len(media_documents_as_sources(documents, limit=3)) == 3
    # Zero switches the verification off rather than checking everything.
    assert media_documents_as_sources(documents, limit=0) == []


def test_the_newest_documents_are_the_ones_this_batch_indexed():
    class _Item:
        def __init__(self, content, created_at):
            self.value = {
                "document": {
                    "kwargs": {
                        "page_content": content,
                        "metadata": {"created_at": created_at},
                    }
                }
            }

    documents = _newest_indexed_documents(
        [
            _Item("old", "2026-01-01"),
            _Item("newest", "2026-09-07"),
            _Item("middle", "2026-05-01"),
            {"malformed": True},
        ],
        limit=2,
    )
    assert [d["page_content"] for d in documents] == ["newest", "middle"]


""" A verified media fact says where it came from """


def test_a_media_fact_is_distinguishable_from_a_researched_one():
    verified = {"proposed_fact": "I loved Yosemite.", "status": "consistent"}
    from_media = build_identity_document(
        verified,
        creator_id="c",
        assistant_id="a",
        subject_name="Claire",
        source=MEDIA_FACT_SOURCE,
    )
    from_web = build_identity_document(
        verified, creator_id="c", assistant_id="a", subject_name="Claire"
    )
    assert from_media.metadata["source"] == MEDIA_FACT_SOURCE
    assert from_web.metadata["source"] == RESEARCH_FACT_SOURCE


""" Switches """


@pytest.mark.asyncio
async def test_verification_can_be_switched_off():
    class _Disabled(_Context):
        media_fact_verification_enabled = "false"

    summary = await verify_facts_from_media(
        store=None,
        context=_Disabled(),
        creator_id="c",
        assistant_id="a",
        subject_name="Claire",
        documents=[_document("anything")],
    )
    assert summary == {"skipped": "disabled"}


@pytest.mark.asyncio
async def test_nothing_to_read_is_not_an_error():
    summary = await verify_facts_from_media(
        store=None,
        context=_Context(),
        creator_id="c",
        assistant_id="a",
        subject_name="Claire",
        documents=[],
    )
    assert summary["applied"] == 0
    assert summary["proposals"] == 0


def test_the_limits_have_configured_defaults():
    from src.anubis.utils.context import GlobalContext

    context = GlobalContext()
    assert context.media_fact_verification_enabled == "true"
    assert context.media_fact_verification_max_documents == 40
