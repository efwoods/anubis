"""Unit tests for acquiring the portrait and the voice a new avatar lacks.

Creating an avatar now starts research, and research goes looking for a picture
and a recording of the subject when the avatar has neither. What is pinned down
here is the judgement in that search, because installing the wrong face or the
wrong voice on an avatar is worse than installing none:

- **A picture is only accepted when the vision model vouches for every bar** —
  the named subject, one person, an unobstructed visible face, a real
  photograph, no moderation risk, and enough confidence. Failing any one of
  those moves on to the next candidate.
- **A page's own ``og:image`` outranks a search result**, because it is the
  page's answer to which picture the page is about.
- **A playlist or a channel is never a voice candidate**, since "the video where
  this person speaks most" has no meaning for a link standing for many videos.
- **A recording outside the duration bounds is dropped**, which is the cost
  bound that makes acquisition safe to run for every tier.
- **A portrait the creator uploaded mid-run cancels the hand-off**, so their
  choice wins over the researched one.
- **The acquisition runs once per avatar**, so re-running research acquires
  nothing a second time.
"""

from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from src.anubis.utils.media_generation.reference_image import store_reference_image
from src.anubis.utils.research import asset_bootstrap
from src.anubis.utils.research.asset_bootstrap import (
    BootstrapGateway,
    PortraitVerdict,
    SpeakingVideoScore,
    acquire_portrait,
    assess_missing_assets,
    claim_bootstrap,
    portrait_candidates_from_sources,
    portrait_verdict_accepts,
    probe_video_candidates,
    rank_speaking_videos,
    youtube_candidates_from_sources,
)
from src.anubis.utils.research.web_search import ImageCandidate, SearchResult

CREATOR_ID = "auth0|creator"
ASSISTANT_ID = "assistant-1"
SUBJECT_NAME = "Ada Lovelace"


def _context(**overrides):
    values = dict(
        deep_research_bootstrap_enabled="true",
        deep_research_bootstrap_max_image_candidates=4,
        deep_research_bootstrap_image_max_bytes=1024,
        deep_research_bootstrap_min_portrait_confidence=0.7,
        deep_research_bootstrap_max_video_candidates=4,
        deep_research_bootstrap_min_video_seconds=120.0,
        deep_research_bootstrap_max_video_seconds=2400.0,
        deep_research_bootstrap_media_wait_seconds=1.0,
        reference_audio_clip_max_seconds=10.0,
        reference_audio_minimum_seconds=1.3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _verdict(**overrides) -> PortraitVerdict:
    values = dict(
        depicts_named_subject=True,
        single_person=True,
        face_clearly_visible=True,
        obstructed=False,
        is_photograph=True,
        moderation_risk="low",
        confidence=0.9,
        reasoning="A clear head-and-shoulders photograph.",
    )
    values.update(overrides)
    return PortraitVerdict(**values)


def _gateway(*, verdicts=None, started=None, fetch_fails=()):
    """A gateway whose four operations are recorded rather than performed."""
    calls = {"fetched": [], "image_jobs": [], "audio_jobs": []}

    async def _fetch(url):
        calls["fetched"].append(url)
        if url in fetch_fails:
            raise RuntimeError("404")
        return "image/jpeg", b"bytes", f"data:image/jpeg;base64,{len(url)}"

    async def _image_job(*, filename, mime_type, content, source_url=None):
        calls["image_jobs"].append(filename)
        return started or {"status": "started", "job_id": "job-image"}

    async def _audio_job(*, url):
        calls["audio_jobs"].append(url)
        return started or {"status": "started", "job_id": "job-audio"}

    async def _await_job(job_id, timeout_seconds):
        return "finished"

    gateway = BootstrapGateway(
        fetch_image_bytes=_fetch,
        start_reference_image_job=_image_job,
        start_reference_audio_job=_audio_job,
        await_media_job=_await_job,
    )
    gateway.calls = calls
    return gateway


# ── which pictures are even considered ──────────────────────────────────────


def test_a_pages_own_image_outranks_a_search_result():
    sources = [
        SearchResult(
            url="https://en.wikipedia.org/wiki/Ada_Lovelace",
            queries=["ada lovelace"],
            images=[
                ImageCandidate(
                    url="https://upload.example/ada.jpg",
                    origin="wikipedia",
                    source_url="https://en.wikipedia.org/wiki/Ada_Lovelace",
                ),
            ],
        ),
        SearchResult(
            url="https://blog.example/post",
            queries=["ada lovelace"],
            images=[
                ImageCandidate(
                    url="https://blog.example/banner.png",
                    origin="json-ld",
                    source_url="https://blog.example/post",
                ),
            ],
        ),
    ]
    ranked = portrait_candidates_from_sources(sources, limit=5)
    assert [candidate.origin for candidate in ranked] == ["wikipedia", "json-ld"]


def test_the_same_picture_on_two_pages_is_considered_once():
    shared = "https://cdn.example/ada.jpg"
    sources = [
        SearchResult(
            url="https://a.example/",
            queries=["q"],
            images=[ImageCandidate(url=shared, origin="og:image")],
        ),
        SearchResult(
            url="https://b.example/",
            queries=["q"],
            images=[ImageCandidate(url=shared, origin="og:image")],
        ),
    ]
    assert len(portrait_candidates_from_sources(sources, limit=5)) == 1


# ── what a picture must clear to become the face ────────────────────────────


@pytest.mark.parametrize(
    "failing_field",
    [
        {"depicts_named_subject": False},
        {"single_person": False},
        {"face_clearly_visible": False},
        {"obstructed": True},
        {"is_photograph": False},
        {"moderation_risk": "high"},
        {"confidence": 0.4},
    ],
)
def test_failing_any_single_bar_rejects_the_picture(failing_field):
    assert (
        portrait_verdict_accepts(_verdict(**failing_field), minimum_confidence=0.7)
        is False
    )


def test_a_picture_clearing_every_bar_is_accepted():
    assert portrait_verdict_accepts(_verdict(), minimum_confidence=0.7) is True


@pytest.mark.asyncio
async def test_a_rejected_candidate_is_followed_by_the_next_one(monkeypatch):
    store = InMemoryStore()
    verdicts = iter([_verdict(single_person=False), _verdict()])
    monkeypatch.setattr(
        asset_bootstrap,
        "vet_portrait_candidate",
        lambda *args, **kwargs: _next_verdict(verdicts),
    )
    monkeypatch.setattr(
        asset_bootstrap, "search_portrait_candidates", _no_extra_candidates
    )
    gateway = _gateway()
    sources = [
        SearchResult(
            url="https://a.example/",
            queries=["q"],
            images=[
                ImageCandidate(url="https://cdn.example/group.jpg", origin="og:image"),
                ImageCandidate(url="https://cdn.example/solo.jpg", origin="og:image"),
            ],
        )
    ]
    result = await acquire_portrait(
        store,
        _context(),
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        subject_name=SUBJECT_NAME,
        subject_summary="A mathematician.",
        sources=sources,
        gateway=gateway,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )
    assert result["acquired"] is True
    assert result["source_url"] == "https://cdn.example/solo.jpg"
    assert gateway.calls["image_jobs"] == ["https://cdn.example/solo.jpg"]


@pytest.mark.asyncio
async def test_no_confirmable_picture_acquires_nothing_and_says_why(monkeypatch):
    store = InMemoryStore()
    monkeypatch.setattr(
        asset_bootstrap,
        "vet_portrait_candidate",
        lambda *args, **kwargs: _resolved(_verdict(depicts_named_subject=False)),
    )
    monkeypatch.setattr(
        asset_bootstrap, "search_portrait_candidates", _no_extra_candidates
    )
    gateway = _gateway()
    sources = [
        SearchResult(
            url="https://a.example/",
            queries=["q"],
            images=[ImageCandidate(url="https://cdn.example/x.jpg", origin="og:image")],
        )
    ]
    result = await acquire_portrait(
        store,
        _context(),
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        subject_name=SUBJECT_NAME,
        subject_summary="",
        sources=sources,
        gateway=gateway,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )
    assert result["acquired"] is False
    assert SUBJECT_NAME in result["reason"]
    assert gateway.calls["image_jobs"] == []


@pytest.mark.asyncio
async def test_a_portrait_the_creator_uploaded_mid_run_cancels_the_hand_off(
    monkeypatch,
):
    store = InMemoryStore()

    async def _vet_then_upload(*args, **kwargs):
        # The creator's own upload lands while the candidate is being vetted.
        await store_reference_image(
            store,
            user_id=CREATOR_ID,
            assistant_id=ASSISTANT_ID,
            image_data_uri="data:image/jpeg;base64,T1dO",
            document_json={"kwargs": {"page_content": "the creator's picture"}},
            replace=True,
        )
        return _verdict()

    monkeypatch.setattr(asset_bootstrap, "vet_portrait_candidate", _vet_then_upload)
    monkeypatch.setattr(
        asset_bootstrap, "search_portrait_candidates", _no_extra_candidates
    )
    gateway = _gateway()
    sources = [
        SearchResult(
            url="https://a.example/",
            queries=["q"],
            images=[ImageCandidate(url="https://cdn.example/x.jpg", origin="og:image")],
        )
    ]
    result = await acquire_portrait(
        store,
        _context(),
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        subject_name=SUBJECT_NAME,
        subject_summary="",
        sources=sources,
        gateway=gateway,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )
    assert result["acquired"] is False
    assert "creator" in result["reason"].lower()
    assert gateway.calls["image_jobs"] == []


# ── which recordings may teach the voice ────────────────────────────────────


def test_a_playlist_or_a_channel_is_never_a_voice_candidate():
    sources = [
        SearchResult(url="https://www.youtube.com/watch?v=abcdefghijk"),
        SearchResult(url="https://www.youtube.com/playlist?list=PL123"),
        SearchResult(url="https://www.youtube.com/@somechannel"),
        SearchResult(url="https://example.com/an-article"),
    ]
    assert youtube_candidates_from_sources(sources) == [
        "https://www.youtube.com/watch?v=abcdefghijk"
    ]


@pytest.mark.asyncio
async def test_recordings_outside_the_duration_bounds_are_dropped(monkeypatch):
    metadata = {
        "https://youtu.be/short": {"duration": 30.0, "title": "clip", "age_limit": 0},
        "https://youtu.be/right": {
            "duration": 900.0,
            "title": "interview",
            "age_limit": 0,
        },
        "https://youtu.be/long": {
            "duration": 9000.0,
            "title": "stream",
            "age_limit": 0,
        },
        "https://youtu.be/live": {
            "duration": 900.0,
            "title": "live",
            "age_limit": 0,
            "live_status": "is_live",
        },
    }

    async def _probe(url):
        return metadata[url]

    monkeypatch.setattr("src.anubis.utils.utility.get_remote_video_metadata", _probe)
    surviving = await probe_video_candidates(list(metadata), context=_context())
    assert [entry["url"] for entry in surviving] == ["https://youtu.be/right"]


@pytest.mark.asyncio
async def test_only_recordings_of_the_subject_speaking_are_ranked(monkeypatch):
    scores = {
        "someone else's review": SpeakingVideoScore(
            subject_speaks_the_most=False,
            format="other",
            is_about_the_subject=False,
            score=0.9,
            reasoning="A reviewer talking about her.",
        ),
        "her own lecture": SpeakingVideoScore(
            subject_speaks_the_most=True,
            format="talk",
            is_about_the_subject=True,
            score=0.8,
            reasoning="She gives the whole talk.",
        ),
        "a short interview": SpeakingVideoScore(
            subject_speaks_the_most=True,
            format="interview",
            is_about_the_subject=True,
            score=0.6,
            reasoning="She answers throughout.",
        ),
    }

    async def _invoke(response_format, system_prompt, human_text):
        for title, score in scores.items():
            if title in human_text:
                return score
        raise AssertionError(f"unexpected candidate: {human_text}")

    monkeypatch.setattr(
        "src.anubis.utils.research.deep_research.invoke_structured", _invoke
    )
    candidates = [
        {"url": f"https://youtu.be/{index}", "title": title, "duration": 600.0}
        for index, title in enumerate(scores)
    ]
    ranked = await rank_speaking_videos(
        candidates, subject_name=SUBJECT_NAME, subject_summary=""
    )
    # The reviewer is dropped rather than ranked low, and the best of the rest
    # comes first.
    assert [entry["title"] for entry in ranked] == [
        "her own lecture",
        "a short interview",
    ]


# ── running once per avatar ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_acquisition_is_claimed_once_per_avatar():
    store = InMemoryStore()
    assert (
        await claim_bootstrap(store, creator_id=CREATOR_ID, assistant_id=ASSISTANT_ID)
        is True
    )
    assert (
        await claim_bootstrap(store, creator_id=CREATOR_ID, assistant_id=ASSISTANT_ID)
        is False
    )


@pytest.mark.asyncio
async def test_an_avatar_with_a_portrait_only_still_needs_a_voice():
    store = InMemoryStore()
    await store_reference_image(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        image_data_uri="data:image/jpeg;base64,QUFB",
        document_json={"kwargs": {"page_content": "a face"}},
        replace=True,
    )
    needs = await assess_missing_assets(
        store, creator_id=CREATOR_ID, assistant_id=ASSISTANT_ID, context=_context()
    )
    assert needs.needs_portrait is False
    assert needs.needs_voice is True
    assert needs.needs_anything is True


# ── helpers ─────────────────────────────────────────────────────────────────


async def _resolved(value):
    return value


def _next_verdict(verdicts):
    return _resolved(next(verdicts))


async def _no_extra_candidates(*args, **kwargs):
    return []


""" The acquisition's mark survives the whole way to the store write """


@pytest.mark.asyncio
async def test_the_acquisition_mark_reaches_the_media_graph(monkeypatch):
    """A researched portrait is marked at the API edge and still marked at the end.

    The mark is what makes the store write conditional, and it crosses three
    hand-offs to get there: the entry the API builds, the media list the graph
    converts it into, and the metadata the write site reads. A break anywhere
    along that chain would silently restore the old behaviour — a researched
    picture overwriting the creator's own — so the whole chain is exercised.
    """
    from src.api import webapp as webapp_module

    monkeypatch.setattr(
        webapp_module,
        "prepare_still_image_upload",
        lambda declared_mime, body: ("image/jpeg", body),
    )
    entries = await webapp_module._build_media_entries_for_file(
        "https://cdn.example/ada.jpg",
        b"\xff\xd8\xff\xe0 pretend jpeg",
        "image/jpeg",
        reference_image=True,
        reference_audio=False,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        bootstrap=True,
        reference_source_url="https://cdn.example/ada.jpg",
    )
    assert entries[0]["bootstrap_reference"] is True

    converted = await _convert_to_media_list(entries)
    portrait = converted["media_list"][0]
    assert portrait["metadata"]["reference_image"] is True
    assert portrait["metadata"]["bootstrap_reference"] is True
    assert portrait["metadata"]["reference_source_url"] == "https://cdn.example/ada.jpg"


@pytest.mark.asyncio
async def test_an_ordinary_upload_is_not_marked(monkeypatch):
    """A picture a person chose carries no mark, so its write still replaces."""
    from src.api import webapp as webapp_module

    monkeypatch.setattr(
        webapp_module,
        "prepare_still_image_upload",
        lambda declared_mime, body: ("image/jpeg", body),
    )
    entries = await webapp_module._build_media_entries_for_file(
        "my-photo.jpg",
        b"\xff\xd8\xff\xe0 pretend jpeg",
        "image/jpeg",
        reference_image=True,
        reference_audio=False,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
    )
    assert entries[0]["bootstrap_reference"] is False

    converted = await _convert_to_media_list(entries)
    assert converted["media_list"][0]["metadata"]["bootstrap_reference"] is False


async def _convert_to_media_list(entries: list[dict]) -> dict:
    """Run the media graph's file-conversion node over one batch of entries."""
    from src.subgraphs.process_media_graph.utils.nodes import (
        process_uploaded_files_and_label_media_type,
    )

    return await process_uploaded_files_and_label_media_type(
        {"media_files": entries},
        SimpleNamespace(context=_context()),
        {"configurable": {"user_id": CREATOR_ID, "assistant_id": ASSISTANT_ID}},
        InMemoryStore(),
    )
