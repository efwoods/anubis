"""Choosing the avatar's portrait from images the owner already uploaded.

Pinned down:

- Switching portraits parks the generated media of the previous face and
  restores the media parked for the chosen face. Nothing is deleted.
- ``POST /avatar_reference_image/select`` analyses an image as a portrait at
  most once: the first switch to an image runs the description and the
  subject assessment, and switching back to a portrait analysed earlier runs
  no model call.
- The previous portrait stays selectable, including a portrait stored before
  uploads were kept as candidates.
- An upload whose image was never kept is refused with a 404 instead of
  silently producing an empty portrait.
- ``/list_avatar_documents`` marks only the active portrait as the reference
  image and says which uploads can become the portrait.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_IDLE_LOOP,
    ASSET_KIND_LIP_SYNC,
    ASSET_KIND_STILL,
    InMemoryMediaAssetRepository,
)
from src.anubis.utils.media_generation import portrait_candidates, reference_subject
from src.api import webapp as webapp_module
from src.subgraphs.process_media_graph.utils import utility as media_utility

ASSISTANT_ID = "assistant-alpha"
CREATOR_ID = "creator-1"
FIRST_IMAGE = "data:image/jpeg;base64,Rmlyc3Q="
SECOND_IMAGE = "data:image/jpeg;base64,U2Vjb25k"


class _Store:
    """In-process stand-in for the LangGraph store with prefix search."""

    def __init__(self):
        self.rows = {}
        self.index_arguments = {}

    async def aget(self, namespace, key):
        value = self.rows.get((tuple(namespace), key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value, index=None):
        self.rows[(tuple(namespace), key)] = value
        self.index_arguments[(tuple(namespace), key)] = index

    async def asearch(self, namespace_prefix, limit=None):
        prefix = tuple(namespace_prefix)
        return [
            SimpleNamespace(namespace=namespace, key=key, value=value)
            for (namespace, key), value in self.rows.items()
            if namespace[: len(prefix)] == prefix
        ]


def _current_user():
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": CREATOR_ID}]}


def _request(body):
    async def json():
        return body

    return SimpleNamespace(json=json)


def _asset(emotion, asset_kind, payload, variant_key=""):
    return {
        "user_id": CREATOR_ID,
        "assistant_id": ASSISTANT_ID,
        "emotion": emotion,
        "asset_kind": asset_kind,
        "variant_key": variant_key,
        "mime_type": "image/jpeg",
        "bytes": payload,
    }


def _indexed_image_document(filename, namespace_filename):
    """The identity row an ordinary image upload leaves behind."""
    return {
        "document": {
            "kwargs": {
                "page_content": f"description of {filename}",
                "metadata": {
                    "filename": filename,
                    "namespace_filename": namespace_filename,
                    "reference_image": False,
                },
            }
        }
    }


def _reference_row(image_data_uri, filename, namespace_filename):
    """The active portrait row as the upload pipeline writes the row."""
    return {
        "reference_image_data": image_data_uri,
        "document": {
            "kwargs": {
                "page_content": f"portrait description of {filename}",
                "metadata": {
                    "filename": filename,
                    "namespace_filename": namespace_filename,
                    "reference_image": True,
                },
            }
        },
        "reference_subject": "person",
        "reference_moderation_risk": "low",
    }


@pytest.fixture
def selection(monkeypatch):
    """Install the store, the creator's avatar and counting model stubs."""
    store = _Store()
    repository = InMemoryMediaAssetRepository()
    calls = {"descriptions": [], "assessments": []}

    async def fake_get(assistant_id):
        return {"assistant_id": assistant_id, "metadata": {"user_id": CREATOR_ID}}

    async def fake_describe(**kwargs):
        calls["descriptions"].append(kwargs["filename"])
        assert kwargs["reference_image"] is True
        return Document(
            page_content=f"portrait description of {kwargs['filename']}",
            metadata={"total_cost": 0.00123, "total_tokens": 10},
        )

    async def fake_assess(image_data_uri, context=None):
        calls["assessments"].append(image_data_uri)
        return {
            "subject": "person",
            "reasoning": "",
            "moderation_risk": "low",
            "moderation_reasons": [],
            "moderation_advice": "",
        }

    async def fake_meter(*args, **kwargs):
        return None

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=SimpleNamespace(get=fake_get)),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    monkeypatch.setattr(webapp_module, "_meter_image_description_usage", fake_meter)
    monkeypatch.setattr(media_utility, "extract_personality_from_image", fake_describe)
    monkeypatch.setattr(reference_subject, "classify_reference_subject", fake_assess)
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    monkeypatch.setattr(webapp_module.app.state, "context", None, raising=False)
    media_repository.set_media_asset_repository(repository)
    yield SimpleNamespace(store=store, repository=repository, calls=calls)
    media_repository.set_media_asset_repository(None)


async def _select(source_document_name):
    response = await webapp_module.select_avatar_reference_image(
        _request(
            {
                "assistant_id": ASSISTANT_ID,
                "source_document_name": source_document_name,
            }
        ),
        current_user=_current_user(),
    )
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_switching_portraits_parks_and_restores_media_without_deleting():
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_emotion_asset(_asset("joy", ASSET_KIND_STILL, b"first-joy"))
    await repository.upsert_emotion_asset(
        _asset("joy", ASSET_KIND_IDLE_LOOP, b"first-loop")
    )
    await repository.upsert_emotion_asset(
        _asset("joy", ASSET_KIND_LIP_SYNC, b"first-lip", variant_key="digest-1")
    )

    counts = await repository.switch_portrait_media(ASSISTANT_ID, "first", "second")
    assert counts == {"parked": 3, "restored": 0}
    assert await repository.list_emotion_assets(ASSISTANT_ID) == []
    assert len(repository.assets) == 3
    assert await repository.count_parked_portrait_media(ASSISTANT_ID) == {"first": 3}

    await repository.upsert_emotion_asset(
        _asset("joy", ASSET_KIND_STILL, b"second-joy")
    )
    counts = await repository.switch_portrait_media(ASSISTANT_ID, "second", "first")
    assert counts == {"parked": 1, "restored": 3}
    active = await repository.list_emotion_assets(ASSISTANT_ID, include_bytes=True)
    assert sorted(asset["bytes"] for asset in active) == [
        b"first-joy",
        b"first-lip",
        b"first-loop",
    ]
    # The lip-sync clip is looked up by its text digest, which survives the trip.
    assert {asset["variant_key"] for asset in active} == {"", "digest-1"}
    assert len(repository.assets) == 4


@pytest.mark.asyncio
async def test_re_storing_a_candidate_keeps_the_analysis_already_paid_for():
    store = _Store()
    await portrait_candidates.store_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-first",
        filename="first.jpg",
        image_data_uri=FIRST_IMAGE,
        reference_document_json={"kwargs": {"page_content": "analysed"}},
        assessment_fields={
            "reference_subject": "person",
            "reference_moderation_risk": "low",
        },
    )
    await portrait_candidates.store_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-first",
        filename="first.jpg",
        image_data_uri=FIRST_IMAGE,
    )
    candidate = await portrait_candidates.read_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-first",
    )
    assert portrait_candidates.candidate_reference_analysis(candidate) is not None
    # The candidate is never embedded by the store's vector index.
    assert set(store.index_arguments.values()) == {False}


@pytest.mark.asyncio
async def test_selecting_an_uploaded_image_analyses_once_and_switching_back_is_free(
    selection,
):
    store = selection.store
    # The first portrait predates portrait candidates: no candidate row.
    store.rows[((CREATOR_ID, ASSISTANT_ID, "reference_image"), ASSISTANT_ID)] = (
        _reference_row(FIRST_IMAGE, "first.jpg", "key-first")
    )
    store.rows[((CREATOR_ID, ASSISTANT_ID, "identity"), "row-second")] = (
        _indexed_image_document("second.jpg", "key-second")
    )
    await portrait_candidates.store_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-second",
        filename="second.jpg",
        image_data_uri=SECOND_IMAGE,
    )
    await selection.repository.upsert_emotion_asset(
        _asset("joy", ASSET_KIND_STILL, b"first-joy")
    )

    first_switch = await _select("second.jpg")

    assert first_switch["changed"] is True
    assert first_switch["analysis_reused"] is False
    assert first_switch["generated_media_parked"] == 1
    assert selection.calls["descriptions"] == ["second.jpg"]
    assert selection.calls["assessments"] == [SECOND_IMAGE]
    active_portrait = store.rows[
        ((CREATOR_ID, ASSISTANT_ID, "reference_image"), ASSISTANT_ID)
    ]
    assert active_portrait["reference_image_data"] == SECOND_IMAGE
    assert (
        active_portrait["document"]["kwargs"]["page_content"]
        == "portrait description of second.jpg"
    )
    assert await selection.repository.list_emotion_assets(ASSISTANT_ID) == []
    assert len(selection.repository.assets) == 1

    # The outgoing portrait was kept with the analysis already on the portrait
    # row, so switching back runs no model call and restores the media.
    store.rows[((CREATOR_ID, ASSISTANT_ID, "reference_image"), "row-first")] = (
        _reference_row(FIRST_IMAGE, "first.jpg", "key-first")
    )
    second_switch = await _select("first.jpg")

    assert second_switch["analysis_reused"] is True
    assert second_switch["generated_media_restored"] == 1
    assert selection.calls["descriptions"] == ["second.jpg"]
    assert selection.calls["assessments"] == [SECOND_IMAGE]
    restored = await selection.repository.list_emotion_assets(
        ASSISTANT_ID, include_bytes=True
    )
    assert [asset["bytes"] for asset in restored] == [b"first-joy"]

    # And the image analysed on the first switch is free from now on.
    third_switch = await _select("second.jpg")
    assert third_switch["analysis_reused"] is True
    assert selection.calls["descriptions"] == ["second.jpg"]


@pytest.mark.asyncio
async def test_selecting_the_active_portrait_changes_nothing(selection):
    store = selection.store
    store.rows[((CREATOR_ID, ASSISTANT_ID, "reference_image"), ASSISTANT_ID)] = (
        _reference_row(FIRST_IMAGE, "first.jpg", "key-first")
    )
    await portrait_candidates.store_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-first",
        filename="first.jpg",
        image_data_uri=FIRST_IMAGE,
    )
    await selection.repository.upsert_emotion_asset(
        _asset("joy", ASSET_KIND_STILL, b"first-joy")
    )

    result = await _select("first.jpg")

    assert result["changed"] is False
    assert selection.calls["descriptions"] == []
    assert len(await selection.repository.list_emotion_assets(ASSISTANT_ID)) == 1


@pytest.mark.asyncio
async def test_an_image_that_was_never_kept_is_refused(selection):
    selection.store.rows[((CREATOR_ID, ASSISTANT_ID, "identity"), "row-old")] = (
        _indexed_image_document("old.jpg", "key-old")
    )

    with pytest.raises(HTTPException) as refusal:
        await _select("old.jpg")

    assert refusal.value.status_code == 404
    assert "Upload the image again" in refusal.value.detail
    assert selection.calls["descriptions"] == []


@pytest.mark.asyncio
async def test_the_list_marks_only_the_active_portrait_and_selectable_uploads(
    selection,
):
    store = selection.store
    store.rows[((CREATOR_ID, ASSISTANT_ID, "reference_image"), ASSISTANT_ID)] = (
        _reference_row(SECOND_IMAGE, "second.jpg", "key-second")
    )
    # The indexed copy of an earlier portrait still carries the reference flag.
    store.rows[((CREATOR_ID, ASSISTANT_ID, "reference_image"), "row-first")] = (
        _reference_row(FIRST_IMAGE, "first.jpg", "key-first")
    )
    store.rows[((CREATOR_ID, ASSISTANT_ID, "identity"), "row-old")] = (
        _indexed_image_document("old.jpg", "key-old")
    )
    await portrait_candidates.store_portrait_candidate(
        store,
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        namespace_filename="key-first",
        filename="first.jpg",
        image_data_uri=FIRST_IMAGE,
    )
    await selection.repository.upsert_emotion_asset(
        _asset(
            "joy",
            ASSET_KIND_STILL,
            b"first-joy",
            variant_key=media_repository.parked_portrait_variant_prefix(
                portrait_candidates.portrait_key_for_image(FIRST_IMAGE)
            ),
        )
    )

    response = await webapp_module.list_avatar_documents(
        assistant_id=ASSISTANT_ID, current_user=_current_user()
    )

    by_label = {document["label"]: document for document in response["documents"]}
    assert by_label["second.jpg"]["is_reference_image"] is True
    assert by_label["first.jpg"]["is_reference_image"] is False
    assert by_label["first.jpg"]["portrait_selectable"] is True
    assert by_label["first.jpg"]["parked_generated_media"] == 1
    assert by_label["old.jpg"]["portrait_selectable"] is False
