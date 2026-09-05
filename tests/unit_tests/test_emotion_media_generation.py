"""Unit tests for the emotion media build (six stills + seven idle loops).

Pinned down:

- **The reference is the neutral still, and every other emotion is generated
  from it** — exactly six image edits and seven video generations per avatar.
- **A vendor failure loses one asset, not the set.** Everything that generated
  is persisted; the failure is reported per asset; a retry with
  ``only_missing`` regenerates only what is absent and pays only for that.
- **Spend is recorded per call** with the configured unit costs.
- **The manifest is a pure lookup**: one URL per (emotion, kind), and
  ``complete`` only when all fourteen entries exist.
- **The routes never expose bytes without an asset id and never generate for
  an avatar without a reference image.**
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_IDLE_LOOP,
    ASSET_KIND_STILL,
    InMemoryMediaAssetRepository,
)
from src.anubis.utils.media_generation import xai_client
from src.anubis.utils.media_generation.emotion_media import (
    build_manifest,
    generate_emotion_media_for_avatar,
)
from src.anubis.utils.media_generation.prompts import (
    BASE_EMOTIONS,
    GENERATED_EMOTIONS,
    idle_loop_prompt_for,
    still_prompt_for,
)

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"
REFERENCE = "data:image/jpeg;base64," + __import__("base64").b64encode(b"reference-image").decode()


def _context(**overrides):
    values = dict(
        xai_api_key="xai-test",
        emotion_media_generation_enabled="true",
        xai_image_edit_model="grok-imagine-image-2.0",
        xai_image_cost_per_image_usd=0.04,
        xai_video_model="grok-imagine-video-1.5",
        xai_video_cost_per_second_usd=0.08,
        xai_idle_loop_duration_seconds=6,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_vendor(monkeypatch, *, fail_stills=(), fail_loops=()):
    calls = {"edits": [], "videos": []}

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        emotion = next(e for e in GENERATED_EMOTIONS if still_prompt_for(e) == prompt)
        calls["edits"].append(emotion)
        if emotion in fail_stills:
            raise xai_client.XaiGenerationError(f"refused {emotion}")
        return {
            "bytes": f"still-{emotion}".encode(),
            "mime_type": "image/jpeg",
            "request_id": f"img-{emotion}",
            "model": "grok-imagine-image-2.0",
        }

    async def _generate_idle_loop(context, *, still_image_data_uri, prompt):
        emotion = next(e for e in BASE_EMOTIONS if idle_loop_prompt_for(e) == prompt)
        calls["videos"].append(emotion)
        if emotion in fail_loops:
            raise xai_client.XaiGenerationError(f"video refused {emotion}")
        return {
            "bytes": f"loop-{emotion}".encode(),
            "mime_type": "video/mp4",
            "request_id": f"vid-{emotion}",
            "model": "grok-imagine-video-1.5",
            "duration_seconds": 6.0,
        }

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)
    return calls


@pytest.mark.asyncio
async def test_a_full_build_makes_six_stills_and_seven_loops(monkeypatch):
    calls = _fake_vendor(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    metrics = []

    async def _metric(inference_type, cost_usd, model, request_id):
        metrics.append((inference_type, round(cost_usd, 4)))

    manifest = await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        metrics=_metric,
    )

    assert sorted(calls["edits"]) == sorted(GENERATED_EMOTIONS)
    assert sorted(calls["videos"]) == sorted(BASE_EMOTIONS)
    assets = await repository.list_emotion_assets(ASSISTANT_ID)
    assert len(assets) == 14
    assert manifest["complete"] is True
    assert manifest["failures"] == []
    neutral_still = next(
        a
        for a in assets
        if a["emotion"] == "neutral" and a["asset_kind"] == ASSET_KIND_STILL
    )
    assert neutral_still["vendor"] is None, "the reference itself is the neutral still"
    assert metrics.count(("image_generation", 0.04)) == 6
    assert metrics.count(("video_generation", 0.48)) == 7


@pytest.mark.asyncio
async def test_one_failed_generation_keeps_the_rest_and_is_reported(monkeypatch):
    _fake_vendor(monkeypatch, fail_stills=("anger",), fail_loops=("fear",))
    repository = InMemoryMediaAssetRepository()

    manifest = await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
    )

    kinds = {
        (a["emotion"], a["asset_kind"])
        for a in await repository.list_emotion_assets(ASSISTANT_ID)
    }
    assert ("anger", ASSET_KIND_STILL) not in kinds
    # No anger still means no anger loop either — it had nothing to animate.
    assert ("anger", ASSET_KIND_IDLE_LOOP) not in kinds
    assert ("fear", ASSET_KIND_STILL) in kinds
    assert ("fear", ASSET_KIND_IDLE_LOOP) not in kinds
    assert ("joy", ASSET_KIND_IDLE_LOOP) in kinds
    assert manifest["complete"] is False
    failed = {(f["emotion"], f["asset_kind"]) for f in manifest["failures"]}
    assert failed == {
        ("anger", ASSET_KIND_STILL),
        ("anger", ASSET_KIND_IDLE_LOOP),
        ("fear", ASSET_KIND_IDLE_LOOP),
    }
    assert sorted(manifest["missing"]) == sorted(
        ["anger:still", "anger:idle_loop", "fear:idle_loop"]
    )


@pytest.mark.asyncio
async def test_a_retry_regenerates_only_what_is_missing(monkeypatch):
    _fake_vendor(monkeypatch, fail_loops=("fear",))
    repository = InMemoryMediaAssetRepository()
    await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
    )

    calls = _fake_vendor(monkeypatch)
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        only_missing=True,
    )

    assert calls["edits"] == [], "existing stills are not paid for again"
    assert calls["videos"] == ["fear"]
    assert manifest["complete"] is True


def test_the_manifest_is_one_url_per_emotion_and_kind():
    assets = [
        {
            "asset_id": "a",
            "emotion": "joy",
            "asset_kind": ASSET_KIND_STILL,
            "mime_type": "image/jpeg",
        },
        {
            "asset_id": "b",
            "emotion": "joy",
            "asset_kind": ASSET_KIND_IDLE_LOOP,
            "mime_type": "video/mp4",
            "duration_seconds": 6.0,
        },
        {
            "asset_id": "c",
            "emotion": "joy",
            "asset_kind": "lip_sync",
            "mime_type": "video/mp4",
        },
    ]
    manifest = build_manifest(assets)
    assert manifest["emotions"]["joy"]["still"]["url"] == "/avatar_emotion_media/a"
    assert manifest["emotions"]["joy"]["idle_loop"]["duration_seconds"] == 6.0
    assert "lip_sync" not in manifest["emotions"]["joy"], (
        "clips are not part of the base set"
    )
    assert manifest["complete"] is False
    assert "neutral:still" in manifest["missing"]


def test_every_base_emotion_has_prompts():
    for emotion in GENERATED_EMOTIONS:
        assert "Change only the facial expression" in still_prompt_for(emotion)
    for emotion in BASE_EMOTIONS:
        assert "MUST match the supplied image" in idle_loop_prompt_for(emotion)
    with pytest.raises(ValueError):
        still_prompt_for("neutral")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_media_repository():
    media_repository.set_media_asset_repository(None)
    yield
    media_repository.set_media_asset_repository(None)


@pytest.mark.asyncio
async def test_the_manifest_route_reads_the_published_repository(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    await repository.upsert_emotion_asset(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "emotion": "neutral",
            "asset_kind": ASSET_KIND_STILL,
            "mime_type": "image/jpeg",
            "bytes": b"ref",
        }
    )
    media_repository.set_media_asset_repository(repository)

    async def _owner(assistant_id, current_user):
        return USER_ID

    monkeypatch.setattr(webapp_module, "_assistant_owner_for_media", _owner)
    response = await webapp_module.get_avatar_emotion_media(
        assistant_id=ASSISTANT_ID, current_user={"API_KEY": "k"}
    )
    body = response.body.decode("utf-8")
    assert "/avatar_emotion_media/" in body
    assert '"complete":false' in body.replace(" ", "")

    asset_id = list(repository.assets)[0]
    asset_response = await webapp_module.get_avatar_emotion_media_asset(
        asset_id=asset_id, current_user={"API_KEY": "k"}
    )
    assert asset_response.body == b"ref"
    assert asset_response.media_type == "image/jpeg"
    assert "immutable" in asset_response.headers["cache-control"]


@pytest.mark.asyncio
async def test_regeneration_requires_a_reference_image(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)

    class _Store:
        async def aget(self, namespace, key):
            return None

    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), store=_Store(), pool=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    async def _resolve(assistant_id, current_user, action_description=""):
        return ({"assistant_id": assistant_id}, USER_ID)

    monkeypatch.setattr(webapp_module, "resolve_assistant_for_creator", _resolve)

    async def _json():
        return {"assistant_id": ASSISTANT_ID}

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.regenerate_avatar_emotion_media(
            request=SimpleNamespace(json=_json),
            current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
        )
    assert raised.value.status_code == 404
    assert repository.jobs == {}


@pytest.mark.asyncio
async def test_regeneration_creates_a_durable_job(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _fake_vendor(monkeypatch)

    class _Store:
        async def aget(self, namespace, key):
            return SimpleNamespace(value={"reference_image_data": REFERENCE})

    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), store=_Store(), pool=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    async def _resolve(assistant_id, current_user, action_description=""):
        return ({"assistant_id": assistant_id}, USER_ID)

    monkeypatch.setattr(webapp_module, "resolve_assistant_for_creator", _resolve)

    async def _no_metrics(*args, **kwargs):
        return None

    from src.anubis.utils.billing import metering

    monkeypatch.setattr(metering, "persist_api_metrics_row", _no_metrics)

    async def _json():
        return {"assistant_id": ASSISTANT_ID, "only_missing": False}

    response = await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 202
    for _ in range(50):
        await asyncio.sleep(0)
        jobs = list(repository.jobs.values())
        if jobs and jobs[0]["state"] in ("completed", "failed"):
            break
    job = list(repository.jobs.values())[0]
    assert job["state"] == "completed", job
    assert len(await repository.list_emotion_assets(ASSISTANT_ID)) == 14


@pytest.mark.asyncio
async def test_a_targeted_loop_regenerates_only_that_loop(monkeypatch):
    _fake_vendor(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
    )
    calls = _fake_vendor(monkeypatch)
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        only_missing=False,
        emotions=("joy",),
        asset_kinds=(ASSET_KIND_IDLE_LOOP,),
    )
    assert calls["edits"] == []
    assert calls["videos"] == ["joy"]
    assert manifest["complete"] is True


@pytest.mark.asyncio
async def test_extra_prompt_is_appended_to_the_generation(monkeypatch):
    captured = {"videos": []}

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        raise AssertionError("stills should not run for a targeted loop")

    async def _generate_idle_loop(context, *, still_image_data_uri, prompt):
        captured["videos"].append(prompt)
        return {
            "bytes": b"loop-joy",
            "mime_type": "video/mp4",
            "request_id": "vid-joy",
            "model": "grok-imagine-video-1.5",
            "duration_seconds": 6.0,
        }

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_emotion_asset(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "emotion": "joy",
            "asset_kind": ASSET_KIND_STILL,
            "mime_type": "image/jpeg",
            "bytes": b"still-joy",
        }
    )
    await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        only_missing=False,
        emotions=("joy",),
        asset_kinds=(ASSET_KIND_IDLE_LOOP,),
        extra_prompt="blink more slowly",
    )
    assert captured["videos"], "the loop should have been generated"
    assert idle_loop_prompt_for("joy") in captured["videos"][0]
    assert "blink more slowly" in captured["videos"][0]


@pytest.mark.asyncio
async def test_delete_emotion_asset_removes_one_row():
    repository = InMemoryMediaAssetRepository()
    asset_id = await repository.upsert_emotion_asset(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "emotion": "joy",
            "asset_kind": ASSET_KIND_STILL,
            "mime_type": "image/jpeg",
            "bytes": b"still-joy",
        }
    )
    assert await repository.delete_emotion_asset(asset_id) is True
    assert await repository.get_emotion_asset(asset_id) is None
    assert await repository.delete_emotion_asset(asset_id) is False
