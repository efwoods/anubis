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
from src.anubis.utils.media_generation import reference_subject, xai_client
from src.anubis.utils.media_generation.emotion_media import (
    build_manifest,
    describe_failure,
    emotion_media_cost_estimate,
    full_build_asset_counts,
    missing_asset_counts,
    generate_emotion_media_for_avatar,
    summarize_failures,
)
from src.anubis.utils.media_generation.prompts import (
    BASE_EMOTIONS,
    GENERATED_EMOTIONS,
    SUBJECT_NON_HUMAN,
    SUBJECT_PERSON,
    SUBJECT_STYLIZED_CHARACTER,
    idle_loop_prompt_for,
    still_prompt_for,
)

USER_ID = "auth0-user"


def _signed_in_owner(tier: str = "premium") -> dict:
    """The signed-in creator, holding a tier. Generation needs the premium tier.

    ``EMOTION_MEDIA_MINIMUM_TIER`` decides which tier may spend at the image and
    video vendor; the routes read the tier off the user record, so the tests
    carry a real one rather than stubbing the gate away.
    """
    return {
        "API_KEY": "k",
        "identities": [{"user_id": USER_ID}],
        "app_metadata": {"subscription_status": {"tier": tier}},
    }

ASSISTANT_ID = "assistant-1"
REFERENCE = (
    "data:image/jpeg;base64,"
    + __import__("base64").b64encode(b"reference-image").decode()
)


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


def _fake_subject_classifier(monkeypatch, subject=SUBJECT_PERSON, reasons=()):
    calls = []

    async def _classify(reference_image_data_uri, context=None):
        calls.append(reference_image_data_uri)
        return reference_subject.normalize_assessment(
            {
                "subject": subject,
                "reasoning": f"looks like a {subject}",
                "moderation_reasons": list(reasons),
                "moderation_risk": "high" if reasons else "low",
                "moderation_advice": "Use a calmer portrait." if reasons else "",
            }
        )

    monkeypatch.setattr(reference_subject, "classify_reference_subject", _classify)
    return calls


# The body xAI returns when it refuses a rendered video after the fact.
MODERATED_BODY = (
    'xAI failed the status check for vid-1 (400): {"code":"imagine:content-moderated",'
    '"error":"Generated video rejected by content moderation.",'
    '"usage":{"cost_in_usd_ticks":8500000000}}'
)


def _fake_vendor(monkeypatch, *, fail_stills=(), fail_loops=()):
    calls = {"edits": [], "videos": []}

    def _emotion_for(prompt, emotions, prompt_for):
        return next(
            e
            for e in emotions
            for subject in (
                SUBJECT_PERSON,
                SUBJECT_STYLIZED_CHARACTER,
                SUBJECT_NON_HUMAN,
            )
            if prompt_for(e, subject) == prompt
        )

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        emotion = _emotion_for(prompt, GENERATED_EMOTIONS, still_prompt_for)
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
        emotion = _emotion_for(prompt, BASE_EMOTIONS, idle_loop_prompt_for)
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
            current_user=_signed_in_owner(),
        )
    assert raised.value.status_code == 404
    assert repository.jobs == {}


@pytest.mark.asyncio
async def test_regeneration_is_refused_below_the_minimum_tier(monkeypatch):
    """A pro owner is told which tier generates emotion media, and pays nothing."""
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _fake_vendor(monkeypatch)
    _fake_subject_classifier(monkeypatch)

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
        raise AssertionError("the tier is checked before the avatar is loaded")

    monkeypatch.setattr(webapp_module, "resolve_assistant_for_creator", _resolve)

    async def _json():
        return {"assistant_id": ASSISTANT_ID, "only_missing": False}

    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.regenerate_avatar_emotion_media(
            request=SimpleNamespace(json=_json),
            current_user=_signed_in_owner("pro"),
        )

    assert raised.value.status_code == 403
    assert "premium" in raised.value.detail
    assert repository.jobs == {}


def test_the_cost_of_a_run_is_the_vendor_arithmetic():
    """The owner confirms a spend, so the estimate is the unit costs, not a guess."""
    still_count, idle_loop_count = full_build_asset_counts()
    # Six stills — the reference IS the neutral still and costs nothing — and
    # one idle loop per base emotion.
    assert (still_count, idle_loop_count) == (6, 7)

    estimate = emotion_media_cost_estimate(
        _context(),
        still_count=still_count,
        idle_loop_count=idle_loop_count,
    )

    assert estimate["stills_usd"] == 0.24  # 6 × $0.04
    assert estimate["idle_loops_usd"] == 3.36  # 7 × 6 s × $0.08
    assert estimate["total_usd"] == 3.60
    assert estimate["idle_loop_seconds"] == 6


def test_a_missing_only_run_is_costed_from_what_is_absent():
    """Only the absent assets are priced, and never the neutral still."""
    assert missing_asset_counts(
        ["neutral:still", "anger:still", "anger:idle_loop", "joy:idle_loop"]
    ) == (1, 2)
    assert missing_asset_counts([]) == (0, 0)
    assert missing_asset_counts(None) == (0, 0)

    estimate = emotion_media_cost_estimate(
        _context(), still_count=1, idle_loop_count=2
    )
    assert estimate["total_usd"] == 1.0  # $0.04 + 2 × 6 s × $0.08


def test_the_manifest_carries_both_estimates_for_the_confirmation(monkeypatch):
    """The settings screen shows the owner what a rebuild and a top-up each cost."""
    from src.api import webapp as webapp_module

    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None),
    )

    permission = webapp_module._emotion_media_generation_permission(
        _signed_in_owner("premium"),
        USER_ID,
        {"missing": ["anger:idle_loop"]},
    )

    assert permission["cost_full_rebuild"]["total_usd"] == 3.60
    assert permission["cost_full_rebuild"]["idle_loops"] == 7
    assert permission["cost_missing_only"]["total_usd"] == 0.48
    assert permission["cost_missing_only"]["stills"] == 0


def test_the_manifest_tells_the_owner_whether_generation_is_permitted(monkeypatch):
    """The settings screen reads the tier answer off the manifest, and nobody else does."""
    from src.api import webapp as webapp_module

    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None),
    )

    premium = webapp_module._emotion_media_generation_permission(
        _signed_in_owner("premium"), USER_ID, {"missing": []}
    )
    assert premium["allowed"] is True
    assert premium["required_tier"] == "premium"

    pro = webapp_module._emotion_media_generation_permission(
        _signed_in_owner("pro"), USER_ID, {"missing": []}
    )
    assert pro["allowed"] is False
    assert pro["tier"] == "pro"

    # A chatter who is not the creator is never offered the control.
    assert (
        webapp_module._emotion_media_generation_permission(
            _signed_in_owner("premium"), "somebody-else", {"missing": []}
        )
        is None
    )


@pytest.mark.asyncio
async def test_regeneration_creates_a_durable_job(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _fake_vendor(monkeypatch)
    _fake_subject_classifier(monkeypatch)

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
        current_user=_signed_in_owner(),
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
async def test_an_owner_can_cancel_a_running_emotion_media_job(monkeypatch):
    """Confirm starts the spend; cancel stops further image and video calls."""
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    started = asyncio.Event()

    async def _hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("cancel should have stopped this vendor call")

    monkeypatch.setattr(xai_client, "edit_image", _hang)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _hang)
    _fake_subject_classifier(monkeypatch)

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

    owner = _signed_in_owner()
    response = await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json),
        current_user=owner,
    )
    assert response.status_code == 202
    job_id = __import__("json").loads(response.body)["job_id"]
    await asyncio.wait_for(started.wait(), timeout=1)
    cancel_response = await webapp_module.cancel_avatar_media_job(
        job_id=job_id,
        current_user=owner,
    )
    assert cancel_response.status_code == 200
    assert __import__("json").loads(cancel_response.body)["state"] == "cancelled"
    for _ in range(50):
        await asyncio.sleep(0)
        if repository.jobs[job_id]["state"] == "cancelled":
            break
    assert repository.jobs[job_id]["state"] == "cancelled"


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


# --- reference subjects and failure reasons ---------------------------------


def test_prompt_families_follow_the_reference_subject():
    """A subject with no face is never asked for one; a character keeps its style."""
    for emotion in GENERATED_EMOTIONS:
        person = still_prompt_for(emotion, SUBJECT_PERSON)
        character = still_prompt_for(emotion, SUBJECT_STYLIZED_CHARACTER)
        non_human = still_prompt_for(emotion, SUBJECT_NON_HUMAN)
        assert person == still_prompt_for(emotion), "person is the default family"
        assert "facial expression" in person
        assert "art style" in character and "facial expression" in character
        assert "facial expression" not in non_human
        assert "brows" not in non_human and "smile" not in non_human
        assert "do not add a person" in non_human
        assert emotion in non_human
    for emotion in BASE_EMOTIONS:
        assert "breathes" in idle_loop_prompt_for(emotion, SUBJECT_PERSON)
        loop = idle_loop_prompt_for(emotion, SUBJECT_NON_HUMAN)
        assert "breathes" not in loop and "blinks" not in loop
        assert "do not add a person" in loop
    # Unknown or missing subjects fall back to the person family.
    assert still_prompt_for("joy", None) == still_prompt_for("joy", SUBJECT_PERSON)
    assert still_prompt_for("joy", "mystery") == still_prompt_for("joy", SUBJECT_PERSON)


@pytest.mark.asyncio
async def test_the_subject_selects_the_prompts_that_are_sent(monkeypatch):
    prompts = {"edits": [], "videos": []}

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        prompts["edits"].append(prompt)
        return {
            "bytes": b"still",
            "mime_type": "image/jpeg",
            "request_id": "img",
            "model": "grok-imagine-image-2.0",
        }

    async def _generate_idle_loop(context, *, still_image_data_uri, prompt):
        prompts["videos"].append(prompt)
        return {
            "bytes": b"loop",
            "mime_type": "video/mp4",
            "request_id": "vid",
            "model": "grok-imagine-video-1.5",
            "duration_seconds": 6.0,
        }

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        InMemoryMediaAssetRepository(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        subject=SUBJECT_NON_HUMAN,
    )
    assert manifest["subject"] == SUBJECT_NON_HUMAN
    assert sorted(prompts["edits"]) == sorted(
        still_prompt_for(e, SUBJECT_NON_HUMAN) for e in GENERATED_EMOTIONS
    )
    assert sorted(prompts["videos"]) == sorted(
        idle_loop_prompt_for(e, SUBJECT_NON_HUMAN) for e in BASE_EMOTIONS
    )
    for prompt in prompts["edits"] + prompts["videos"]:
        assert "facial expression" not in prompt and "breathes" not in prompt


def test_a_moderation_refusal_is_named_and_not_retried():
    failure = describe_failure("joy", ASSET_KIND_IDLE_LOOP, MODERATED_BODY)
    assert failure["error_code"] == xai_client.ERROR_CODE_CONTENT_MODERATED
    assert "content moderation" in failure["message"]
    assert "repeats" in failure["message"]
    assert "Try again" not in failure["message"]

    transient = describe_failure(
        "joy", ASSET_KIND_STILL, "The image edit could not reach xAI: timeout"
    )
    assert transient["error_code"] == xai_client.ERROR_CODE_VENDOR
    assert "timeout" in transient["message"]

    summary = summarize_failures([failure] * 7)
    assert summary == {
        "failed_stills": 0,
        "failed_loops": 7,
        "moderated": 7,
        "predicted": 0,
        "message": summary["message"],
    }
    assert summary["message"].startswith(
        "xAI's content moderation refused 7 emotion videos"
    )
    mixed = summarize_failures([failure, transient])
    assert mixed["failed_stills"] == 1 and mixed["moderated"] == 1
    assert "refused 1 of them" in mixed["message"]
    assert summarize_failures([])["message"] == ""


@pytest.mark.asyncio
async def test_every_refused_loop_reaches_the_completion_frame(monkeypatch):
    """Seven moderated videos must not tick the loops step as done."""

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        return {
            "bytes": b"still",
            "mime_type": "image/jpeg",
            "request_id": "img",
            "model": "grok-imagine-image-2.0",
        }

    async def _generate_idle_loop(context, *, still_image_data_uri, prompt):
        raise xai_client.XaiGenerationError(MODERATED_BODY)

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)
    frames = []
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        InMemoryMediaAssetRepository(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        progress=lambda stage, fields: frames.append((stage, fields)),
    )
    assert len(manifest["failures"]) == 7
    assert {f["error_code"] for f in manifest["failures"]} == {
        xai_client.ERROR_CODE_CONTENT_MODERATED
    }
    stage, fields = frames[-1]
    assert stage == "emotion_media_complete"
    assert fields["complete"] is False
    assert fields["failures"] == 7
    assert fields["failed_loops"] == 7 and fields["failed_stills"] == 0
    assert fields["moderated"] == 7
    assert "content moderation" in fields["failure_message"]
    assert len(fields["failed_assets"]) == 7
    assert fields["failed_assets"][0]["error_code"] == "content_moderated"
    assert fields["subject"] == SUBJECT_PERSON


@pytest.mark.asyncio
async def test_the_manifest_reports_the_last_generation(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    failures = [
        describe_failure(e, ASSET_KIND_IDLE_LOOP, MODERATED_BODY) for e in BASE_EMOTIONS
    ]
    job_id = await repository.create_job(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        job_kind="emotion_media",
        detail={"source": "upload", "subject": SUBJECT_STYLIZED_CHARACTER},
    )
    await repository.update_job(
        job_id,
        state="failed",
        detail={"failures": failures, "summary": summarize_failures(failures)},
    )

    async def _owner(assistant_id, current_user):
        return USER_ID

    monkeypatch.setattr(webapp_module, "_assistant_owner_for_media", _owner)
    response = await webapp_module.get_avatar_emotion_media(
        assistant_id=ASSISTANT_ID, current_user={"API_KEY": "k"}
    )
    body = __import__("json").loads(response.body)
    last = body["last_generation"]
    assert last["state"] == "failed"
    assert last["subject"] == SUBJECT_STYLIZED_CHARACTER
    assert last["summary"]["moderated"] == 7
    assert len(last["failures"]) == 7
    assert last["failures"][0]["error_code"] == "content_moderated"
    assert "message" in last["failures"][0]
    # The raw vendor text stays server-side; only the reason travels.
    assert "cost_in_usd_ticks" not in response.body.decode()


@pytest.mark.asyncio
async def test_regeneration_classifies_an_unclassified_reference_once(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _fake_vendor(monkeypatch)
    classifier_calls = _fake_subject_classifier(monkeypatch, SUBJECT_NON_HUMAN)
    writes = []

    class _Store:
        async def aget(self, namespace, key):
            return SimpleNamespace(value={"reference_image_data": REFERENCE})

        async def aput(self, namespace, key, value):
            writes.append((namespace, key, value))

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
        return {"assistant_id": ASSISTANT_ID, "only_missing": False}

    response = await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json),
        current_user=_signed_in_owner(),
    )
    assert response.status_code == 202
    assert classifier_calls == [REFERENCE]
    assert writes and writes[0][2]["reference_subject"] == SUBJECT_NON_HUMAN
    assert writes[0][2]["reference_moderation_risk"] == "low"
    assert writes[0][2]["reference_moderation_reasons"] == []
    assert writes[0][2]["reference_image_data"] == REFERENCE
    for _ in range(50):
        await asyncio.sleep(0)
        jobs = list(repository.jobs.values())
        if jobs and jobs[0]["state"] in ("completed", "failed"):
            break
    job = list(repository.jobs.values())[0]
    assert job["detail"]["subject"] == SUBJECT_NON_HUMAN
    assert job["detail"]["summary"]["message"] == ""


@pytest.mark.asyncio
async def test_a_stored_subject_is_reused_without_classifying(monkeypatch):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _fake_vendor(monkeypatch)
    classifier_calls = _fake_subject_classifier(monkeypatch)

    class _Store:
        async def aget(self, namespace, key):
            return SimpleNamespace(
                value={
                    "reference_image_data": REFERENCE,
                    "reference_subject": SUBJECT_STYLIZED_CHARACTER,
                    "reference_moderation_risk": "low",
                    "reference_moderation_reasons": [],
                }
            )

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

    await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json),
        current_user=_signed_in_owner(),
    )
    assert classifier_calls == []
    job = list(repository.jobs.values())[0]
    assert job["detail"]["subject"] == SUBJECT_STYLIZED_CHARACTER


@pytest.mark.asyncio
async def test_subject_classification_falls_back_to_person(monkeypatch):
    from src.anubis.utils import model as model_module

    def _broken_model():
        raise RuntimeError("no image model configured")

    monkeypatch.setattr(model_module, "init_image_description_model", _broken_model)
    result = await reference_subject.classify_reference_subject(REFERENCE)
    assert result == reference_subject.default_assessment()
    assert result["moderation_risk"] == "low"


@pytest.mark.asyncio
async def test_subject_classification_reads_the_structured_answer(monkeypatch):
    from src.anubis.utils import model as model_module

    class _Structured:
        async def ainvoke(self, messages):
            assert messages[1].content[0]["image_url"]["url"] == REFERENCE
            return SimpleNamespace(
                subject="non_human",
                reasoning="A heads-up display with no face.",
                moderation_risk="low",
                moderation_reasons=[],
                moderation_advice="",
            )

    class _Model:
        def with_structured_output(self, schema):
            return _Structured()

    monkeypatch.setattr(model_module, "init_image_description_model", _Model)
    result = await reference_subject.classify_reference_subject(REFERENCE)
    assert result["subject"] == SUBJECT_NON_HUMAN
    assert "heads-up" in result["reasoning"]


# --- predicted refusals: caught before the first call ----------------------


def _vendor_that_must_not_be_called(monkeypatch):
    async def _edit_image(context, **kwargs):
        raise AssertionError("xAI was called for a reference that should be withheld")

    async def _generate_idle_loop(context, **kwargs):
        raise AssertionError("xAI was called for a reference that should be withheld")

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)


HULK_ASSESSMENT = {
    "subject": SUBJECT_STYLIZED_CHARACTER,
    "reasoning": "A green comic-book hero lunging with a raised fist.",
    "moderation_risk": "high",
    "moderation_reasons": ["trademarked_character", "violence"],
    "moderation_advice": "Use a calm head-and-shoulders portrait of an original character.",
}


def test_a_listed_reason_is_a_high_risk_whatever_the_flag_says():
    assessment = reference_subject.normalize_assessment(
        {
            "subject": "person",
            "moderation_risk": "low",
            "moderation_reasons": ["weapon"],
        }
    )
    assert assessment["moderation_risk"] == "high"
    assert reference_subject.moderation_blocks_generation(assessment) is True
    assert reference_subject.moderation_blocks_generation(None) is False
    # Unknown reasons are dropped rather than trusted.
    assessment = reference_subject.normalize_assessment(
        {"moderation_reasons": ["made_up"], "moderation_risk": "low"}
    )
    assert assessment["moderation_reasons"] == []
    assert assessment["moderation_risk"] == "low"
    warning = reference_subject.moderation_warning(HULK_ASSESSMENT)
    assert "nothing was charged" in warning
    assert "a well-known trademarked character and a fighting or attack pose" in warning
    assert "Use a calm head-and-shoulders portrait" in warning
    assert "generate anyway" in warning


def test_the_assessment_round_trips_through_the_store_value():
    fields = reference_subject.assessment_store_fields(HULK_ASSESSMENT)
    assert fields["reference_moderation_risk"] == "high"
    assert fields["reference_moderation_reasons"] == [
        "trademarked_character",
        "violence",
    ]
    restored = reference_subject.assessment_from_store_value(
        {"reference_image_data": REFERENCE, **fields}
    )
    assert restored["subject"] == SUBJECT_STYLIZED_CHARACTER
    assert restored["moderation_reasons"] == ["trademarked_character", "violence"]
    # An image stored before assessments existed is assessed again.
    assert (
        reference_subject.assessment_from_store_value(
            {"reference_image_data": REFERENCE}
        )
        is None
    )
    assert (
        reference_subject.assessment_from_store_value(
            {"reference_image_data": REFERENCE, "reference_subject": "person"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_high_risk_reference_is_withheld_before_any_call(monkeypatch):
    _vendor_that_must_not_be_called(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    frames = []
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        assessment=HULK_ASSESSMENT,
        progress=lambda stage, fields: frames.append((stage, fields)),
    )
    # The owner's own image is still the neutral still; nothing else exists.
    kinds = {
        (a["emotion"], a["asset_kind"])
        for a in await repository.list_emotion_assets(ASSISTANT_ID)
    }
    assert kinds == {("neutral", ASSET_KIND_STILL)}
    assert manifest["withheld"] is True
    assert manifest["subject"] == SUBJECT_STYLIZED_CHARACTER
    assert manifest["moderation_reasons"] == ["trademarked_character", "violence"]
    assert len(manifest["failures"]) == 13
    assert {f["error_code"] for f in manifest["failures"]} == {
        xai_client.ERROR_CODE_MODERATION_PREDICTED
    }
    summary = summarize_failures(manifest["failures"])
    assert summary["predicted"] == 13 and summary["moderated"] == 0
    assert summary["failed_stills"] == 6 and summary["failed_loops"] == 7
    assert "nothing was charged" in summary["message"]
    stage, fields = frames[-1]
    assert stage == "emotion_media_complete"
    assert fields["withheld"] is True and fields["predicted"] == 13
    assert "nothing was charged" in fields["failure_message"]
    assert fields["moderation_reasons"] == ["trademarked_character", "violence"]
    # No progress frame for stills or loops was emitted: nothing ran.
    assert all(stage == "emotion_media_complete" for stage, _ in frames)


@pytest.mark.asyncio
async def test_generate_anyway_attempts_the_calls(monkeypatch):
    calls = _fake_vendor(monkeypatch)
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        InMemoryMediaAssetRepository(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        assessment=HULK_ASSESSMENT,
        proceed_despite_moderation_risk=True,
    )
    assert manifest["withheld"] is False
    assert manifest["complete"] is True
    assert len(calls["edits"]) == 6 and len(calls["videos"]) == 7
    # The assessment's subject picked the character prompt family.
    assert manifest["subject"] == SUBJECT_STYLIZED_CHARACTER


@pytest.mark.asyncio
async def test_regeneration_withholds_a_risky_reference_and_can_be_overridden(
    monkeypatch,
):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    _vendor_that_must_not_be_called(monkeypatch)
    classifier_calls = _fake_subject_classifier(monkeypatch)

    class _Store:
        async def aget(self, namespace, key):
            return SimpleNamespace(
                value={
                    "reference_image_data": REFERENCE,
                    **reference_subject.assessment_store_fields(HULK_ASSESSMENT),
                }
            )

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

    async def _settle():
        for _ in range(50):
            await asyncio.sleep(0)
            jobs = list(repository.jobs.values())
            if jobs and all(j["state"] in ("completed", "failed") for j in jobs):
                break

    async def _json():
        return {"assistant_id": ASSISTANT_ID, "only_missing": False}

    response = await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json),
        current_user=_signed_in_owner(),
    )
    assert response.status_code == 202
    await _settle()
    assert classifier_calls == [], "a stored assessment is reused"
    job = list(repository.jobs.values())[0]
    assert job["state"] == "failed"
    assert job["detail"]["withheld"] is True
    assert job["detail"]["moderation_reasons"] == ["trademarked_character", "violence"]
    assert job["detail"]["summary"]["predicted"] == 13
    assert "nothing was charged" in job["detail"]["summary"]["message"]

    # The manifest tells the settings screen the run was withheld and why.
    async def _owner(assistant_id, current_user):
        return USER_ID

    monkeypatch.setattr(webapp_module, "_assistant_owner_for_media", _owner)
    manifest_response = await webapp_module.get_avatar_emotion_media(
        assistant_id=ASSISTANT_ID, current_user={"API_KEY": "k"}
    )
    last = __import__("json").loads(manifest_response.body)["last_generation"]
    assert last["withheld"] is True
    assert last["moderation_reasons"] == ["trademarked_character", "violence"]
    assert last["failures"][0]["error_code"] == "moderation_predicted"

    # "Generate anyway" reaches the vendor.
    calls = _fake_vendor(monkeypatch)

    async def _json_anyway():
        return {
            "assistant_id": ASSISTANT_ID,
            "only_missing": True,
            "proceed_despite_moderation_risk": True,
        }

    await webapp_module.regenerate_avatar_emotion_media(
        request=SimpleNamespace(json=_json_anyway),
        current_user=_signed_in_owner(),
    )
    await _settle()
    jobs = sorted(repository.jobs.values(), key=lambda j: j["created_at"])
    assert jobs[-1]["state"] == "completed"
    assert jobs[-1]["detail"]["proceed_despite_moderation_risk"] is True
    assert len(calls["edits"]) == 6 and len(calls["videos"]) == 7


def test_a_faceless_subject_is_never_a_trademarked_character():
    """A famous red lens is an object: only the body or face of a character counts."""
    assessment = reference_subject.normalize_assessment(
        {
            "subject": "non_human",
            "moderation_risk": "high",
            "moderation_reasons": ["trademarked_character"],
        }
    )
    assert assessment["moderation_reasons"] == []
    assert assessment["moderation_risk"] == "low"
    # A weapon on a faceless image still counts.
    assessment = reference_subject.normalize_assessment(
        {
            "subject": "non_human",
            "moderation_reasons": ["trademarked_character", "weapon"],
        }
    )
    assert assessment["moderation_reasons"] == ["weapon"]
    assert assessment["moderation_risk"] == "high"
    # A character with a face keeps the reason.
    assessment = reference_subject.normalize_assessment(
        {
            "subject": "stylized_character",
            "moderation_reasons": ["trademarked_character"],
        }
    )
    assert assessment["moderation_reasons"] == ["trademarked_character"]


CREDITS_BODY = (
    'xAI refused the image edit (403): {"code":"permission-denied","error":"Your team '
    "1db9c97a has either used all available credits or reached its monthly spending "
    'limit. To continue, add credits."}'
)


@pytest.mark.asyncio
async def test_an_exhausted_credit_line_stops_the_run_after_one_call(monkeypatch):
    calls = {"edits": 0, "videos": 0}

    async def _edit_image(context, *, reference_image_data_uri, prompt):
        calls["edits"] += 1
        raise xai_client.XaiGenerationError(CREDITS_BODY)

    async def _generate_idle_loop(context, *, still_image_data_uri, prompt):
        calls["videos"] += 1
        raise AssertionError("no loop should be attempted once credits are exhausted")

    monkeypatch.setattr(xai_client, "edit_image", _edit_image)
    monkeypatch.setattr(xai_client, "generate_idle_loop", _generate_idle_loop)
    frames = []
    manifest = await generate_emotion_media_for_avatar(
        _context(),
        InMemoryMediaAssetRepository(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        reference_image_data_uri=REFERENCE,
        progress=lambda stage, fields: frames.append((stage, fields)),
    )
    # The six stills run concurrently, so more than one may have been in
    # flight before the first 403 landed; no loop is ever attempted.
    assert 1 <= calls["edits"] <= 6
    assert calls["videos"] == 0
    codes = [f["error_code"] for f in manifest["failures"]]
    assert xai_client.ERROR_CODE_VENDOR_CREDITS_EXHAUSTED in codes
    assert xai_client.ERROR_CODE_NOT_ATTEMPTED in codes
    assert len(manifest["failures"]) == 13
    summary = summarize_failures(manifest["failures"])
    assert "used all of its available credits" in summary["message"]
    assert "Nothing was generated" in summary["message"]
    stage, fields = frames[-1]
    assert stage == "emotion_media_complete"
    assert fields["failed_stills"] == 6 and fields["failed_loops"] == 7
    assert "credits" in fields["failure_message"]
    # Every not-attempted entry carries the same explanation for the owner.
    assert all(
        "credits" in f["message"]
        for f in manifest["failures"]
        if f["error_code"] == xai_client.ERROR_CODE_NOT_ATTEMPTED
    )


def test_non_human_prompts_keep_the_subjects_own_color():
    for emotion in GENERATED_EMOTIONS:
        prompt = still_prompt_for(emotion, SUBJECT_NON_HUMAN)
        assert "Keep the subject's own identifying colors" in prompt
        assert "do not turn the subject into a character" in prompt
    for emotion in BASE_EMOTIONS:
        loop = idle_loop_prompt_for(emotion, SUBJECT_NON_HUMAN)
        assert "keeps its own colors" in loop
        assert "The light itself pulses" in loop
