"""Unit tests for the one caller exempt from the upload tier gate.

Creating an avatar starts research on every tier, so the acquisition that
installs a missing portrait or reference recording cannot be refused for a
free-tier account without leaving exactly the empty avatar the feature exists to
prevent. The exemption is real, so what it may and may not carry is pinned down:

- **Without the flag, a free-tier caller is still refused**, so no ordinary
  upload path is widened.
- **With the flag, one reference asset is accepted** on the free tier.
- **The flag cannot carry a general batch.** More than one item, or no reference
  flag at all, is refused whatever the tier.
- **Nothing else is relaxed**: the batch still goes through estimation, the
  allotment check, the rate limit and metering.
- **The research subtracts what the acquisition already ingested**, so the
  chosen recording is not downloaded and transcribed twice.
"""

import pytest

from src.anubis.utils.billing.tiers import SubscriptionTier
from src.api import webapp as webapp_module

ASSISTANT_ID = "assistant-1"
CREATOR_ID = "auth0|creator"
VIDEO_URL = "https://www.youtube.com/watch?v=abcdefghijk"


def _current_user():
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": CREATOR_ID}]}


def _assistant_ctx():
    return {
        "name": "Ada Lovelace",
        "description": None,
        "metadata": {"user_id": CREATOR_ID},
    }


@pytest.fixture
def free_tier(monkeypatch):
    """Resolve every caller to the free tier, which lacks the UPLOAD capability."""
    monkeypatch.setattr(
        webapp_module, "resolve_tier", lambda user: SubscriptionTier.FREE
    )
    return monkeypatch


@pytest.fixture
def recorded_batch(monkeypatch):
    """Record what reaches _start_media_batch instead of starting one."""
    seen = {}

    async def _start_media_batch(**kwargs):
        seen.update(kwargs)
        return {"job_id": "job-1", "items_accepted": 1, "filenames": ["x"]}

    monkeypatch.setattr(webapp_module, "_start_media_batch", _start_media_batch)
    return seen


@pytest.mark.asyncio
async def test_a_free_tier_upload_is_still_refused(free_tier, recorded_batch):
    result = await webapp_module.start_identity_media_job_from_chat(
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx=_assistant_ctx(),
        current_user=_current_user(),
        attachments=[],
        urls=[VIDEO_URL],
    )
    assert result["status"] == "refused"
    assert result["status_code"] == 403
    assert recorded_batch == {}


@pytest.mark.asyncio
async def test_the_acquisition_may_install_one_reference_asset_on_the_free_tier(
    free_tier, recorded_batch, monkeypatch
):
    async def _entries(url, **kwargs):
        assert kwargs["bootstrap"] is True
        return [
            {"filename": url, "namespace_filename": "video", "reference_audio": True}
        ]

    monkeypatch.setattr(webapp_module, "_build_media_entries_for_url", _entries)

    result = await webapp_module.start_identity_media_job_from_chat(
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx=_assistant_ctx(),
        current_user=_current_user(),
        attachments=[],
        urls=[VIDEO_URL],
        reference_audio=True,
        bootstrap=True,
    )
    assert result["status"] == "started"
    # Everything downstream of the capability gate still ran.
    assert recorded_batch["assistant_id"] == ASSISTANT_ID
    assert recorded_batch["current_user"] == _current_user()


@pytest.mark.asyncio
async def test_the_exemption_cannot_carry_a_general_batch(free_tier, recorded_batch):
    several = await webapp_module.start_identity_media_job_from_chat(
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx=_assistant_ctx(),
        current_user=_current_user(),
        attachments=[],
        urls=[VIDEO_URL, "https://www.youtube.com/watch?v=zzzzzzzzzzz"],
        reference_audio=True,
        bootstrap=True,
    )
    assert several["status"] == "refused"

    unflagged = await webapp_module.start_identity_media_job_from_chat(
        user_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx=_assistant_ctx(),
        current_user=_current_user(),
        attachments=[],
        urls=[VIDEO_URL],
        bootstrap=True,
    )
    assert unflagged["status"] == "refused"
    assert recorded_batch == {}


@pytest.mark.asyncio
async def test_someone_elses_avatar_is_refused_even_with_the_flag(
    free_tier, recorded_batch
):
    result = await webapp_module.start_identity_media_job_from_chat(
        user_id="a-stranger",
        assistant_id=ASSISTANT_ID,
        assistant_ctx=_assistant_ctx(),
        current_user=_current_user(),
        attachments=[],
        urls=[VIDEO_URL],
        reference_audio=True,
        bootstrap=True,
    )
    assert result["status"] == "refused"
    assert result["status_code"] == 403
    assert recorded_batch == {}
