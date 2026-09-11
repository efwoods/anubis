"""Unit tests for research starting when an avatar is created.

An avatar used to arrive knowing nothing and looking like nothing until someone
uploaded to it. Now ``POST /create_avatar`` starts the research itself and hands
the job back so the settings screen can show its progress. What is pinned down:

- **The created avatar reports its research job**, which is what lets the client
  attach the existing progress stream without starting a second job.
- **A research failure never costs the avatar.** The assistant row has already
  been written by that point, and a 500 would tell the client no avatar exists.
- **The kill switch stops only the automatic trigger**, leaving the manual
  "Research & verify facts" button alone.
- **No tier gate stands here**, because an avatar created on the free tier is
  exactly the empty avatar this feature exists to prevent.
"""

from types import SimpleNamespace

import pytest

from src.api import webapp as webapp_module

ASSISTANT_NAME = "Ada Lovelace"
CREATOR_ID = "auth0|creator"


def _current_user(user_id=CREATOR_ID):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


class _AssistantsAPI:
    async def create(self, **kwargs):
        return {"assistant_id": kwargs["assistant_id"], "name": kwargs["name"]}


class _StoreClient:
    async def put_item(self, namespace, key, value):
        return None


class _Client:
    def __init__(self):
        self.assistants = _AssistantsAPI()
        self.store = _StoreClient()


@pytest.fixture
def created_avatar_environment(monkeypatch):
    """Everything ``create_avatar`` reaches outside itself, stubbed."""
    monkeypatch.setattr(webapp_module, "get_client", lambda headers=None: _Client())
    monkeypatch.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(
            anonymous_user_id="anonymous",
            admin_user_id="the-admin",
            research_on_create_enabled="true",
        ),
        raising=False,
    )
    return monkeypatch


@pytest.mark.asyncio
async def test_a_created_avatar_reports_the_research_job_it_started(
    created_avatar_environment,
):
    started_with = {}

    def _start(app_state, current_user, **kwargs):
        started_with.update(kwargs)
        return SimpleNamespace(job_id="job-7", status="pending")

    created_avatar_environment.setattr(
        webapp_module, "_start_deep_research_job", _start
    )

    response = await webapp_module.create_avatar(
        name=ASSISTANT_NAME,
        description="A mathematician.",
        research_hint="the one who worked with Babbage",
        current_user=_current_user(),
    )
    import json

    body = json.loads(response.body)
    assert body["research_job"]["job_id"] == "job-7"
    assert body["research_job"]["progress_url"] == "/research_job/job-7/progress"
    # The research is aimed at the avatar that was just created, with the name
    # and hint the creator typed.
    assert started_with["subject_name"] == ASSISTANT_NAME
    assert started_with["research_hint"] == "the one who worked with Babbage"
    assert started_with["assistant_id"] == body["assistant_id"]


@pytest.mark.asyncio
async def test_the_avatar_survives_research_failing_to_start(
    created_avatar_environment,
):
    def _explode(app_state, current_user, **kwargs):
        raise RuntimeError("Tavily is unreachable")

    created_avatar_environment.setattr(
        webapp_module, "_start_deep_research_job", _explode
    )

    response = await webapp_module.create_avatar(
        name=ASSISTANT_NAME, current_user=_current_user()
    )
    import json

    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["assistant_id"]
    # The client is told nothing is running rather than being told nothing exists.
    assert "research_job" not in body


@pytest.mark.asyncio
async def test_the_kill_switch_stops_only_the_automatic_trigger(
    created_avatar_environment,
):
    created_avatar_environment.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(
            anonymous_user_id="anonymous",
            admin_user_id="the-admin",
            research_on_create_enabled="false",
        ),
        raising=False,
    )

    def _must_not_run(app_state, current_user, **kwargs):
        raise AssertionError("research must not start when the switch is off")

    created_avatar_environment.setattr(
        webapp_module, "_start_deep_research_job", _must_not_run
    )

    response = await webapp_module.create_avatar(
        name=ASSISTANT_NAME, current_user=_current_user()
    )
    import json

    body = json.loads(response.body)
    assert response.status_code == 200
    assert "research_job" not in body
    # The manual route is untouched by the switch.
    assert webapp_module._research_on_create_enabled(
        SimpleNamespace(research_on_create_enabled="on")
    )


@pytest.mark.asyncio
async def test_an_anonymous_caller_still_cannot_create_an_avatar(
    created_avatar_environment,
):
    created_avatar_environment.setattr(
        webapp_module,
        "_start_deep_research_job",
        lambda *args, **kwargs: SimpleNamespace(job_id="job-7", status="pending"),
    )
    response = await webapp_module.create_avatar(
        name=ASSISTANT_NAME, current_user=_current_user("anonymous")
    )
    assert response.status_code == 400
