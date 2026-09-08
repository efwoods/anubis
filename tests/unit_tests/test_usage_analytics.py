"""Opt-in usage analytics: consent, event normalisation, captures, and the routes."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.anubis.utils.postgres_ddl import split_sql_statements
from src.anubis.utils.usage_analytics import repository as repository_module
from src.anubis.utils.usage_analytics import retention as retention_module
from src.anubis.utils.usage_analytics import screenshots as screenshots_module
from src.anubis.utils.usage_analytics.events import (
    UsageEventError,
    normalise_event,
    normalise_event_batch,
    summarise_recent_actions,
)
from src.anubis.utils.usage_analytics.repository import (
    InMemoryUsageAnalyticsRepository,
    PostgresUsageAnalyticsRepository,
    consent_view,
    screenshot_view,
    set_usage_analytics_repository,
)
from src.anubis.utils.usage_analytics.screenshots import (
    describe_and_store,
    describer_system_prompt,
    make_thumbnail,
    usage_analytics_namespace,
)
from src.api import webapp as webapp_module


def _png_bytes(width: int = 1200, height: int = 800) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 20, 20)).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeStore:
    def __init__(self):
        self.items = {}

    async def aput(self, namespace, key, value):
        self.items[(namespace, key)] = value

    async def asearch(self, namespace, query=None, filter=None, limit=10):
        return [
            SimpleNamespace(key=key, value=value)
            for (item_namespace, key), value in self.items.items()
            if item_namespace == namespace
        ][:limit]

    async def adelete(self, namespace, key):
        self.items.pop((namespace, key), None)


class _FakeDescriber:
    def __init__(
        self, description="The person is reading the avatar gallery.", fail=False
    ):
        self.description = description
        self.fail = fail
        self.calls = []

    async def describe(self, image_data, filename):
        self.calls.append((image_data[:30], filename))
        if self.fail:
            raise RuntimeError("vision is down")
        return {
            "description": self.description,
            "model_name": "vision-test",
            "total_cost": 0.001,
        }


# ── events ──────────────────────────────────────────────────────────────────


def test_an_event_is_normalised_and_the_event_fields_win_over_the_batch():
    row = normalise_event(
        {
            "kind": "Click",
            "name": "button:Send",
            "route": "/chat/a1",
            "target": "button#send",
            "occurred_at": "2026-09-07T12:00:00Z",
            "detail": {"x": 1},
        },
        user_id="u1",
        session_id="s1",
        assistant_id="batch-avatar",
        route="/avatars",
    )
    assert row["event_kind"] == "click"
    assert row["route"] == "/chat/a1"
    assert row["assistant_id"] == "batch-avatar"
    assert row["occurred_at"] == datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    assert row["detail"] == {"x": 1}


def test_an_unknown_kind_a_missing_name_and_an_oversized_batch_are_refused():
    with pytest.raises(UsageEventError):
        normalise_event(
            {"kind": "teleport", "name": "x"}, user_id="u1", session_id=None
        )
    with pytest.raises(UsageEventError):
        normalise_event({"kind": "click"}, user_id="u1", session_id=None)
    with pytest.raises(UsageEventError):
        normalise_event_batch(
            [{"kind": "click", "name": "a"}] * 3,
            user_id="u1",
            session_id=None,
            max_events=2,
        )
    with pytest.raises(UsageEventError):
        normalise_event_batch({"kind": "click"}, user_id="u1", session_id=None)


def test_an_oversized_detail_is_trimmed_and_marked():
    row = normalise_event(
        {"kind": "custom", "name": "big", "detail": {"blob": "x" * 5000, "small": 1}},
        user_id="u1",
        session_id=None,
    )
    assert row["detail"].get("truncated") is True
    assert "blob" not in row["detail"]


def test_recent_actions_are_summarised_oldest_first_with_targets_and_routes():
    when = datetime(2026, 9, 7, 12, 0, 5, tzinfo=UTC)
    text = summarise_recent_actions(
        [
            {
                "event_kind": "navigation",
                "event_name": "route",
                "route": "/avatars",
                "occurred_at": when,
            },
            {
                "event_kind": "click",
                "event_name": "button:Open",
                "target": "button",
                "occurred_at": when,
            },
        ]
    )
    assert text.splitlines() == [
        "- 12:00:05 navigation route at /avatars",
        "- 12:00:05 click button:Open on button",
    ]


# ── repository ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consent_defaults_to_off_and_records_where_the_choice_was_made():
    repository = InMemoryUsageAnalyticsRepository()
    assert consent_view(await repository.get_consent("u1"), "u1") == {
        "user_id": "u1",
        "enabled": False,
        "source": None,
        "created_at": None,
        "updated_at": None,
        "recorded": False,
    }
    assert await repository.is_enabled("u1") is False
    row = await repository.set_consent("u1", True, source="signup")
    assert row["enabled"] is True and row["source"] == "signup"
    assert await repository.is_enabled("u1") is True
    off = await repository.set_consent("u1", False, source="nowhere")
    assert off["source"] == "api"
    assert off["created_at"] == row["created_at"]


@pytest.mark.asyncio
async def test_events_and_captures_are_listed_per_user_newest_first_and_purged_by_age():
    repository = InMemoryUsageAnalyticsRepository()
    now = datetime.now(UTC)
    await repository.record_events(
        [
            {
                "user_id": "u1",
                "session_id": "s1",
                "event_kind": "click",
                "event_name": "old",
                "occurred_at": now - timedelta(days=100),
            },
            {
                "user_id": "u1",
                "session_id": "s1",
                "event_kind": "click",
                "event_name": "new",
                "occurred_at": now,
            },
            {
                "user_id": "u2",
                "session_id": "s2",
                "event_kind": "click",
                "event_name": "other",
                "occurred_at": now,
            },
        ]
    )
    await repository.create_screenshot(
        {
            "user_id": "u1",
            "session_id": "s1",
            "route": "/avatars",
            "occurred_at": now - timedelta(days=100),
        }
    )
    kept = await repository.create_screenshot(
        {"user_id": "u1", "session_id": "s1", "route": "/chat/a1", "occurred_at": now}
    )
    listed = await repository.list_events("u1")
    assert [row["event_name"] for row in listed] == ["new", "old"]
    assert [
        row["event_name"] for row in await repository.recent_events("u1", "s1")
    ] == ["old", "new"]
    assert len(await repository.list_events("u2")) == 1

    removed = await retention_module.purge_once(repository, 90)
    assert removed == {"events": 1, "screenshots": 1}
    assert [row["id"] for row in await repository.list_screenshots("u1")] == [
        kept["id"]
    ]
    assert await retention_module.purge_once(repository, 0) == {
        "events": 0,
        "screenshots": 0,
    }

    deleted = await repository.delete_user_data("u1")
    assert deleted == {"events": 1, "screenshots": 1}
    assert await repository.list_events("u2")


def test_the_postgres_ddl_splits_into_single_statements_for_the_prepared_pool():
    statements = split_sql_statements(repository_module._CREATE_TABLES_SQL)
    assert len(statements) == 6
    assert statements[0].startswith(
        "CREATE TABLE IF NOT EXISTS usage_analytics_consent"
    )
    assert all(";" not in statement for statement in statements)


@pytest.mark.asyncio
async def test_the_postgres_repository_upserts_consent_and_inserts_events_with_jsonb():
    class _Cursor:
        def __init__(self, pool):
            self.pool = pool
            self.rowcount = 1

        async def execute(self, statement, params=None, *, prepare=None):
            self.pool.calls.append((" ".join(statement.split()), params))

        async def fetchone(self):
            return self.pool.rows[0] if self.pool.rows else None

        async def fetchall(self):
            return list(self.pool.rows)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Connection:
        def __init__(self, pool):
            self.pool = pool

        def cursor(self):
            return _Cursor(self.pool)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def __init__(self):
            self.calls = []
            self.rows = [
                ("u1", True, "avatar_settings", datetime.now(UTC), datetime.now(UTC))
            ]

        def connection(self):
            return _Connection(self)

    pool = _Pool()
    repository = PostgresUsageAnalyticsRepository(pool)
    row = await repository.set_consent("u1", True, source="avatar_settings")
    assert row["enabled"] is True
    assert "ON CONFLICT (user_id) DO UPDATE" in pool.calls[0][0]

    pool.calls.clear()
    stored = await repository.record_events(
        [
            {
                "user_id": "u1",
                "event_kind": "click",
                "event_name": "a",
                "detail": {"k": 1},
            }
        ]
    )
    assert stored == 1
    statement, params = pool.calls[0]
    assert statement.startswith("INSERT INTO usage_analytics_events")
    assert params[2] == "u1" and params[6] == "click"
    assert type(params[10]).__name__ == "Jsonb"


# ── captures ────────────────────────────────────────────────────────────────


def test_a_capture_is_downscaled_to_a_jpeg_thumbnail():
    thumbnail, mime, width, height = make_thumbnail(
        _png_bytes(1200, 800), max_width=300
    )
    assert mime == "image/jpeg"
    assert (width, height) == (300, 200)
    assert thumbnail[:2] == b"\xff\xd8"


def test_the_describer_prompt_names_the_recent_actions_and_forbids_secrets():
    prompt = describer_system_prompt("- 12:00:00 click button:Send")
    assert "never use the pronoun" in prompt
    assert "passwords" in prompt
    assert prompt.endswith("- 12:00:00 click button:Send")
    assert describer_system_prompt(None) == describer_system_prompt("  ")


@pytest.mark.asyncio
async def test_a_described_capture_lands_on_the_row_and_in_the_store_namespace():
    repository = InMemoryUsageAnalyticsRepository()
    store = _FakeStore()
    created = await repository.create_screenshot(
        {
            "user_id": "u1",
            "session_id": "s1",
            "route": "/avatars",
            "occurred_at": datetime.now(UTC),
        }
    )
    describer = _FakeDescriber()
    completed = await describe_and_store(
        repository=repository,
        store=store,
        screenshot_id=created["id"],
        user_id="u1",
        image_bytes=_png_bytes(64, 64),
        image_mime="image/png",
        recent_actions="- click",
        route="/avatars",
        assistant_id=None,
        session_id="s1",
        occurred_at=datetime.now(UTC),
        describer=describer,
    )
    assert completed["status"] == "described"
    assert describer.calls[0][0].startswith("data:image/png;base64,")
    stored = await repository.get_screenshot("u1", created["id"])
    view = screenshot_view(stored)
    assert view["description"] == "The person is reading the avatar gallery."
    assert view["model_name"] == "vision-test"
    assert view["status"] == "described"
    item = store.items[(usage_analytics_namespace("u1"), created["id"])]
    assert (
        "On /avatars: The person is reading the avatar gallery."
        in item["document"]["kwargs"]["page_content"]
    )
    assert (
        item["document"]["kwargs"]["metadata"]["kind"] == "usage_analytics_screenshot"
    )


@pytest.mark.asyncio
async def test_a_failed_description_is_recorded_as_failed_and_never_raises():
    repository = InMemoryUsageAnalyticsRepository()
    created = await repository.create_screenshot(
        {"user_id": "u1", "occurred_at": datetime.now(UTC)}
    )
    result = await describe_and_store(
        repository=repository,
        store=_FakeStore(),
        screenshot_id=created["id"],
        user_id="u1",
        image_bytes=_png_bytes(8, 8),
        image_mime="image/png",
        recent_actions=None,
        route=None,
        assistant_id=None,
        session_id=None,
        occurred_at=None,
        describer=_FakeDescriber(fail=True),
    )
    assert result is None
    assert (await repository.get_screenshot("u1", created["id"]))["status"] == "failed"


# ── routes ──────────────────────────────────────────────────────────────────


def _user(user_id="u1"):
    # An email marks a signed-in Auth0 account; without one the gating helper
    # treats the caller as anonymous.
    return {
        "identities": [{"user_id": user_id}],
        "user_id": user_id,
        "email": f"{user_id}@example.com",
    }


@pytest.fixture
def published_repository(monkeypatch):
    repository = InMemoryUsageAnalyticsRepository()
    set_usage_analytics_repository(repository)
    monkeypatch.setenv("USAGE_ANALYTICS_ENABLED", "true")
    monkeypatch.setenv("USAGE_ANALYTICS_CAPTURE_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(webapp_module, "_usage_analytics_capture_throttle", None)
    yield repository
    set_usage_analytics_repository(None)


@pytest.mark.asyncio
async def test_events_are_refused_until_the_account_opts_in(published_repository):
    body = webapp_module.UsageAnalyticsEventsRequest(
        session_id="s1",
        route="/avatars",
        events=[{"kind": "click", "name": "button:Open"}],
    )
    with pytest.raises(HTTPException) as refused:
        await webapp_module.record_usage_analytics_events_route(
            body, current_user=_user()
        )
    assert refused.value.status_code == 403

    consent = await webapp_module.set_usage_analytics_consent_route(
        webapp_module.UsageAnalyticsConsentRequest(
            enabled=True, source="avatar_settings"
        ),
        current_user=_user(),
    )
    assert b'"enabled":true' in consent.body
    accepted = await webapp_module.record_usage_analytics_events_route(
        body, current_user=_user()
    )
    assert accepted.status_code == 202
    assert b'"recorded":1' in accepted.body
    assert (await published_repository.list_events("u1"))[0]["route"] == "/avatars"


@pytest.mark.asyncio
async def test_anonymous_callers_and_a_disabled_deployment_are_refused(
    published_repository, monkeypatch
):
    await published_repository.set_consent("u1", True)
    body = webapp_module.UsageAnalyticsEventsRequest(
        events=[{"kind": "click", "name": "a"}]
    )
    monkeypatch.setattr(webapp_module, "is_anonymous_user", lambda user: True)
    with pytest.raises(HTTPException) as anonymous:
        await webapp_module.record_usage_analytics_events_route(
            body, current_user=_user()
        )
    assert anonymous.value.status_code == 403
    monkeypatch.setattr(webapp_module, "is_anonymous_user", lambda user: False)
    monkeypatch.setenv("USAGE_ANALYTICS_ENABLED", "false")
    with pytest.raises(HTTPException) as disabled:
        await webapp_module.record_usage_analytics_events_route(
            body, current_user=_user()
        )
    assert disabled.value.status_code == 404


@pytest.mark.asyncio
async def test_a_malformed_event_batch_is_a_422(published_repository):
    await published_repository.set_consent("u1", True)
    body = webapp_module.UsageAnalyticsEventsRequest(
        events=[{"kind": "teleport", "name": "a"}]
    )
    with pytest.raises(HTTPException) as refused:
        await webapp_module.record_usage_analytics_events_route(
            body, current_user=_user()
        )
    assert refused.value.status_code == 422


@pytest.mark.asyncio
async def test_a_capture_is_stored_with_a_thumbnail_and_described_in_the_background(
    published_repository, monkeypatch
):
    await published_repository.set_consent("u1", True)
    scheduled = []

    async def _fake_describe(**kwargs):
        scheduled.append(kwargs)
        await published_repository.complete_screenshot(
            kwargs["screenshot_id"],
            description="Typing a message",
            model_name="vision-test",
            total_cost=0.0,
            status="described",
        )

    monkeypatch.setattr(screenshots_module, "describe_and_store", _fake_describe)

    class _Upload:
        filename = "capture.png"
        content_type = "image/png"

        async def read(self):
            return _png_bytes(1000, 500)

    response = await webapp_module.record_usage_analytics_screenshot_route(
        request=None,
        file=_Upload(),
        session_id="s1",
        assistant_id="a1",
        thread_id=None,
        route="/chat/a1",
        trigger="interval",
        recent_actions="- click button:Send",
        occurred_at="2026-09-07T12:00:00Z",
        current_user=_user(),
    )
    assert response.status_code == 202
    import asyncio

    await asyncio.sleep(0)
    assert scheduled and scheduled[0]["recent_actions"] == "- click button:Send"
    rows = await published_repository.list_screenshots("u1")
    assert rows[0]["width"] == 640 and rows[0]["height"] == 320
    assert rows[0]["trigger"] == "interval"
    assert rows[0]["description"] == "Typing a message"
    found = await published_repository.get_thumbnail("u1", rows[0]["id"])
    assert found[1] == "image/jpeg"

    summary = await webapp_module.usage_analytics_summary_route(current_user=_user())
    assert b'"screenshots":[{' in summary.body
    assert b"Typing a message" in summary.body
    assert b'"has_thumbnail":true' in summary.body


@pytest.mark.asyncio
async def test_captures_faster_than_the_floor_are_paced_with_retry_after(
    published_repository, monkeypatch
):
    await published_repository.set_consent("u1", True)
    monkeypatch.setenv("USAGE_ANALYTICS_CAPTURE_MIN_INTERVAL_SECONDS", "30")

    async def _noop(**kwargs):
        return None

    monkeypatch.setattr(screenshots_module, "describe_and_store", _noop)

    class _Upload:
        filename = "capture.png"
        content_type = "image/png"

        async def read(self):
            return _png_bytes(10, 10)

    first = await webapp_module.record_usage_analytics_screenshot_route(
        request=None,
        file=_Upload(),
        session_id="s1",
        assistant_id=None,
        thread_id=None,
        route=None,
        trigger=None,
        recent_actions=None,
        occurred_at=None,
        current_user=_user(),
    )
    assert first.status_code == 202
    with pytest.raises(HTTPException) as paced:
        await webapp_module.record_usage_analytics_screenshot_route(
            request=None,
            file=_Upload(),
            session_id="s1",
            assistant_id=None,
            thread_id=None,
            route=None,
            trigger=None,
            recent_actions=None,
            occurred_at=None,
            current_user=_user(),
        )
    assert paced.value.status_code == 429
    assert "Retry-After" in paced.value.headers


@pytest.mark.asyncio
async def test_only_the_administrator_reads_another_account(
    published_repository, monkeypatch
):
    monkeypatch.setenv("ADMIN_USER_ID", "admin")
    await published_repository.record_events(
        [
            {
                "user_id": "u2",
                "event_kind": "click",
                "event_name": "x",
                "occurred_at": datetime.now(UTC),
            }
        ]
    )
    with pytest.raises(HTTPException) as refused:
        await webapp_module.usage_analytics_summary_route(
            user_id="u2", current_user=_user("u1")
        )
    assert refused.value.status_code == 403
    allowed = await webapp_module.usage_analytics_summary_route(
        user_id="u2", current_user=_user("admin")
    )
    assert b'"user_id":"u2"' in allowed.body
    assert b'"event_name":"x"' in allowed.body


@pytest.mark.asyncio
async def test_deleting_data_clears_rows_and_store_documents_but_keeps_consent(
    published_repository, monkeypatch
):
    await published_repository.set_consent("u1", True)
    await published_repository.record_events(
        [
            {
                "user_id": "u1",
                "event_kind": "click",
                "event_name": "x",
                "occurred_at": datetime.now(UTC),
            }
        ]
    )
    store = _FakeStore()
    await store.aput(usage_analytics_namespace("u1"), "shot-1", {"document": {}})
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    response = await webapp_module.delete_usage_analytics_data_route(
        current_user=_user()
    )
    assert b'"events":1' in response.body
    assert store.items == {}
    assert await published_repository.is_enabled("u1") is True


def test_a_route_without_a_repository_answers_503():
    set_usage_analytics_repository(None)
    with pytest.raises(HTTPException) as refused:
        webapp_module._usage_analytics_repository_or_503()
    assert refused.value.status_code == 503


@pytest.mark.asyncio
async def test_the_signup_form_choice_is_recorded_for_the_new_account(
    published_repository,
):
    from src.security.auth import SignupRequest, record_signup_usage_analytics_consent

    assert (
        SignupRequest(email="a@b.c", password="secret1").usage_analytics_opt_in is None
    )
    result = {"api_key": "k", "user_id": "auth0|new"}
    await record_signup_usage_analytics_consent(None, result, True)
    assert result["usage_analytics_enabled"] is True
    assert await published_repository.is_enabled("auth0|new") is True
    # No user id, no row, no error.
    await record_signup_usage_analytics_consent(None, {"api_key": "k"}, True)
    assert len(published_repository.consent) == 1
