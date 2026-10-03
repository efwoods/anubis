"""Replies stored without a LangSmith record are matched to their root run."""

import asyncio
from types import SimpleNamespace

from src.anubis.utils import langsmith_trace_lookup
from src.anubis.utils.langsmith_trace_lookup import (
    find_langsmith_record_for_human_turn,
    human_message_ids_of_run_inputs,
    lookup_enabled,
    lookup_project_ids,
    mark_replies_for_lookup,
    root_run_of_human_turn,
    root_run_start_window_filter,
)

WORKSPACE_ID = "f0d930e2-b7c3-462e-8faa-8320f2a735b1"
PRODUCTION_PROJECT_ID = "75a95ebc-1f86-421a-a9e9-0441b25d75ed"
LOCAL_TESTING_PROJECT_ID = "aea5ce01-cd9f-498a-859c-ad0facb45f35"
ADMINISTRATOR_USER_ID = "administrator-user"
THREAD_ID = "479cd3ac-5d40-479a-a149-7407b8c38f5e"


def _root_run(run_id, project_id, human_message_id):
    return {
        "id": run_id,
        "session_id": project_id,
        "inputs": {"messages": [{"type": "human", "id": human_message_id, "content": "hi"}]},
    }


def _development_context(**overrides):
    settings = {
        "dev": "TRUE",
        "admin_user_id": ADMINISTRATOR_USER_ID,
        "langsmith_workspace_id": WORKSPACE_ID,
        "langsmith_trace_lookup_project_ids": f"{LOCAL_TESTING_PROJECT_ID}, {PRODUCTION_PROJECT_ID}",
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_each_reply_without_a_record_names_the_human_turn_the_reply_answers():
    recorded_trace = {"workspace_id": WORKSPACE_ID, "project_id": PRODUCTION_PROJECT_ID, "run_id": "run-recorded"}
    messages = [
        {"type": "human", "id": "human-hidden", "additional_kwargs": {"hidden": True, "created_at": "2026-09-28T15:53:54+00:00"}},
        {"type": "ai", "id": "ai-1", "response_metadata": {"sentiment": "joy"}},
        {"type": "human", "id": "human-2", "additional_kwargs": {}},
        {"type": "ai", "id": "ai-2", "response_metadata": {"langsmith": recorded_trace}},
        {"type": "ai", "id": "ai-orphan"},
    ]
    messages.insert(0, {"type": "ai", "id": "ai-before-any-human"})

    assert mark_replies_for_lookup(messages) == 2
    assert "response_metadata" not in messages[0]
    assert messages[2]["response_metadata"] == {
        "sentiment": "joy",
        "langsmith_lookup": {
            "human_message_id": "human-hidden",
            "human_created_at": "2026-09-28T15:53:54+00:00",
        },
    }
    assert messages[4]["response_metadata"] == {"langsmith": recorded_trace}
    # A second reply to the same human turn is marked with that turn too.
    assert messages[5]["response_metadata"]["langsmith_lookup"] == {
        "human_message_id": "human-2",
        "human_created_at": None,
    }


def test_the_root_run_is_the_run_started_by_the_human_turn():
    root_runs = [
        _root_run("run-development", LOCAL_TESTING_PROJECT_ID, "human-2"),
        _root_run("run-production", PRODUCTION_PROJECT_ID, "human-1"),
    ]
    assert root_run_of_human_turn(root_runs, "human-1")["id"] == "run-production"
    assert root_run_of_human_turn(root_runs, "human-9") is None


def test_human_message_ids_are_read_from_both_serialization_shapes():
    assert human_message_ids_of_run_inputs(
        {
            "messages": [
                {"type": "human", "id": "plain"},
                {"lc": 1, "kwargs": {"type": "human", "id": "constructor"}},
                {"type": "ai", "id": "not-human"},
            ]
        }
    ) == ["plain", "constructor"]
    assert human_message_ids_of_run_inputs({"input": None}) == []
    assert human_message_ids_of_run_inputs(None) == []


def test_the_search_window_brackets_the_human_turn():
    run_filter = root_run_start_window_filter(THREAD_ID, "2026-09-28T15:53:54.123456+00:00")
    assert run_filter == (
        'and(eq(metadata_key, "thread_id"), eq(metadata_value, "' + THREAD_ID + '"), '
        'gte(start_time, "2026-09-28T15:53:49.123456"), '
        'lte(start_time, "2026-09-28T15:55:54.123456"))'
    )
    # An instant in another zone is converted to UTC, as LangSmith stores it.
    assert 'gte(start_time, "2026-09-28T15:53:49.000000")' in root_run_start_window_filter(
        THREAD_ID, "2026-09-28T08:53:54-07:00"
    )
    # Without an instant the whole thread is searched.
    assert root_run_start_window_filter(THREAD_ID, None) == (
        'and(eq(metadata_key, "thread_id"), eq(metadata_value, "' + THREAD_ID + '"))'
    )


def test_lookup_runs_for_every_signed_in_account_in_development_only():
    assert lookup_project_ids(_development_context()) == [
        LOCAL_TESTING_PROJECT_ID,
        PRODUCTION_PROJECT_ID,
    ]
    assert lookup_enabled(_development_context(), ADMINISTRATOR_USER_ID) is True
    assert lookup_enabled(_development_context(), "someone-else") is True
    assert lookup_enabled(_development_context(), "") is False
    assert lookup_enabled(_development_context(dev="FALSE"), ADMINISTRATOR_USER_ID) is False
    assert (
        lookup_enabled(
            _development_context(langsmith_trace_lookup_project_ids=""),
            ADMINISTRATOR_USER_ID,
        )
        is False
    )
    assert (
        lookup_enabled(_development_context(langsmith_workspace_id=None), ADMINISTRATOR_USER_ID)
        is False
    )


def test_a_production_turn_opened_from_development_resolves_to_the_production_project(monkeypatch):
    requested_project_ids = []

    def read_root_runs(run_filter, project_ids):
        requested_project_ids.append(project_ids)
        return [
            _root_run("run-development", LOCAL_TESTING_PROJECT_ID, "human-2"),
            _root_run("run-production", PRODUCTION_PROJECT_ID, "human-1"),
        ]

    monkeypatch.setattr(langsmith_trace_lookup, "_read_root_runs", read_root_runs)
    monkeypatch.setitem(
        langsmith_trace_lookup._project_names_by_project_id, PRODUCTION_PROJECT_ID, "anubis"
    )
    langsmith_record = asyncio.run(
        find_langsmith_record_for_human_turn(
            thread_id=THREAD_ID,
            human_message_id="human-1",
            human_created_at="2026-09-28T15:53:54+00:00",
            context=_development_context(),
        )
    )
    assert requested_project_ids == [[LOCAL_TESTING_PROJECT_ID, PRODUCTION_PROJECT_ID]]
    assert langsmith_record == {
        "workspace_id": WORKSPACE_ID,
        "project_id": PRODUCTION_PROJECT_ID,
        "project_name": "anubis",
        "run_id": "run-production",
    }
