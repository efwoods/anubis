"""The LangSmith location each reply records for the client's trace link."""

from types import SimpleNamespace

from src.anubis.utils.message_record import langsmith_trace_record

RUN_ID = "e309fc6a-b75d-4fe5-bb0c-6527e8f0343e"
WORKSPACE_ID = "f0d930e2-b7c3-462e-8faa-8320f2a735b1"
PRODUCTION_PROJECT_ID = "75a95ebc-1f86-421a-a9e9-0441b25d75ed"


def _context(**overrides):
    settings = {
        "langsmith_tracing": "true",
        "langsmith_project": "anubis",
        "langsmith_workspace_id": WORKSPACE_ID,
        "langsmith_project_id": PRODUCTION_PROJECT_ID,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_record_names_the_workspace_project_and_run_the_reply_was_traced_to():
    assert langsmith_trace_record(RUN_ID, _context()) == {
        "workspace_id": WORKSPACE_ID,
        "project_id": PRODUCTION_PROJECT_ID,
        "project_name": "anubis",
        "run_id": RUN_ID,
    }


def test_no_record_when_tracing_is_off():
    assert langsmith_trace_record(RUN_ID, _context(langsmith_tracing="false")) is None
    assert langsmith_trace_record(RUN_ID, _context(langsmith_tracing=None)) is None


def test_no_record_when_the_workspace_or_project_is_not_configured():
    assert langsmith_trace_record(RUN_ID, _context(langsmith_workspace_id=" ")) is None
    assert langsmith_trace_record(RUN_ID, _context(langsmith_project_id=None)) is None


def test_no_record_without_a_run_id():
    assert langsmith_trace_record(None, _context()) is None
