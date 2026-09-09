"""Unit tests for the message-grounding safeguard on ``update_self_identity_mem_from_user_txt``.

The avatar's model frequently surfaces facts from its own retrieved consciousness (identity /
quote transcripts injected into the system prompt) when the user merely ASKS about those
topics, then tries to "learn" them as if the user had asserted them. ``_user_message_grounds_fact``
re-checks each proposed fact against the user's MOST RECENT message before anything is stored.
The verification model is stubbed so the deterministic plumbing — latest-message extraction,
previous-assistant-reply extraction, avatar-name lookup, content-block flattening, verdict
combination, and fail-open behaviour — is exercised without a live model.
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage

import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    _assistant_message_before_latest_user_message,
    _assistant_name_from_config,
    _extract_message_text,
    _latest_user_message_text,
    _user_message_grounds_fact,
)


class _FakeVerdict:
    def __init__(
        self,
        proposed_fact_kind: str,
        user_message_role: str,
        supporting_quote_from_user_message: str,
    ):
        self.proposed_fact_kind = proposed_fact_kind
        self.user_message_role = user_message_role
        self.supporting_quote_from_user_message = supporting_quote_from_user_message
        self.reason = "stub"


class _FakeModel:
    """Records the prompt it was invoked with and returns a canned verdict."""

    def __init__(self, verdict: _FakeVerdict):
        self._verdict = verdict
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._verdict


def _stub_model(
    monkeypatch,
    proposed_fact_kind: str = "identity_fact",
    user_message_role: str = "states_the_information",
    supporting_quote_from_user_message: str = "",
) -> _FakeModel:
    """Stub the verifier. The quote defaults to "" — tests that expect a fact to be ACCEPTED
    must pass a quote that really is part of the user message they hand in, because the caller
    verifies it."""
    model = _FakeModel(
        _FakeVerdict(
            proposed_fact_kind, user_message_role, supporting_quote_from_user_message
        )
    )
    monkeypatch.setattr(identity_tools, "init_model", lambda **kwargs: model)
    return model


def test_extract_message_text_handles_str_and_blocks():
    assert _extract_message_text("hello") == "hello"
    assert (
        _extract_message_text(
            [{"type": "text", "text": "a"}, "b", {"type": "image", "url": "x"}]
        )
        == "a b"
    )
    assert _extract_message_text(None) == ""


def test_latest_user_message_text_picks_most_recent_human_message():
    messages = [
        SystemMessage(content="consciousness with retrieved transcript facts"),
        HumanMessage(content="I grew up in Markham."),
        AIMessage(content="Good to know."),
        HumanMessage(content="tell me about University of Toronto"),
    ]
    assert _latest_user_message_text(messages) == "tell me about University of Toronto"
    assert _latest_user_message_text([]) == ""


def test_assistant_message_before_latest_user_message_picks_the_previous_reply():
    """The reply that preceded the latest user message — not a later or earlier one."""
    messages = [
        AIMessage(content="An older reply."),
        HumanMessage(content="I grew up in Markham."),
        AIMessage(content="I believe consciousness survives beyond death."),
        HumanMessage(content="why do you say that?"),
    ]
    assert (
        _assistant_message_before_latest_user_message(messages)
        == "I believe consciousness survives beyond death."
    )


def test_assistant_message_before_latest_user_message_skips_tool_call_only_reply():
    """The tool-calling ``AIMessage`` for THIS turn carries no text and must not be picked."""
    messages = [
        AIMessage(content="I believe consciousness survives beyond death."),
        HumanMessage(content="why do you say that?"),
        AIMessage(content="", tool_calls=[]),
    ]
    assert (
        _assistant_message_before_latest_user_message(messages)
        == "I believe consciousness survives beyond death."
    )


def test_assistant_message_before_latest_user_message_handles_missing_history():
    assert _assistant_message_before_latest_user_message([]) == ""
    assert _assistant_message_before_latest_user_message(None) == ""
    # No assistant reply before the user's first message.
    assert (
        _assistant_message_before_latest_user_message([HumanMessage(content="hello")])
        == ""
    )


def test_assistant_name_from_config_reads_the_avatar_name():
    config = {"configurable": {"assistant_ctx": {"name": "Dr. Jane Goodall"}}}
    assert _assistant_name_from_config(config) == "Dr. Jane Goodall"


def test_assistant_name_from_config_handles_object_and_missing_context():
    class _AssistantContext:
        name = "Shivon Zilis"

    assert (
        _assistant_name_from_config(
            {"configurable": {"assistant_ctx": _AssistantContext()}}
        )
        == "Shivon Zilis"
    )
    assert _assistant_name_from_config({}) == ""
    assert _assistant_name_from_config(None) == ""
    assert _assistant_name_from_config({"configurable": {"assistant_ctx": {}}}) == ""


@pytest.mark.asyncio
async def test_grounds_fact_accepts_identity_fact_the_user_stated(monkeypatch):
    model = _stub_model(
        monkeypatch,
        proposed_fact_kind="identity_fact",
        user_message_role="states_the_information",
        supporting_quote_from_user_message="I grew up in Markham.",
    )
    assert (
        await _user_message_grounds_fact(
            "I grew up in Markham.", "I grew up in Markham."
        )
        is True
    )
    # The proposed fact and the user message are both handed to the verifier.
    (system_message, human_message) = model.calls[0]
    assert isinstance(system_message, SystemMessage)
    assert "I grew up in Markham." in human_message.content


@pytest.mark.asyncio
async def test_grounds_fact_hands_the_verifier_the_avatar_name_and_previous_reply(
    monkeypatch,
):
    """Both extra signals reach the prompt: the user names the avatar in the third person,
    and the avatar's own previous reply is what a user echo would be measured against."""
    model = _stub_model(
        monkeypatch,
        supporting_quote_from_user_message="Dr. Jane Goodall died at the age of 91.",
    )
    await _user_message_grounds_fact(
        "I died at the age of 91.",
        "Dr. Jane Goodall died at the age of 91.",
        assistant_name="Dr. Jane Goodall",
        assistant_previous_message_text="I do not have specific dates available to me.",
    )
    (_system_message, human_message) = model.calls[0]
    assert "ASSISTANT_NAME: Dr. Jane Goodall" in human_message.content
    assert (
        "ASSISTANT_PREVIOUS_MESSAGE: I do not have specific dates available to me."
        in human_message.content
    )


@pytest.mark.asyncio
async def test_grounds_fact_marks_absent_name_and_previous_reply(monkeypatch):
    """Neither signal is required — the check still runs with placeholders."""
    model = _stub_model(
        monkeypatch, supporting_quote_from_user_message="you have twins"
    )
    assert await _user_message_grounds_fact("I have twins.", "you have twins") is True
    (_system_message, human_message) = model.calls[0]
    assert "ASSISTANT_NAME: (not provided)" in human_message.content
    assert "ASSISTANT_PREVIOUS_MESSAGE: (none)" in human_message.content


@pytest.mark.asyncio
async def test_grounds_fact_rejects_question_only_message(monkeypatch):
    """A request ("tell me about X") is not the user asserting the fact -> rejected."""
    _stub_model(monkeypatch, user_message_role="asks_or_requests_only")
    assert (
        await _user_message_grounds_fact(
            "I get to work with the incredible people at University of Toronto.",
            "please tell me about the University of Toronto and spell your name",
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_rejects_the_assistants_own_words_echoed_back(monkeypatch):
    """The avatar said it first; the user quoting it back is not the user asserting it."""
    _stub_model(
        monkeypatch, user_message_role="repeats_what_the_assistant_already_said"
    )
    assert (
        await _user_message_grounds_fact(
            "I believe consciousness survives beyond death.",
            "why do you say that? believe consciousness survives beyond death",
            assistant_previous_message_text="I believe consciousness survives beyond death.",
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_rejects_social_remark(monkeypatch):
    _stub_model(monkeypatch, user_message_role="social_remark_only")
    assert (
        await _user_message_grounds_fact(
            "I am thanked by the people I speak with.",
            "Thank you, Dr. Goodall.",
            assistant_name="Dr. Jane Goodall",
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_rejects_a_request_about_building_the_avatar(monkeypatch):
    """Stated by the user, but a build request is not a fact about the person."""
    _stub_model(
        monkeypatch,
        proposed_fact_kind="request_about_building_the_avatar",
        user_message_role="states_the_information",
        supporting_quote_from_user_message="I want you to sound authentic like shivon zilis",
    )
    assert (
        await _user_message_grounds_fact(
            "I want you to sound authentic like shivon zilis.",
            "I want you to sound authentic like shivon zilis",
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_fails_open_on_empty_message(monkeypatch):
    """No verifiable user text -> do not block (fail open), and never call the model."""
    model = _stub_model(monkeypatch, user_message_role="asks_or_requests_only")
    assert await _user_message_grounds_fact("I grew up in Markham.", "   ") is True
    assert model.calls == []


@pytest.mark.asyncio
async def test_grounds_fact_fails_open_on_model_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(identity_tools, "init_model", _boom)
    assert await _user_message_grounds_fact("I grew up in Markham.", "anything") is True


def test_quote_is_present_in_user_message_ignores_case_and_punctuation():
    """An ordinary transcription difference in the quoted span still matches."""
    assert identity_tools._quote_is_present_in_user_message(
        "died at the age of 91 on october 1 2025",
        "Dr. Jane Goodall died at the age of 91 on October 1, 2025, in California.",
    )


def test_quote_is_present_in_user_message_rejects_absent_and_empty_spans():
    assert not identity_tools._quote_is_present_in_user_message(
        "you have twins", "you went to the University of Toronto"
    )
    assert not identity_tools._quote_is_present_in_user_message(
        "", "you went to the University of Toronto"
    )
    assert not identity_tools._quote_is_present_in_user_message("   ", "anything")


@pytest.mark.asyncio
async def test_grounds_fact_rejects_a_fact_with_no_span_in_the_user_message(monkeypatch):
    """The leak this guard exists to stop: the user's message really does share a fact, so the
    verifier reads it as "the user is telling me things" and waves through a DIFFERENT fact
    carried over from earlier in the conversation. It has no span to quote, so it is refused."""
    _stub_model(
        monkeypatch,
        user_message_role="states_the_information",
        supporting_quote_from_user_message="you have twins",
    )
    assert (
        await _user_message_grounds_fact(
            "I have twins.", "you went to the University of Toronto"
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_rejects_when_the_verifier_quotes_nothing(monkeypatch):
    _stub_model(
        monkeypatch,
        user_message_role="states_the_information",
        supporting_quote_from_user_message="",
    )
    assert (
        await _user_message_grounds_fact("I have twins.", "tell me about your family")
        is False
    )
