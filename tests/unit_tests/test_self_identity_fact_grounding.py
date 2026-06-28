"""Unit tests for the message-grounding safeguard on ``update_self_identity_mem_from_user_txt``.

The avatar's model frequently surfaces facts from its own retrieved consciousness (identity /
quote transcripts injected into the system prompt) when the user merely ASKS about those
topics, then tries to "learn" them as if the user had asserted them. ``_user_message_grounds_fact``
re-checks each proposed fact against the user's MOST RECENT message before anything is stored.
The verification model is stubbed so the deterministic plumbing — latest-message extraction,
content-block flattening, verdict passthrough, and fail-open behaviour — is exercised without a
live model.
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage

import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    _extract_message_text,
    _latest_user_message_text,
    _user_message_grounds_fact,
)


class _FakeVerdict:
    def __init__(self, user_asserted_the_fact: bool):
        self.user_asserted_the_fact = user_asserted_the_fact
        self.reason = "stub"


class _FakeModel:
    """Records the prompt it was invoked with and returns a canned verdict."""

    def __init__(self, verdict: bool):
        self._verdict = verdict
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return _FakeVerdict(self._verdict)


def _stub_model(monkeypatch, verdict: bool) -> _FakeModel:
    model = _FakeModel(verdict)
    monkeypatch.setattr(identity_tools, "init_model", lambda **kwargs: model)
    return model


def test_extract_message_text_handles_str_and_blocks():
    assert _extract_message_text("hello") == "hello"
    assert _extract_message_text(
        [{"type": "text", "text": "a"}, "b", {"type": "image", "url": "x"}]
    ) == "a b"
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


@pytest.mark.asyncio
async def test_grounds_fact_passes_through_true_verdict(monkeypatch):
    model = _stub_model(monkeypatch, verdict=True)
    assert await _user_message_grounds_fact("I grew up in Markham.", "I grew up in Markham.") is True
    # The proposed fact and the user message are both handed to the verifier.
    (system_message, human_message) = model.calls[0]
    assert isinstance(system_message, SystemMessage)
    assert "I grew up in Markham." in human_message.content


@pytest.mark.asyncio
async def test_grounds_fact_rejects_question_only_message(monkeypatch):
    """A request ("tell me about X") is not the user asserting the fact -> rejected."""
    _stub_model(monkeypatch, verdict=False)
    assert (
        await _user_message_grounds_fact(
            "I get to work with the incredible people at University of Toronto.",
            "please tell me about the University of Toronto and spell your name",
        )
        is False
    )


@pytest.mark.asyncio
async def test_grounds_fact_fails_open_on_empty_message(monkeypatch):
    """No verifiable user text -> do not block (fail open), and never call the model."""
    model = _stub_model(monkeypatch, verdict=False)
    assert await _user_message_grounds_fact("I grew up in Markham.", "   ") is True
    assert model.calls == []


@pytest.mark.asyncio
async def test_grounds_fact_fails_open_on_model_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(identity_tools, "init_model", _boom)
    assert await _user_message_grounds_fact("I grew up in Markham.", "anything") is True
