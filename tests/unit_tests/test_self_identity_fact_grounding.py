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
        proposed_fact_preserves_meaning: bool = True,
        meaning_difference: str = "",
        sentence_of_user_message_stating_the_action: str = "",
        details_missing_from_that_sentence: list | None = None,
    ):
        self.proposed_fact_kind = proposed_fact_kind
        self.user_message_role = user_message_role
        self.supporting_quote_from_user_message = supporting_quote_from_user_message
        self.proposed_fact_preserves_meaning = proposed_fact_preserves_meaning
        self.meaning_difference = meaning_difference
        self.sentence_of_user_message_stating_the_action = (
            sentence_of_user_message_stating_the_action
        )
        self.details_missing_from_that_sentence = details_missing_from_that_sentence or []
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
    proposed_fact_preserves_meaning: bool = True,
    meaning_difference: str = "",
    sentence_of_user_message_stating_the_action: str = "",
    details_missing_from_that_sentence: list | None = None,
) -> _FakeModel:
    """Stub the verifier. The quote defaults to "" — tests that expect a fact to be ACCEPTED
    must pass a quote that really is part of the user message they hand in, because the caller
    verifies it."""
    model = _FakeModel(
        _FakeVerdict(
            proposed_fact_kind,
            user_message_role,
            supporting_quote_from_user_message,
            proposed_fact_preserves_meaning,
            meaning_difference,
            sentence_of_user_message_stating_the_action,
            details_missing_from_that_sentence,
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
async def test_grounds_fact_fails_closed_on_empty_message(monkeypatch):
    """No verifiable user text -> refuse (fail closed), and never call the model."""
    model = _stub_model(monkeypatch, user_message_role="asks_or_requests_only")
    assert await _user_message_grounds_fact("I grew up in Markham.", "   ") is False
    assert model.calls == []


@pytest.mark.asyncio
async def test_grounds_fact_fails_closed_on_model_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(identity_tools, "init_model", _boom)
    assert await _user_message_grounds_fact("I grew up in Markham.", "anything") is False


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


def test_document_clean_fact_reads_metadata_and_fact_tag():
    """Media identity docs often carry the claim only inside ``<FACT>``; tool-written
    conversation facts carry ``metadata.fact``."""
    from langchain_core.documents import Document

    from src.anubis.utils.tools.identity.identity_tools import (
        _document_clean_fact,
        wrap_fact_with_context,
    )

    metadata_document = Document(
        page_content="ignored wrapper",
        metadata={"fact": "I grew up in Markham, Ontario."},
    )
    assert (
        _document_clean_fact(metadata_document) == "I grew up in Markham, Ontario."
    )

    media_document = Document(
        page_content=wrap_fact_with_context(
            "I believe humans have a lot going on in their own heads.",
            "biographical source",
        ),
        metadata={},
    )
    assert (
        _document_clean_fact(media_document)
        == "I believe humans have a lot going on in their own heads."
    )

    assert _document_clean_fact(Document(page_content="a long quote blob", metadata={})) == ""


@pytest.mark.asyncio
async def test_create_refuses_media_identity_fact_already_in_state(monkeypatch):
    """A fact already held from media identity (``<FACT>`` in state, no metadata.fact)
    must not be re-stored into ``identity_memory`` as if newly taught in chat."""
    from langchain_core.documents import Document
    from langgraph.store.memory import InMemoryStore

    from src.anubis.utils.tools.identity.identity_tools import (
        update_self_identity_mem_from_user_txt,
        wrap_fact_with_context,
    )

    media_fact = "I grew up in Markham, Ontario."
    media_document = Document(
        page_content=wrap_fact_with_context(media_fact, "media biography"),
        metadata={},
    )

    async def _fake_grounds_fact(*args, **kwargs):
        return identity_tools.FactGroundingVerdict(user_message_states_fact=True)

    async def _fake_extract(config):
        return {"user_id": "creator-1"}, {"assistant_id": "asst-1"}

    monkeypatch.setattr(identity_tools, "_judge_fact_against_user_message", _fake_grounds_fact)
    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)

    class _Runtime:
        store = InMemoryStore()
        tool_call_id = "tc-media"
        state = {
            "messages": [HumanMessage(content="tell me about your childhood")],
            "assistant_identity_documents": [media_document],
        }
        config = {
            "configurable": {
                "user_id": "creator-1",
                "assistant_id": "asst-1",
                "assistant_ctx": {"metadata": {"user_id": "creator-1"}},
            }
        }

    command = await update_self_identity_mem_from_user_txt.coroutine(
        fact_shared_about_the_assistant_from_the_user=media_fact,
        fact_context="ctx",
        runtime=_Runtime(),
    )
    assert "previously learned" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_create_always_runs_grounding_even_when_message_embeds_similarly(
    monkeypatch,
):
    """High embedding similarity alone must not accept a fact — grounding always runs."""
    from langgraph.store.memory import InMemoryStore

    from src.anubis.utils.tools.identity.identity_tools import (
        update_self_identity_mem_from_user_txt,
    )

    grounding_calls: list = []

    async def _fake_grounds_fact(fact, user_message_text, **kwargs):
        grounding_calls.append((fact, user_message_text))
        return identity_tools.FactGroundingVerdict(user_message_states_fact=False)

    async def _fake_extract(config):
        return {"user_id": "creator-1"}, {"assistant_id": "asst-1"}

    monkeypatch.setattr(identity_tools, "_judge_fact_against_user_message", _fake_grounds_fact)
    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)

    class _Runtime:
        store = InMemoryStore()
        tool_call_id = "tc-ground"
        state = {
            "messages": [
                HumanMessage(content="please tell me about growing up in Markham")
            ],
            "assistant_identity_documents": [],
        }
        config = {
            "configurable": {
                "user_id": "creator-1",
                "assistant_id": "asst-1",
                "assistant_ctx": {"metadata": {"user_id": "creator-1"}},
            }
        }

    proposed_fact = "I grew up in Markham, Ontario."
    command = await update_self_identity_mem_from_user_txt.coroutine(
        fact_shared_about_the_assistant_from_the_user=proposed_fact,
        fact_context="ctx",
        runtime=_Runtime(),
    )
    assert grounding_calls == [
        (proposed_fact, "please tell me about growing up in Markham")
    ]
    assert "Not learned:" in command.update["messages"][0].content
    assert "Learned:" not in command.update["messages"][0].content


# ---- Meaning check: the rewrite must say only what the user's message says ------------------

DIRT_MESSAGE = (
    "There was this one time we went to get dirt at the old house. You asked me to come "
    "with you. We went to the forest that was next to the house."
)


@pytest.mark.asyncio
async def test_grounds_fact_rejects_a_fact_that_moves_a_place_onto_another_action(
    monkeypatch,
):
    """The fact is sourced from the message, yet says the invitation pointed at the old
    house, which the message never says (avatar 681fe8da, 2026-11-18)."""
    _stub_model(
        monkeypatch,
        supporting_quote_from_user_message="You asked me to come with you",
        proposed_fact_preserves_meaning=False,
        meaning_difference="The message never says where the assistant asked the user to come.",
    )
    proposed_fact = "I asked the user to come with me to the old house."
    assert await _user_message_grounds_fact(proposed_fact, DIRT_MESSAGE) is False

    fact_grounding_verdict = await identity_tools._judge_fact_against_user_message(
        proposed_fact, DIRT_MESSAGE
    )
    assert fact_grounding_verdict.user_message_states_fact is True
    assert fact_grounding_verdict.fact_preserves_meaning is False
    assert fact_grounding_verdict.is_accepted is False
    assert "never says where" in fact_grounding_verdict.meaning_difference


@pytest.mark.asyncio
async def test_grounds_fact_accepts_a_fact_that_keeps_the_meaning(monkeypatch):
    _stub_model(
        monkeypatch,
        supporting_quote_from_user_message="You asked me to come with you",
    )
    fact_grounding_verdict = await identity_tools._judge_fact_against_user_message(
        "I asked the user to come with me.", DIRT_MESSAGE
    )
    assert fact_grounding_verdict.is_accepted is True


def test_grounding_prompt_names_both_changes_of_meaning():
    prompt = identity_tools._FACT_GROUNDING_SYSTEM_PROMPT
    assert "THE MEANING MUST MATCH THE MESSAGE" in prompt
    assert "A DETAIL MOVED BETWEEN SENTENCES" in prompt
    assert "A SHARED ACTION TURNED INTO A CAUSED ACTION" in prompt


@pytest.mark.asyncio
async def test_create_refuses_a_fact_that_changes_the_meaning_and_names_the_change(
    monkeypatch,
):
    from langgraph.store.memory import InMemoryStore

    from src.anubis.utils.tools.identity.identity_tools import (
        update_self_identity_mem_from_user_txt,
    )

    async def _fake_judge(*args, **kwargs):
        return identity_tools.FactGroundingVerdict(
            user_message_states_fact=True,
            fact_preserves_meaning=False,
            meaning_difference="The message never says where the assistant asked the user to come.",
        )

    async def _fake_extract(config):
        return {"user_id": "creator-1"}, {"assistant_id": "asst-1"}

    monkeypatch.setattr(identity_tools, "_judge_fact_against_user_message", _fake_judge)
    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)

    store = InMemoryStore()

    class _Runtime:
        tool_call_id = "tc-meaning"
        state = {
            "messages": [HumanMessage(content=DIRT_MESSAGE)],
            "assistant_identity_documents": [],
        }
        config = {
            "configurable": {
                "user_id": "creator-1",
                "assistant_id": "asst-1",
                "assistant_ctx": {"metadata": {"user_id": "creator-1"}},
            }
        }

    _Runtime.store = store
    command = await update_self_identity_mem_from_user_txt.coroutine(
        fact_shared_about_the_assistant_from_the_user=(
            "I asked the user to come with me to the old house."
        ),
        runtime=_Runtime(),
    )
    content = command.update["messages"][0].content
    assert content.startswith("Not learned:")
    assert "changes the meaning of the user's most recent message" in content
    assert "never says where" in content
    assert await store.asearch(("creator-1", "asst-1", "identity_memory")) == []


@pytest.mark.asyncio
async def test_grounds_fact_hands_the_verifier_the_users_name(monkeypatch):
    """Without the user's name the verifier read "Evan" as a person the fact added."""
    model = _stub_model(
        monkeypatch, supporting_quote_from_user_message="there was a snake next to me"
    )
    await identity_tools._judge_fact_against_user_message(
        "There was a snake next to Evan.",
        "You said watch out! there was a snake next to me.",
        user_name="Evan",
    )
    human_prompt = model.calls[0][1].content
    assert "USER_NAME: Evan" in human_prompt
    assert "HOW PROPOSED_FACT REFERS TO THE USER" in identity_tools._FACT_GROUNDING_SYSTEM_PROMPT


def test_user_name_from_runtime_reads_state_then_configurable():
    class _RuntimeWithState:
        state = {"user_state": {"user_name": "Evan"}}
        config = {"configurable": {"user_ctx": {"name": "Other"}}}

    class _RuntimeWithConfigOnly:
        state = {}
        config = {"configurable": {"user_ctx": {"name": "Evan"}}}

    class _RuntimeWithNothing:
        state = {}
        config = {}

    assert identity_tools._user_name_from_runtime(_RuntimeWithState()) == "Evan"
    assert identity_tools._user_name_from_runtime(_RuntimeWithConfigOnly()) == "Evan"
    assert identity_tools._user_name_from_runtime(_RuntimeWithNothing()) == ""


@pytest.mark.asyncio
async def test_a_listed_missing_detail_refuses_the_fact_even_when_the_flag_says_preserved(
    monkeypatch,
):
    """The small verifier model lists the moved detail more reliably than the model sets
    the yes-or-no flag, so a non-empty list alone refuses the fact."""
    _stub_model(
        monkeypatch,
        supporting_quote_from_user_message="You asked me to come with you",
        proposed_fact_preserves_meaning=True,
        sentence_of_user_message_stating_the_action="You asked me to come with you.",
        details_missing_from_that_sentence=["to the old house"],
    )
    fact_grounding_verdict = await identity_tools._judge_fact_against_user_message(
        "I asked the user to come with me to the old house.", DIRT_MESSAGE
    )
    assert fact_grounding_verdict.user_message_states_fact is True
    assert fact_grounding_verdict.fact_preserves_meaning is False
    assert "to the old house" in fact_grounding_verdict.meaning_difference
    assert "You asked me to come with you." in fact_grounding_verdict.meaning_difference
