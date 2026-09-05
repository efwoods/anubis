"""The avatar's summarization middleware and its cross-turn summary event.

deepagents keeps the summarization event in a private state key on the deep
agent's own thread, and the avatar runs the deep agent on a fresh thread every
turn. These tests pin the three things that make compaction survive a turn:
the avatar's summarizer takes the built-in's slot (merged by name), a summary
event handed in through the public key is applied without a new summary call,
and a freshly created event is mirrored onto the public key so the outer
workflow can store it.
"""

import pytest
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.graph import _apply_custom_middleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.middleware.avatar_summarization import (
    AVATAR_CONVERSATION_SUMMARY_PROMPT,
    CONVERSATION_SUMMARY_EVENT_KEY,
    CONVERSATION_SUMMARY_SESSION_ID_KEY,
    SUMMARIZATION_EVENT_KEY,
    AvatarSummarizationMiddleware,
    build_avatar_summarization_middleware,
    clamp_summary_event,
)


def _request(model, messages, state):
    return ModelRequest(
        model=model,
        messages=list(messages),
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=None,
        model_settings={},
    )


def test_the_avatar_summarizer_replaces_the_built_in_by_name():
    model = FakeListChatModel(responses=["summary"])
    built_in = create_summarization_middleware(model, StateBackend())
    avatar = build_avatar_summarization_middleware(GlobalContext(), model)
    assert avatar.name == built_in.name == "SummarizationMiddleware"
    merged = _apply_custom_middleware([built_in], [avatar])
    assert merged == [avatar]
    assert isinstance(merged[0], AvatarSummarizationMiddleware)


def test_the_summary_prompt_keeps_the_messages_contract():
    assert "\n<messages>\n" in AVATAR_CONVERSATION_SUMMARY_PROMPT
    assert "{messages}" in AVATAR_CONVERSATION_SUMMARY_PROMPT
    assert "AMBIENT OBSERVATIONS" in AVATAR_CONVERSATION_SUMMARY_PROMPT
    assert AVATAR_CONVERSATION_SUMMARY_PROMPT.count("{") == 1


@pytest.mark.asyncio
async def test_a_public_event_is_applied_without_a_new_summary(tmp_path):
    model = FakeListChatModel(responses=["should not be called"])
    middleware = AvatarSummarizationMiddleware(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path)),
        trigger=("tokens", 10_000_000),
        keep=("messages", 1),
    )
    messages = [
        HumanMessage(id="1", content="one"),
        AIMessage(id="2", content="two"),
        HumanMessage(id="3", content="three"),
    ]
    summary = HumanMessage(
        id="s",
        content="Here is a summary",
        additional_kwargs={"lc_source": "summarization"},
    )
    state = {
        "messages": messages,
        CONVERSATION_SUMMARY_EVENT_KEY: {
            "cutoff_index": 2,
            "summary_message": summary,
            "file_path": None,
        },
        CONVERSATION_SUMMARY_SESSION_ID_KEY: "session_abc",
    }
    seen = {}

    async def handler(request):
        seen["messages"] = list(request.messages)
        seen["state_event"] = request.state.get(SUMMARIZATION_EVENT_KEY)
        return AIMessage(content="reply")

    response = await middleware.awrap_model_call(
        _request(model, messages, state), handler
    )
    assert [m.content for m in seen["messages"]] == ["Here is a summary", "three"]
    assert seen["state_event"]["cutoff_index"] == 2
    assert not isinstance(response, ExtendedModelResponse)


@pytest.mark.asyncio
async def test_a_new_event_is_mirrored_onto_the_public_key(tmp_path):
    model = FakeListChatModel(responses=["the summary"])
    middleware = AvatarSummarizationMiddleware(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path)),
        trigger=("messages", 3),
        keep=("messages", 1),
    )
    messages = [
        HumanMessage(id="1", content="one"),
        AIMessage(id="2", content="two"),
        HumanMessage(id="3", content="three"),
        AIMessage(id="4", content="four"),
        HumanMessage(id="5", content="five"),
    ]

    async def handler(request):
        return AIMessage(content="reply")

    response = await middleware.awrap_model_call(
        _request(model, messages, {"messages": messages}), handler
    )
    assert isinstance(response, ExtendedModelResponse)
    update = response.command.update
    assert SUMMARIZATION_EVENT_KEY in update
    assert update[CONVERSATION_SUMMARY_EVENT_KEY] is update[SUMMARIZATION_EVENT_KEY]
    assert update[CONVERSATION_SUMMARY_EVENT_KEY]["cutoff_index"] > 0
    assert (
        "the summary"
        in update[CONVERSATION_SUMMARY_EVENT_KEY]["summary_message"].content
    )
    assert update[CONVERSATION_SUMMARY_SESSION_ID_KEY]


def test_the_cutoff_is_clamped_to_the_outer_conversation():
    summary = HumanMessage(content="s")
    assert clamp_summary_event(None, outer_message_count=5) is None
    assert clamp_summary_event({"cutoff_index": "x"}, outer_message_count=5) is None
    assert clamp_summary_event({"cutoff_index": 3}, outer_message_count=5) is None
    kept = clamp_summary_event(
        {"cutoff_index": 3, "summary_message": summary, "file_path": None},
        outer_message_count=5,
    )
    assert kept["cutoff_index"] == 3
    clamped = clamp_summary_event(
        {"cutoff_index": 9, "summary_message": summary, "file_path": None},
        outer_message_count=5,
    )
    assert clamped["cutoff_index"] == 5 and clamped["summary_message"] is summary


def test_the_factory_reads_the_deep_agent_settings(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_SUMMARIZATION_MAX_TOKENS", "5000")
    monkeypatch.setenv("DEEP_AGENT_SUMMARIZATION_KEEP_LAST_N_MESSAGES", "4")
    middleware = build_avatar_summarization_middleware(
        GlobalContext(), FakeListChatModel(responses=["x"])
    )
    helper = (
        middleware._lc_helper
    )  # the LangChain summarizer the deepagents class wraps
    assert helper.trigger == ("tokens", 5000)
    assert helper.keep == ("messages", 4)
    assert helper.summary_prompt == AVATAR_CONVERSATION_SUMMARY_PROMPT
