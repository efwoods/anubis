"""Automatic conversation naming: what gets named, and what is left alone.

The messaging service names a conversation twice — once on the turn that starts
it, and again when the reader leaves it — and the two rules that matter are that
a name the reader typed is never replaced, and that naming a conversation never
costs the conversation any of the other metadata stored beside its name.

Every test here replaces the structured-output call
(``conversation_titles.invoke_structured``) and the platform client
(``langgraph_sdk.get_client``), so no model and no database are needed.
"""

import pytest

import src.anubis.utils.conversation_titles as conversation_titles
from src.anubis.utils.conversation_titles import (
    AUTOMATIC_TITLE_SOURCE,
    MANUAL_TITLE_SOURCE,
    ConversationTitle,
    conversation_title_may_be_replaced,
    generate_conversation_title,
    name_conversation_thread,
    stored_conversation_title,
    visible_conversation_messages,
)


class _FakeThreadsClient:
    """The two thread calls the namer makes, over one in-memory thread record."""

    def __init__(self, thread: dict):
        self.thread = thread
        self.messages: list = []
        self.updates: list[dict] = []

    async def get(self, thread_id: str):
        return self.thread

    async def get_state(self, thread_id: str):
        return {"values": {"messages": self.messages}}

    async def update(self, thread_id: str, metadata: dict):
        self.updates.append(metadata)
        self.thread["metadata"] = {**self.thread.get("metadata", {}), **metadata}


class _FakeLangGraphClient:
    def __init__(self, threads: _FakeThreadsClient):
        self.threads = threads


@pytest.fixture
def fake_client(monkeypatch):
    """A platform client over one thread, installed for ``get_client``."""

    def _install(thread: dict, messages: list) -> _FakeThreadsClient:
        threads = _FakeThreadsClient(thread)
        threads.messages = messages
        import langgraph_sdk

        monkeypatch.setattr(
            langgraph_sdk, "get_client", lambda **kwargs: _FakeLangGraphClient(threads)
        )
        return threads

    return _install


@pytest.fixture
def named_by_the_model(monkeypatch):
    """Make the structured-output call answer with a fixed title."""

    def _install(title: str) -> list[str]:
        transcripts: list[str] = []

        async def _invoke(response_format, system_prompt, human_text):
            transcripts.append(human_text)
            return ConversationTitle(conversation_title=title)

        monkeypatch.setattr(conversation_titles, "invoke_structured", _invoke)
        return transcripts

    return _install


def _conversation() -> list[dict]:
    return [
        {"type": "human", "content": "My deployment pipeline broke this morning."},
        {"type": "ai", "content": "Tell me what the last green build changed."},
    ]


def test_a_thread_named_after_itself_counts_as_unnamed():
    assert stored_conversation_title({"conversation_title": "abc-123"}, "abc-123") == ""
    assert stored_conversation_title({"conversation_title": "  "}, "abc-123") == ""
    assert (
        stored_conversation_title({"conversation_title": "Dinner plans"}, "abc-123")
        == "Dinner plans"
    )


def test_a_name_the_reader_typed_is_never_replaced():
    reader_named = {
        "conversation_title": "Dinner plans",
        "conversation_title_source": MANUAL_TITLE_SOURCE,
    }
    service_named = {
        "conversation_title": "Broken deployment pipeline",
        "conversation_title_source": AUTOMATIC_TITLE_SOURCE,
    }
    assert conversation_title_may_be_replaced(reader_named, "abc-123") is False
    assert conversation_title_may_be_replaced(service_named, "abc-123") is True
    assert conversation_title_may_be_replaced({}, "abc-123") is True


def test_a_scene_the_camera_noticed_does_not_name_the_conversation():
    messages = [
        {
            "type": "human",
            "content": "[AMBIENT_OBSERVATION id=1] A cat on the desk",
            "additional_kwargs": {"hidden": True},
        },
        {"type": "human", "content": "What did I ask you yesterday?"},
    ]
    visible = visible_conversation_messages(messages)
    assert len(visible) == 1
    assert visible[0]["content"] == "What did I ask you yesterday?"


@pytest.mark.asyncio
async def test_a_title_is_trimmed_of_the_punctuation_a_model_adds(named_by_the_model):
    named_by_the_model('"Broken deployment pipeline."')
    assert (
        await generate_conversation_title(_conversation())
        == "Broken deployment pipeline"
    )


@pytest.mark.asyncio
async def test_a_long_title_is_cut_on_a_word_boundary(named_by_the_model, monkeypatch):
    named_by_the_model(
        "The morning the deployment pipeline broke and everything else with it"
    )
    monkeypatch.setenv("CONVERSATION_TITLE_MAX_CHARACTERS", "30")
    title = await generate_conversation_title(_conversation())
    assert len(title) <= 31  # the ellipsis is one character past the ceiling
    assert title.endswith("…")
    assert not title.replace("…", "").endswith(" ")


@pytest.mark.asyncio
async def test_an_empty_conversation_is_not_named(named_by_the_model):
    named_by_the_model("Something")
    assert await generate_conversation_title([]) == ""


@pytest.mark.asyncio
async def test_a_model_that_cannot_be_reached_leaves_the_conversation_unnamed(
    monkeypatch,
):
    async def _refuse(response_format, system_prompt, human_text):
        raise RuntimeError("no classification model here")

    monkeypatch.setattr(conversation_titles, "invoke_structured", _refuse)
    assert await generate_conversation_title(_conversation()) == ""


@pytest.mark.asyncio
async def test_naming_a_thread_keeps_the_metadata_stored_beside_the_name(
    fake_client, named_by_the_model
):
    named_by_the_model("Broken deployment pipeline")
    threads = fake_client(
        {
            "thread_id": "abc-123",
            "metadata": {
                "graph_id": "Anubis",
                "thread_metadata": {
                    "user_id": "user-1",
                    "assistant_id": "avatar-1",
                    "shared": True,
                    "most_recent_message": "2026-09-09T10:00:00+00:00",
                },
            },
        },
        _conversation(),
    )

    title = await name_conversation_thread(
        "abc-123",
        langgraph_client_headers={"API-KEY": "k"},
        user_id="user-1",
        assistant_id="avatar-1",
    )

    assert title == "Broken deployment pipeline"
    written = threads.updates[-1]["thread_metadata"]
    assert written["conversation_title"] == "Broken deployment pipeline"
    assert written["conversation_title_source"] == AUTOMATIC_TITLE_SOURCE
    assert written["shared"] is True
    assert written["user_id"] == "user-1"
    assert written["most_recent_message"] == "2026-09-09T10:00:00+00:00"


@pytest.mark.asyncio
async def test_a_conversation_the_reader_named_is_left_alone(
    fake_client, named_by_the_model
):
    named_by_the_model("Broken deployment pipeline")
    threads = fake_client(
        {
            "thread_id": "abc-123",
            "metadata": {
                "thread_metadata": {
                    "user_id": "user-1",
                    "assistant_id": "avatar-1",
                    "conversation_title": "Dinner plans",
                    "conversation_title_source": MANUAL_TITLE_SOURCE,
                }
            },
        },
        _conversation(),
    )

    title = await name_conversation_thread(
        "abc-123", langgraph_client_headers={"API-KEY": "k"}
    )

    assert title == ""
    assert threads.updates == []


@pytest.mark.asyncio
async def test_a_message_turn_does_not_rename_a_conversation_that_has_a_name(
    fake_client, named_by_the_model
):
    """``only_when_unnamed`` is what makes naming one call per conversation."""
    named_by_the_model("A different name")
    threads = fake_client(
        {
            "thread_id": "abc-123",
            "metadata": {
                "thread_metadata": {
                    "conversation_title": "Broken deployment pipeline",
                    "conversation_title_source": AUTOMATIC_TITLE_SOURCE,
                }
            },
        },
        _conversation(),
    )

    assert (
        await name_conversation_thread(
            "abc-123",
            langgraph_client_headers={"API-KEY": "k"},
            only_when_unnamed=True,
        )
        == ""
    )
    assert threads.updates == []

    # Leaving the conversation is allowed to rename it.
    assert (
        await name_conversation_thread(
            "abc-123",
            langgraph_client_headers={"API-KEY": "k"},
            only_when_unnamed=False,
        )
        == "A different name"
    )


@pytest.mark.asyncio
async def test_another_readers_thread_is_not_named(fake_client, named_by_the_model):
    named_by_the_model("Broken deployment pipeline")
    threads = fake_client(
        {
            "thread_id": "abc-123",
            "metadata": {
                "thread_metadata": {
                    "user_id": "someone-else",
                    "assistant_id": "avatar-1",
                }
            },
        },
        _conversation(),
    )

    title = await name_conversation_thread(
        "abc-123",
        langgraph_client_headers={"API-KEY": "k"},
        user_id="user-1",
        assistant_id="avatar-1",
    )

    assert title == ""
    assert threads.updates == []


@pytest.mark.asyncio
async def test_naming_can_be_switched_off(fake_client, named_by_the_model, monkeypatch):
    named_by_the_model("Broken deployment pipeline")
    threads = fake_client(
        {"thread_id": "abc-123", "metadata": {"thread_metadata": {}}}, _conversation()
    )
    monkeypatch.setenv("CONVERSATION_TITLE_ENABLED", "FALSE")

    assert (
        await name_conversation_thread(
            "abc-123", langgraph_client_headers={"API-KEY": "k"}
        )
        == ""
    )
    assert threads.updates == []


@pytest.mark.asyncio
async def test_a_message_turn_no_longer_writes_over_the_name_it_stored():
    """The metadata a ``/message`` turn writes is merged, not substituted.

    The platform stores ``thread_metadata`` as one value, so the object a turn
    writes replaces the object that was there. Before the merge, the name the
    namer stored after one turn was gone by the end of the next turn, and the
    ``shared`` flag with it.
    """
    from src.api.webapp import _thread_metadata_updates, _write_thread_metadata

    threads = _FakeThreadsClient(
        {
            "thread_id": "abc-123",
            "metadata": {
                "thread_metadata": {
                    "user_id": "user-1",
                    "assistant_id": "avatar-1",
                    "shared": True,
                    "conversation_title": "Broken deployment pipeline",
                    "conversation_title_source": AUTOMATIC_TITLE_SOURCE,
                }
            },
        }
    )

    await _write_thread_metadata(
        _FakeLangGraphClient(threads),
        "abc-123",
        _thread_metadata_updates(
            user_id="user-1",
            assistant_id="avatar-1",
            thread_id="abc-123",
            conversation_title_value=None,
        ),
    )

    written = threads.updates[-1]["thread_metadata"]
    assert written["conversation_title"] == "Broken deployment pipeline"
    assert written["conversation_title_source"] == AUTOMATIC_TITLE_SOURCE
    assert written["shared"] is True
    assert written["most_recent_message"]


def test_a_name_the_caller_sent_is_the_callers_own():
    """A caller that names its own conversations keeps its names.

    ``/message`` accepts a ``conversation_title``. A caller that sends one has
    named the conversation itself, so the name is stamped as typed rather than
    generated and the namer leaves the conversation alone from then on.
    """
    from src.api.webapp import _thread_metadata_updates

    updates = _thread_metadata_updates(
        user_id="user-1",
        assistant_id="avatar-1",
        thread_id="abc-123",
        conversation_title_value="Dinner plans",
    )
    assert updates["conversation_title"] == "Dinner plans"
    assert updates["conversation_title_source"] == MANUAL_TITLE_SOURCE

    # An empty name, and the thread's own identifier standing in for a name,
    # are both "the caller did not name this conversation".
    for not_a_name in ("", "   ", "abc-123"):
        updates = _thread_metadata_updates(
            user_id="user-1",
            assistant_id="avatar-1",
            thread_id="abc-123",
            conversation_title_value=not_a_name,
        )
        assert "conversation_title" not in updates
