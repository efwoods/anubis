"""The standard set of conversation starters is generated once per avatar and stored.

A new conversation used to cost a hidden avatar turn per browser to produce
three opening chips that never change between conversations. These tests cover
the pure half — normalization, the stored record, reading the record off an
assistant, the AVATAR block the prompt is given — and the storage side: the
public listing lifts the record out of the stripped metadata, the research job
regenerates the set when the research finishes, and the refresh helper writes
the record through the assistants client and never raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.anubis.utils.conversation_starters as conversation_starters_module
from src.anubis.utils.conversation_starters import (
    CONVERSATION_STARTERS_METADATA_KEY,
    SOURCE_DEEP_RESEARCH,
    SOURCE_IDENTITY_ONLY,
    ConversationStarters,
    build_conversation_starters_human_text,
    build_conversation_starters_record,
    conversation_starters_enabled,
    conversation_starters_of,
    conversation_starters_record_of,
    generate_conversation_starters,
    normalize_conversation_starters,
)
from src.api import webapp as webapp_module

CREATOR_ID = "creator-1"
ASSISTANT_ID = "assistant-1"
CLAIRE_PLACE_URL = "https://clairesplacefoundation.org/"
CLAIRE_DESCRIPTION = (
    "Founder of Claire's Place Foundation, supporting families living "
    f"with cystic fibrosis. {CLAIRE_PLACE_URL}"
)
RECRUITER_STARTERS = [
    "What are the requirements to join the National Guard?",
    "Which roles are you recruiting for right now?",
    "Walk me through the first step to enlist.",
]


class _FactStore:
    """Fake store whose identity namespace holds a few facts."""

    def __init__(self, facts: list[str]) -> None:
        self.facts = facts
        self.searched: list[tuple] = []

    async def asearch(self, namespace, query=None, limit=None):
        self.searched.append((namespace, limit))
        return [
            SimpleNamespace(
                key=f"fact-{index}",
                value={"document": {"kwargs": {"page_content": fact}}},
            )
            for index, fact in enumerate(self.facts)
        ]


@pytest.fixture
def answered_by_the_model(monkeypatch):
    """Install a fake structured call and return the human texts it was given."""

    def _install(starters: list[str]) -> list[str]:
        human_texts: list[str] = []

        async def _invoke(response_format, system_prompt, human_text):
            human_texts.append(human_text)
            return ConversationStarters(starters=starters)

        monkeypatch.setattr(conversation_starters_module, "invoke_structured", _invoke)
        return human_texts

    return _install


# ── normalization and the record ────────────────────────────────────────────


def test_normalization_strips_numbering_and_quotes_and_caps_at_three() -> None:
    cleaned = normalize_conversation_starters(
        [
            '1. "Can I see the menu?"',
            "- What do you recommend tonight?",
            "• I'd like to place a takeout order.",
            "What are your hours?",
        ]
    )
    assert cleaned == [
        "Can I see the menu?",
        "What do you recommend tonight?",
        "I'd like to place a takeout order.",
    ]


def test_normalization_drops_generic_greetings_duplicates_and_blanks() -> None:
    cleaned = normalize_conversation_starters(
        [
            "Hey, how are you?",
            "Tell me about yourself",
            "",
            "Can we pray together?",
            "can we pray together?",
            "x" * 200,
            42,
        ]
    )
    assert cleaned == ["Can we pray together?"]


def test_the_record_carries_the_starters_the_stamp_and_the_source() -> None:
    record = build_conversation_starters_record(
        RECRUITER_STARTERS, source=SOURCE_DEEP_RESEARCH, identity_fact_count=12
    )
    assert record["starters"] == RECRUITER_STARTERS
    assert record["source"] == SOURCE_DEEP_RESEARCH
    assert record["identity_fact_count"] == 12
    assert record["generated_at"].endswith("+00:00")


def test_the_record_is_read_from_metadata_or_a_lifted_field() -> None:
    record = build_conversation_starters_record(
        RECRUITER_STARTERS, source=SOURCE_IDENTITY_ONLY
    )
    from_metadata = {"metadata": {CONVERSATION_STARTERS_METADATA_KEY: record}}
    lifted = {CONVERSATION_STARTERS_METADATA_KEY: record}

    assert conversation_starters_of(from_metadata) == RECRUITER_STARTERS
    assert conversation_starters_of(lifted) == RECRUITER_STARTERS
    assert conversation_starters_record_of(from_metadata)["source"] == (
        SOURCE_IDENTITY_ONLY
    )


def test_a_missing_or_malformed_record_reads_as_nothing() -> None:
    assert conversation_starters_record_of(None) is None
    assert conversation_starters_record_of({"metadata": {}}) is None
    assert (
        conversation_starters_record_of(
            {"metadata": {CONVERSATION_STARTERS_METADATA_KEY: {"starters": ["Hi"]}}}
        )
        is None
    )
    assert (
        conversation_starters_of(
            {"metadata": {CONVERSATION_STARTERS_METADATA_KEY: "x"}}
        )
        == []
    )


def test_the_switch_reads_the_context_and_defaults_on() -> None:
    assert conversation_starters_enabled(SimpleNamespace()) is True
    assert (
        conversation_starters_enabled(
            SimpleNamespace(conversation_starters_enabled="false")
        )
        is False
    )
    assert (
        conversation_starters_enabled(
            SimpleNamespace(conversation_starters_enabled="0")
        )
        is False
    )
    assert (
        conversation_starters_enabled(
            SimpleNamespace(conversation_starters_enabled=None)
        )
        is True
    )


# ── the prompt input ────────────────────────────────────────────────────────


def test_the_avatar_block_names_identity_links_and_facts() -> None:
    text = build_conversation_starters_human_text(
        name="Claire Wineland",
        description=CLAIRE_DESCRIPTION,
        organization_links=[CLAIRE_PLACE_URL],
        identity_facts=["I founded Claire's Place Foundation when I was 13."],
    )
    assert "Name: Claire Wineland" in text
    assert CLAIRE_DESCRIPTION in text
    assert f"- {CLAIRE_PLACE_URL}" in text
    assert "- I founded Claire's Place Foundation when I was 13." in text


def test_the_avatar_block_says_when_no_facts_are_held_yet() -> None:
    text = build_conversation_starters_human_text(
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
        organization_links=[],
        identity_facts=[],
    )
    assert "none yet" in text
    assert "Organization links" not in text


@pytest.mark.asyncio
async def test_generation_grounds_the_prompt_in_held_facts_and_the_website(
    answered_by_the_model,
) -> None:
    human_texts = answered_by_the_model(
        [
            "Can you share the Claire's Place Foundation website?",
            "How does the foundation support families with cystic fibrosis?",
            "What made you start the foundation so young?",
        ]
    )
    store = _FactStore(
        [
            "I founded Claire's Place Foundation when I was 13.",
            "I have cystic fibrosis.",
        ]
    )

    starters, fact_count = await generate_conversation_starters(
        store,
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        name="Claire Wineland",
        description=CLAIRE_DESCRIPTION,
    )

    assert fact_count == 2
    assert len(starters) == 3
    assert starters[0].startswith("Can you share")
    (human_text,) = human_texts
    assert CLAIRE_PLACE_URL in human_text
    assert "I founded Claire's Place Foundation when I was 13." in human_text
    # The identity namespace was read with a bounded limit.
    assert store.searched == [
        (
            (CREATOR_ID, ASSISTANT_ID, "identity"),
            conversation_starters_module.CONVERSATION_STARTER_IDENTITY_FACT_LIMIT,
        )
    ]


@pytest.mark.asyncio
async def test_generation_still_works_when_the_store_fails(
    answered_by_the_model,
) -> None:
    answered_by_the_model(RECRUITER_STARTERS)

    class _BrokenStore:
        async def asearch(self, *args, **kwargs):
            raise RuntimeError("store down")

    starters, fact_count = await generate_conversation_starters(
        _BrokenStore(),
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
    )
    assert starters == RECRUITER_STARTERS
    assert fact_count == 0


@pytest.mark.asyncio
async def test_generation_drops_generic_lines_the_model_slipped_in(
    answered_by_the_model,
) -> None:
    answered_by_the_model(
        ["Hey, how are you?", "I want to enlist.", "What roles are open?"]
    )
    starters, _ = await generate_conversation_starters(
        None,
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
    )
    assert starters == ["I want to enlist.", "What roles are open?"]


# ── the storage side ────────────────────────────────────────────────────────


def _public_avatar(record) -> dict:
    return {
        "assistant_id": ASSISTANT_ID,
        "name": "National Guard",
        "description": "U.S. Army National Guard recruiter.",
        "metadata": {
            "user_id": CREATOR_ID,
            "is_public": True,
            CONVERSATION_STARTERS_METADATA_KEY: record,
        },
    }


def test_the_public_listing_lifts_the_starters_out_of_the_stripped_metadata() -> None:
    record = build_conversation_starters_record(
        RECRUITER_STARTERS, source=SOURCE_DEEP_RESEARCH
    )
    stripped = webapp_module._assistant_without_metadata(_public_avatar(record))

    assert "metadata" not in stripped
    assert (
        stripped[CONVERSATION_STARTERS_METADATA_KEY]["starters"] == RECRUITER_STARTERS
    )
    assert conversation_starters_of(stripped) == RECRUITER_STARTERS


def test_the_public_listing_lifts_nothing_when_no_set_is_stored() -> None:
    avatar = _public_avatar(None)
    stripped = webapp_module._assistant_without_metadata(avatar)
    assert CONVERSATION_STARTERS_METADATA_KEY not in stripped


class _RecordingAssistants:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    async def update(self, **kwargs):
        self.updates.append(kwargs)
        return kwargs


@pytest.mark.asyncio
async def test_the_refresh_helper_writes_the_record_through_the_assistants_client(
    monkeypatch, answered_by_the_model
) -> None:
    answered_by_the_model(RECRUITER_STARTERS)
    assistants = _RecordingAssistants()
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda headers=None: SimpleNamespace(assistants=assistants),
    )
    app_state = SimpleNamespace(
        context=SimpleNamespace(conversation_starters_enabled="true"), store=None
    )

    record = await webapp_module.refresh_avatar_conversation_starters(
        app_state,
        {"API_KEY": "token", "identities": [{"user_id": CREATOR_ID}]},
        assistant_id=ASSISTANT_ID,
        creator_id=CREATOR_ID,
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
        source=SOURCE_DEEP_RESEARCH,
    )

    assert record["starters"] == RECRUITER_STARTERS
    assert record["source"] == SOURCE_DEEP_RESEARCH
    (update,) = assistants.updates
    assert update["assistant_id"] == ASSISTANT_ID
    assert update["metadata"] == {CONVERSATION_STARTERS_METADATA_KEY: record}


@pytest.mark.asyncio
async def test_the_refresh_helper_is_off_when_the_switch_is_off(
    monkeypatch, answered_by_the_model
) -> None:
    human_texts = answered_by_the_model(RECRUITER_STARTERS)
    assistants = _RecordingAssistants()
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda headers=None: SimpleNamespace(assistants=assistants),
    )
    app_state = SimpleNamespace(
        context=SimpleNamespace(conversation_starters_enabled="false"), store=None
    )

    record = await webapp_module.refresh_avatar_conversation_starters(
        app_state,
        {"API_KEY": "token", "identities": [{"user_id": CREATOR_ID}]},
        assistant_id=ASSISTANT_ID,
        creator_id=CREATOR_ID,
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
        source=SOURCE_DEEP_RESEARCH,
    )

    assert record is None
    assert human_texts == []
    assert assistants.updates == []


@pytest.mark.asyncio
async def test_the_refresh_helper_keeps_the_previous_set_when_the_model_is_unusable(
    monkeypatch, answered_by_the_model
) -> None:
    answered_by_the_model(["Hey, how are you?", "Hello", "Tell me more"])
    assistants = _RecordingAssistants()
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda headers=None: SimpleNamespace(assistants=assistants),
    )
    app_state = SimpleNamespace(context=SimpleNamespace(), store=None)

    record = await webapp_module.refresh_avatar_conversation_starters(
        app_state,
        {"API_KEY": "token", "identities": [{"user_id": CREATOR_ID}]},
        assistant_id=ASSISTANT_ID,
        creator_id=CREATOR_ID,
        name="National Guard",
        description="U.S. Army National Guard recruiter.",
        source=SOURCE_DEEP_RESEARCH,
    )

    assert record is None
    assert assistants.updates == []


@pytest.mark.asyncio
async def test_the_refresh_helper_never_raises(monkeypatch) -> None:
    async def _explode(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(conversation_starters_module, "invoke_structured", _explode)
    app_state = SimpleNamespace(context=SimpleNamespace(), store=None)

    record = await webapp_module.refresh_avatar_conversation_starters(
        app_state,
        {"API_KEY": "token", "identities": [{"user_id": CREATOR_ID}]},
        assistant_id=ASSISTANT_ID,
        creator_id=CREATOR_ID,
        name="National Guard",
        description=None,
        source=SOURCE_IDENTITY_ONLY,
    )
    assert record is None
