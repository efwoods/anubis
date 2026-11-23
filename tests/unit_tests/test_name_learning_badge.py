"""A name the user states is shown as a learned fact on the avatar's reply.

The name is stored while the system prompt is built, with no tool call, so the
store change is announced from ``load_consciousness`` and handed to ``think``
through the ``name_facts_learned_in_turn`` state key.
"""

from __future__ import annotations

import hashlib
import math
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.learning.fact_learned as fact_learned
import src.anubis.utils.nodes as nodes
from src.anubis.utils.learning.address_correction import (
    ADDRESS_PREFERENCE_CATEGORY,
    apply_names_stated_in_message,
    apply_names_stated_in_message_with_reports,
    learned_fact_entries_for_name_reports,
)
from src.anubis.utils.learning.fact_learned import (
    TurnLearnedFactsCollector,
    attach_learned_facts_metadata,
    seed_turn_learned_facts,
)
from src.anubis.utils.learning.feedback import (
    list_namespace_items,
    store_user_preference,
)
from src.anubis.utils.learning.namespaces import preference_namespace

pytestmark = pytest.mark.asyncio

USER = "user-1"
AVATAR = "avatar-1"
DIMENSIONS = 16


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def _make_store() -> InMemoryStore:
    return InMemoryStore(
        index={
            "dims": DIMENSIONS,
            "embed": _embed,
            "fields": ["document.kwargs.page_content"],
        }
    )


async def _entries(store, message: str) -> list[dict[str, str]]:
    (
        rejected_names,
        stated_name,
        reports,
    ) = await apply_names_stated_in_message_with_reports(store, USER, AVATAR, message)
    return learned_fact_entries_for_name_reports(rejected_names, stated_name, reports)


async def _address_facts(store) -> list[str]:
    return sorted(
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(
            store, preference_namespace(USER, AVATAR)
        )
        if item.value["document"]["kwargs"]["metadata"].get("category")
        == ADDRESS_PREFERENCE_CATEGORY
    )


async def test_a_first_name_is_learned_a_changed_name_is_updated_a_repeated_name_is_known():
    store = _make_store()
    assert await _entries(store, "My name is Evan.") == [
        {"fact": "Address the user as Evan.", "kind": "preference", "status": "learned"}
    ]
    assert await _entries(store, "My name is David.") == [
        {
            "fact": "Address the user as David.",
            "kind": "preference",
            "status": "updated",
            "previous_fact": "Address the user as Evan.",
        }
    ]
    assert await _entries(store, "My name is David.") == [
        {"fact": "Address the user as David.", "kind": "preference", "status": "known"}
    ]


async def test_a_rejected_name_is_reported_as_removed_and_never_repeated():
    store = _make_store()
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is David.")
    entries = await _entries(store, "My name is Evan, do not call me David.")
    assert entries == [
        {
            "fact": "The user rejected a name.",
            "kind": "preference",
            "status": "removed",
        },
        {
            "fact": "Address the user as Evan.",
            "kind": "preference",
            "status": "learned",
        },
    ]
    assert "David" not in str(entries)


async def test_a_message_without_a_name_yields_no_entry():
    assert await _entries(_make_store(), "I turned off the light.") == []


async def test_an_older_preference_with_a_comma_before_the_name_is_replaced():
    """The production row that survived beside "Address the user as David."."""
    store = _make_store()
    for preference in (
        "You want to be addressed by your first name, Evan.",
        "Address me conversationally and directly when I say hello.",
        "Address me directly by name when responding.",
    ):
        await store_user_preference(
            store,
            USER,
            AVATAR,
            preference=preference,
            preference_context=None,
            category=ADDRESS_PREFERENCE_CATEGORY,
            source="inferred",
        )
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is Evan.")
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is David.")
    assert await _address_facts(store) == [
        "Address me conversationally and directly when I say hello.",
        "Address me directly by name when responding.",
        "Address the user as David.",
    ]


async def test_an_unread_phrasing_that_mentions_the_replaced_name_is_deleted():
    store = _make_store()
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is Evan.")
    await store_user_preference(
        store,
        USER,
        AVATAR,
        preference="Evan is the form of address the user likes best.",
        preference_context=None,
        category=ADDRESS_PREFERENCE_CATEGORY,
        source="inferred",
    )
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is David.")
    assert await _address_facts(store) == ["Address the user as David."]


async def test_the_name_is_announced_once_per_human_message(monkeypatch):
    announced: list[dict] = []
    monkeypatch.setattr(
        fact_learned,
        "announce_fact_learned",
        lambda fact, **keywords: announced.append({"fact": fact, **keywords}),
    )
    store = _make_store()
    latest_message = HumanMessage(content="My name is Evan.", id="human-1")
    state: dict = {"messages": [latest_message]}

    await nodes._apply_names_stated_in_latest_message(
        store, USER, AVATAR, latest_message
    )
    nodes._announce_name_facts_once(state, latest_message)
    state_value = nodes._name_facts_state_value_of_latest_build.get()
    assert state_value == {
        "message_id": "human-1",
        "facts": [
            {
                "fact": "Address the user as Evan.",
                "kind": "preference",
                "status": "learned",
            }
        ],
    }
    assert [entry["fact"] for entry in announced] == ["Address the user as Evan."]

    # The avatar's loop rebuilds the prompt with the same human message.
    state["name_facts_learned_in_turn"] = state_value
    await nodes._apply_names_stated_in_latest_message(
        store, USER, AVATAR, latest_message
    )
    nodes._announce_name_facts_once(state, latest_message)
    assert len(announced) == 1
    assert nodes._name_facts_state_value_of_latest_build.get() is None


async def test_seeded_name_facts_reach_the_reply_metadata():
    TurnLearnedFactsCollector.begin_turn()
    try:
        seed_turn_learned_facts(
            [
                {
                    "fact": "Address the user as David.",
                    "kind": "preference",
                    "status": "updated",
                    "previous_fact": "Address the user as Evan.",
                }
            ]
        )
        reply = AIMessage(content="Got it, David.")
        attach_learned_facts_metadata(reply, [])
    finally:
        TurnLearnedFactsCollector.end_turn()
    assert reply.response_metadata["learned_facts"] == [
        {
            "fact": "Address the user as David.",
            "kind": "preference",
            "status": "updated",
            "previous_fact": "Address the user as Evan.",
        }
    ]
