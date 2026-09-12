"""Opening suggestion harvests retrieve identity and override spoken-reply rules.

Composer chips for a new conversation have to come from who the avatar is
(a recruiter, a restaurant) rather than generic greetings. The harvest text
itself is about JSON lists, so consciousness must search with the avatar's
name and description, and must append the harvest instruction so the identity
prompt's "write a spoken paragraph" rule does not win.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

import src.anubis.utils.nodes as nodes
from src.anubis.utils.client_harvest_turns import (
    CONVERSATION_SUGGESTIONS_MARKER,
    identity_retrieval_query_for_suggestion_harvest,
)

CLAIRE_PLACE_URL = "https://clairesplacefoundation.org/"
GRANT_STEAM_URL = "https://www.grantimaharafoundation.org/"
CLAIRE_DESCRIPTION = (
    "Founder of Claire's Place Foundation, supporting families living "
    f"with cystic fibrosis. {CLAIRE_PLACE_URL}"
)
GRANT_DESCRIPTION = (
    "Engineer and founder of Grant Imahara's STEAM Foundation. "
    f"{GRANT_STEAM_URL}"
)

CREATOR_ID = "creator-1"
ASSISTANT_ID = "assistant-1"
VISITOR_ID = "visitor-1"

OPENING_HARVEST = (
    f"{CONVERSATION_SUGGESTIONS_MARKER} Suggest three short opening messages "
    "the person might send to start this conversation. Reply with a JSON "
    "array of three strings and nothing else."
)


class _RecordingStore:
    """Fake store that records every searched namespace and query."""

    def __init__(self) -> None:
        self.searched_queries: list[str] = []

    async def asearch(self, namespace, query=None, limit=None):
        self.searched_queries.append(str(query or ""))
        return []

    async def aget(self, namespace, key):
        return None


async def _build_prompt(
    store,
    *,
    message_content: str,
    assistant_name: str,
    assistant_description: str,
) -> str:
    """Drive the real consciousness builder and return the rendered system prompt."""
    assistant_ctx = {
        "name": assistant_name,
        "description": assistant_description,
        "metadata": {"user_id": CREATOR_ID},
    }
    state = {
        "messages": [HumanMessage(content=message_content)],
        "user_state": {"user_id": VISITOR_ID},
        "assistant_state": {"assistant_id": ASSISTANT_ID},
    }
    config = {
        "configurable": {
            "user_id": VISITOR_ID,
            "assistant_id": ASSISTANT_ID,
            "assistant_ctx": assistant_ctx,
            "user_ctx": {},
            "thread_id": "thread-1",
        }
    }
    runtime = SimpleNamespace(
        store=store,
        context=SimpleNamespace(assistant_ctx=assistant_ctx, user_ctx={}),
    )
    update = await nodes._build_consciousness_system_message_update(
        state, config, runtime
    )
    return update["system_message"][0].content


@pytest.mark.asyncio
async def test_opening_harvest_searches_identity_with_the_avatar_role() -> None:
    store = _RecordingStore()

    await _build_prompt(
        store,
        message_content=OPENING_HARVEST,
        assistant_name="National Guard",
        assistant_description="U.S. Army National Guard recruiter.",
    )

    expected_query = identity_retrieval_query_for_suggestion_harvest(
        assistant_name="National Guard",
        assistant_description="U.S. Army National Guard recruiter.",
    )
    assert expected_query in store.searched_queries
    assert not any(
        CONVERSATION_SUGGESTIONS_MARKER in query for query in store.searched_queries
    )


@pytest.mark.asyncio
async def test_opening_harvest_appends_the_chip_instruction() -> None:
    store = _RecordingStore()

    system_prompt = await _build_prompt(
        store,
        message_content=OPENING_HARVEST,
        assistant_name="Mellow Mushroom",
        assistant_description="Pizza restaurant that takes dine-in and takeout orders.",
    )

    assert "<CONVERSATION_SUGGESTION_HARVEST>" in system_prompt
    assert "opening messages" in system_prompt
    assert "placing" in system_prompt


@pytest.mark.asyncio
async def test_an_ordinary_turn_does_not_append_the_chip_instruction() -> None:
    store = _RecordingStore()

    system_prompt = await _build_prompt(
        store,
        message_content="Hey, how are you?",
        assistant_name="National Guard",
        assistant_description="U.S. Army National Guard recruiter.",
    )

    assert "<CONVERSATION_SUGGESTION_HARVEST>" not in system_prompt
    assert any(
        "Hey, how are you?" in query for query in store.searched_queries
    )


@pytest.mark.asyncio
async def test_claire_wineland_opening_harvest_shares_claires_place_foundation() -> None:
    store = _RecordingStore()

    system_prompt = await _build_prompt(
        store,
        message_content=OPENING_HARVEST,
        assistant_name="Claire Wineland",
        assistant_description=CLAIRE_DESCRIPTION,
    )

    assert "=== YOUR ORGANIZATION LINKS ===" in system_prompt
    assert CLAIRE_PLACE_URL in system_prompt
    assert "share the matching URL" in system_prompt
    assert "website link" in system_prompt
    expected_query = identity_retrieval_query_for_suggestion_harvest(
        assistant_name="Claire Wineland",
        assistant_description=CLAIRE_DESCRIPTION,
    )
    assert expected_query in store.searched_queries


@pytest.mark.asyncio
async def test_grant_imahara_opening_harvest_shares_the_steam_foundation() -> None:
    store = _RecordingStore()

    system_prompt = await _build_prompt(
        store,
        message_content=OPENING_HARVEST,
        assistant_name="Grant Imahara",
        assistant_description=GRANT_DESCRIPTION,
    )

    assert GRANT_STEAM_URL in system_prompt
    assert "STEAM Foundation" in system_prompt
    assert "website link" in system_prompt
    expected_query = identity_retrieval_query_for_suggestion_harvest(
        assistant_name="Grant Imahara",
        assistant_description=GRANT_DESCRIPTION,
    )
    assert expected_query in store.searched_queries
