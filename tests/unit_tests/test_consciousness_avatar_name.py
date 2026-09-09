"""Unit tests for how ``load_consciousness`` resolves the avatar's own name.

``=== YOUR NAME ===`` is the only place the system prompt states who the avatar
is outright. When it renders empty the avatar has no anchor by which to notice
that a stored identity fact naming it in the third person ("I have worked with
<name> for years") contradicts who it is, and it will introduce itself as its
own colleague.

Two defects emptied that section:

* the fallback name lookup was gated on ``assistant_name is None``, so an avatar
  row carrying an empty-string name skipped the lookup and rendered blank; and
* the fallback searched the USER's identity namespace
  ``(assistant_id, user_id, "identity")`` rather than the avatar's own
  ``(creator_id, assistant_id, "identity")``, so it could not find the avatar's
  name at all whenever the speaker was not also the avatar's creator.

The store is a fake recording every namespace searched, so these exercise the
deterministic lookup plumbing without a live store or live inference.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

import src.anubis.utils.nodes as nodes

CREATOR_ID = "creator-1"
ASSISTANT_ID = "assistant-1"
VISITOR_ID = "visitor-1"

ASSISTANT_IDENTITY_NAMESPACE = (CREATOR_ID, ASSISTANT_ID, "identity")
VISITOR_IDENTITY_NAMESPACE = (ASSISTANT_ID, VISITOR_ID, "identity")


class _RecordingStore:
    """Fake store that records every searched namespace and serves canned items.

    ``name_items_by_namespace`` is served ONLY to the name lookup, which is the
    one search that asks "WHAT IS YOUR NAME?". Every other search — including
    the full identity load over the same namespace — gets an empty list, so a
    name item never has to satisfy the Document shape the identity load builds.
    """

    _NAME_QUERY_MARKER = "WHAT IS YOUR NAME?"

    def __init__(self, name_items_by_namespace: dict | None = None):
        self.name_items_by_namespace = name_items_by_namespace or {}
        self.searched_namespaces: list[tuple] = []

    async def asearch(self, namespace, query=None, limit=None):
        self.searched_namespaces.append(tuple(namespace))
        if self._NAME_QUERY_MARKER in str(query or ""):
            return self.name_items_by_namespace.get(tuple(namespace), [])
        return []

    async def aget(self, namespace, key):
        return None


def _name_item(fact: str):
    """A store item shaped the way the name lookup reads it."""
    return SimpleNamespace(
        value={"document": {"kwargs": {"metadata": {"fact": fact}}}},
        score=0.9,
    )


async def _build_prompt(store, *, assistant_name, user_id=VISITOR_ID):
    """Drive the real consciousness builder and return the rendered system prompt."""
    assistant_ctx = {
        "name": assistant_name,
        "metadata": {"user_id": CREATOR_ID},
    }
    state = {
        "messages": [HumanMessage(content="Will you tell me about yourself?")],
        "user_state": {"user_id": user_id},
        "assistant_state": {"assistant_id": ASSISTANT_ID},
    }
    config = {
        "configurable": {
            "user_id": user_id,
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


def _rendered_name_section(system_prompt: str) -> str:
    """The ROLE's ``=== YOUR NAME ===`` value.

    The marker also appears in the prompt's worked example, and the instruction
    block is repeated after the ROLE, so neither the first nor the last
    occurrence is the right one — anchor on ``<ROLE>`` and read forward.
    """
    role_start = system_prompt.find("<ROLE>")
    assert role_start != -1, "the rendered prompt carries no ROLE block"
    marker = "=== YOUR NAME ==="
    start = system_prompt.index(marker, role_start) + len(marker)
    end = system_prompt.index("=== YOUR IDENTITY ===", start)
    return system_prompt[start:end].strip()


@pytest.mark.asyncio
async def test_blank_avatar_name_still_runs_the_fallback_lookup():
    """An empty-string name is as absent as ``None`` and must trigger the lookup.

    The avatar's identity namespace is searched twice when the fallback runs:
    once for the name, once for the full identity load.
    """
    store = _RecordingStore()

    await _build_prompt(store, assistant_name="")

    identity_searches = store.searched_namespaces.count(ASSISTANT_IDENTITY_NAMESPACE)
    assert identity_searches == 2, store.searched_namespaces


@pytest.mark.asyncio
async def test_a_supplied_avatar_name_skips_the_fallback_lookup():
    """When the context already carries a name, no extra lookup is spent."""
    store = _RecordingStore()

    await _build_prompt(store, assistant_name="Grant Imahara")

    identity_searches = store.searched_namespaces.count(ASSISTANT_IDENTITY_NAMESPACE)
    assert identity_searches == 1, store.searched_namespaces


@pytest.mark.asyncio
async def test_fallback_searches_the_avatar_namespace_not_the_visitor_namespace():
    """The avatar's name lives under the creator's identity namespace.

    The visitor here is not the creator, so the previously-searched
    ``(assistant_id, user_id, "identity")`` namespace holds the VISITOR's facts
    and could never contain the avatar's name.
    """
    store = _RecordingStore()

    await _build_prompt(store, assistant_name=None)

    assert ASSISTANT_IDENTITY_NAMESPACE in store.searched_namespaces
    # The visitor namespace is still searched for the visitor's own name and
    # identity — but it must not be where the avatar's name is looked for.
    assert store.searched_namespaces.index(ASSISTANT_IDENTITY_NAMESPACE) < (
        store.searched_namespaces.index(VISITOR_IDENTITY_NAMESPACE)
    )


@pytest.mark.asyncio
async def test_blank_name_is_recovered_from_the_avatar_identity_namespace():
    """The end the two fixes serve: the ROLE states who the avatar is."""
    store = _RecordingStore(
        {ASSISTANT_IDENTITY_NAMESPACE: [_name_item("I'm Grant Imahara.")]}
    )

    system_prompt = await _build_prompt(store, assistant_name="")

    assert "Grant Imahara" in _rendered_name_section(system_prompt)


@pytest.mark.asyncio
async def test_whitespace_only_name_is_treated_as_blank():
    """A name of spaces must not satisfy the "already have a name" gate."""
    store = _RecordingStore(
        {ASSISTANT_IDENTITY_NAMESPACE: [_name_item("I'm Grant Imahara.")]}
    )

    system_prompt = await _build_prompt(store, assistant_name="   ")

    assert "Grant Imahara" in _rendered_name_section(system_prompt)
