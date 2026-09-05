"""Unit tests for the owner-only "what the avatar has learned" endpoints.

``GET /avatar_identity_facts`` lists everything an avatar has learned about its own
identity across the four store groups (``identity_memory`` → conversation, ``identity``
→ media, ``analysis``, ``memory``); ``DELETE`` forgets one row and ``PUT`` rewrites one
row through the same helper the conversational ``edit_identity_fact`` tool uses. All
three authorize the caller against the avatar's ``metadata.user_id`` through
``resolve_assistant_for_creator`` before touching the store.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context
from src.api import webapp as webapp_module

ASSISTANT_ID = "assistant-alpha"
CREATOR_ID = "6a5e59310832afadd626e583"
STRANGER_ID = "someone-else"


def _current_user(user_id):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


class _AssistantsAPI:
    def __init__(self, metadata):
        self._metadata = metadata

    async def get(self, assistant_id):
        return {"assistant_id": assistant_id, "metadata": self._metadata}


class _Store:
    """Stands in for ``app.state.store`` with prefix search, get, put, and delete."""

    def __init__(self, items=()):
        self.items = {(item.namespace, item.key): item for item in items}
        self.searched_namespaces = []
        self.deleted = []
        self.puts = []

    async def asearch(self, namespace, limit=None, query=None):
        self.searched_namespaces.append(namespace)
        return [
            item
            for (item_namespace, _key), item in self.items.items()
            if item_namespace[: len(namespace)] == tuple(namespace)
        ]

    async def aget(self, namespace, key):
        return self.items.get((tuple(namespace), key))

    async def aput(self, namespace, key, value):
        self.puts.append((tuple(namespace), key, value))
        self.items[(tuple(namespace), key)] = SimpleNamespace(
            namespace=tuple(namespace),
            key=key,
            value=value,
            updated_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        )

    async def adelete(self, namespace, key):
        self.deleted.append((tuple(namespace), key))
        self.items.pop((tuple(namespace), key), None)


def _item(namespace, key, document, updated_at=None):
    return SimpleNamespace(
        namespace=tuple(namespace),
        key=key,
        value={"document": document.to_json()},
        updated_at=updated_at or datetime(2026, 9, 1, tzinfo=UTC),
    )


def _conversation_fact(key="conv-1", fact="I was born in Ottawa."):
    context = "The owner said they were born in Ottawa."
    return _item(
        (CREATOR_ID, ASSISTANT_ID, "identity_memory"),
        key,
        Document(
            page_content=wrap_fact_with_context(fact, context),
            metadata={
                "user_id": CREATOR_ID,
                "assistant_id": ASSISTANT_ID,
                "document_id": "doc-conv-1",
                "fact": fact,
                "fact_context": context,
                "created_at": "2026-09-03T10:00:00+00:00",
            },
        ),
    )


def _media_fact(key="media-1", fact="I grew up on a farm."):
    return _item(
        (CREATOR_ID, ASSISTANT_ID, "identity", "uuid5-of-mom-m4a"),
        key,
        Document(
            page_content=wrap_fact_with_context(fact, "Mom described the farm."),
            metadata={
                "user_id": CREATOR_ID,
                "assistant_id": ASSISTANT_ID,
                "document_id": "doc-media-1",
                "filename": "Mom.m4a",
                "classified_situation": "biographical_facts",
                "synthetic": True,
                "created_at": "2026-09-02T10:00:00+00:00",
            },
        ),
    )


def _media_transcript(key="media-transcript"):
    return _item(
        (CREATOR_ID, ASSISTANT_ID, "identity", "uuid5-of-mom-m4a"),
        key,
        Document(
            page_content="So anyway we had chickens and a barn and...",
            metadata={
                "user_id": CREATOR_ID,
                "assistant_id": ASSISTANT_ID,
                "document_id": "doc-transcript",
                "filename": "Mom.m4a",
                "classified_situation": "dialogue",
            },
        ),
    )


def _analysis_trait(key="analysis-1"):
    return _item(
        (CREATOR_ID, ASSISTANT_ID, "analysis"),
        key,
        Document(
            page_content="Context: farm life. Statement: I value hard work.",
            metadata={
                "user_id": CREATOR_ID,
                "assistant_id": ASSISTANT_ID,
                "document_id": "doc-analysis-1",
                "feature": "values",
                "values": "I value hard work.",
                "supporting_reason": "Repeated praise of early mornings.",
                "filename": "Mom.m4a",
                "created_at": "2026-09-02T11:00:00+00:00",
            },
        ),
    )


def _episodic_memory(key="memory-1"):
    return _item(
        (CREATOR_ID, ASSISTANT_ID, "memory"),
        key,
        Document(
            page_content="We planned a trip to Lisbon.\n\nThe owner asked for travel tips.",
            metadata={
                "user_id": CREATOR_ID,
                "assistant_id": ASSISTANT_ID,
                "id": "doc-memory-1",
                "fact": "We planned a trip to Lisbon.",
                "fact_context": "The owner asked for travel tips.",
            },
        ),
        updated_at=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
    )


def _install(monkeypatch, metadata, items=()):
    assistants_api = _AssistantsAPI(metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    store = _Store(items)
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    return store


def _request(body):
    async def _json():
        return body

    return SimpleNamespace(json=_json)


@pytest.mark.asyncio
async def test_the_creator_sees_every_group_and_only_real_facts(monkeypatch):
    store = _install(
        monkeypatch,
        {"user_id": CREATOR_ID},
        items=(
            _conversation_fact(),
            _media_fact(),
            _media_transcript(),
            _analysis_trait(),
            _episodic_memory(),
        ),
    )

    response = await webapp_module.list_avatar_identity_facts(
        assistant_id=ASSISTANT_ID, current_user=_current_user(CREATOR_ID)
    )

    assert response["counts"] == {
        "conversation": 1,
        "media": 1,
        "analysis": 1,
        "memory": 1,
    }
    by_group = {row["learned_from"]: row for row in response["facts"]}
    assert set(by_group) == {"conversation", "media", "analysis", "memory"}
    # The transcript chunk sharing the identity prefix is a source, not a fact.
    assert all(row["fact_id"] != "doc-transcript" for row in response["facts"])
    # The media fact is unwrapped from its <FACT> span and labelled by its upload.
    assert by_group["media"]["fact"] == "I grew up on a farm."
    assert by_group["media"]["context"] == "Mom described the farm."
    assert by_group["media"]["source_label"] == "Mom.m4a"
    assert by_group["media"]["namespace"] == [
        CREATOR_ID,
        ASSISTANT_ID,
        "identity",
        "uuid5-of-mom-m4a",
    ]
    assert by_group["analysis"]["fact"] == "I value hard work."
    assert by_group["analysis"]["feature"] == "values"
    assert by_group["memory"]["fact"] == "We planned a trip to Lisbon."
    assert by_group["memory"]["fact_id"] == "doc-memory-1"
    # Newest first: the episodic memory (store updated_at today) leads.
    assert response["facts"][0]["learned_from"] == "memory"
    # Every group is read under the creator's id, never the caller's session.
    assert store.searched_namespaces == [
        (CREATOR_ID, ASSISTANT_ID, "identity_memory"),
        (CREATOR_ID, ASSISTANT_ID, "identity"),
        (CREATOR_ID, ASSISTANT_ID, "analysis"),
        (CREATOR_ID, ASSISTANT_ID, "memory"),
    ]


@pytest.mark.asyncio
async def test_a_stranger_may_not_list_what_an_avatar_learned(monkeypatch):
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(_conversation_fact(),))

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.list_avatar_identity_facts(
            assistant_id=ASSISTANT_ID, current_user=_current_user(STRANGER_ID)
        )

    assert rejection.value.status_code == 403
    assert store.searched_namespaces == []


@pytest.mark.asyncio
async def test_deleting_a_fact_removes_the_row_and_clears_the_cache(monkeypatch):
    fact = _conversation_fact()
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(fact,))
    invalidated = []
    monkeypatch.setattr(
        webapp_module, "invalidate_store_cache_for_assistant", invalidated.append
    )

    response = await webapp_module.delete_avatar_identity_fact(
        request=_request({"namespace": list(fact.namespace), "key": fact.key}),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
    )

    assert response.status_code == 204
    assert store.deleted == [(fact.namespace, fact.key)]
    assert invalidated == [ASSISTANT_ID]


@pytest.mark.asyncio
async def test_a_namespace_outside_the_avatar_is_refused(monkeypatch):
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(_conversation_fact(),))

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.delete_avatar_identity_fact(
            request=_request(
                {"namespace": [STRANGER_ID, "other-avatar", "identity_memory"], "key": "x"}
            ),
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
        )
    assert rejection.value.status_code == 403
    assert store.deleted == []

    with pytest.raises(webapp_module.HTTPException) as missing:
        await webapp_module.delete_avatar_identity_fact(
            request=_request(
                {"namespace": [CREATOR_ID, ASSISTANT_ID, "identity_memory"], "key": "nope"}
            ),
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
        )
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_editing_a_wrapped_fact_keeps_the_wrapper_and_records_the_old_text(
    monkeypatch,
):
    fact = _media_fact()
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(fact,))
    monkeypatch.setattr(
        webapp_module, "invalidate_store_cache_for_assistant", lambda _id: None
    )

    response = await webapp_module.update_avatar_identity_fact(
        request=_request(
            {
                "namespace": list(fact.namespace),
                "key": fact.key,
                "fact": "I grew up on a dairy farm.",
            }
        ),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
    )

    assert response["fact"] == "I grew up on a dairy farm."
    assert response["context"] == "Mom described the farm."
    assert response["corrected_from"] == "I grew up on a farm."
    stored = store.items[(fact.namespace, fact.key)].value["document"]["kwargs"]
    assert stored["page_content"].startswith("<FACT_CONTEXT_AND_FACT>")
    assert "<FACT>I grew up on a dairy farm.</FACT>" in stored["page_content"]
    assert stored["metadata"]["correction_origin"] == "user"
    assert stored["metadata"]["filename"] == "Mom.m4a"


@pytest.mark.asyncio
async def test_editing_an_episodic_memory_keeps_the_plain_format(monkeypatch):
    memory = _episodic_memory()
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(memory,))
    monkeypatch.setattr(
        webapp_module, "invalidate_store_cache_for_assistant", lambda _id: None
    )

    response = await webapp_module.update_avatar_identity_fact(
        request=_request(
            {
                "namespace": list(memory.namespace),
                "key": memory.key,
                "fact": "We planned a trip to Porto.",
                "context": "The owner asked for Portugal tips.",
            }
        ),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
    )

    assert response["learned_from"] == "memory"
    stored = store.items[(memory.namespace, memory.key)].value["document"]["kwargs"]
    assert stored["page_content"] == (
        "We planned a trip to Porto.\n\nThe owner asked for Portugal tips."
    )
    assert stored["metadata"]["corrected_from"] == "We planned a trip to Lisbon."


@pytest.mark.asyncio
async def test_editing_an_analysis_trait_updates_the_feature_statement(monkeypatch):
    trait = _analysis_trait()
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(trait,))
    monkeypatch.setattr(
        webapp_module, "invalidate_store_cache_for_assistant", lambda _id: None
    )

    response = await webapp_module.update_avatar_identity_fact(
        request=_request(
            {
                "namespace": list(trait.namespace),
                "key": trait.key,
                "fact": "I value honest work.",
            }
        ),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
    )

    assert response["fact"] == "I value honest work."
    assert response["feature"] == "values"
    stored = store.items[(trait.namespace, trait.key)].value["document"]["kwargs"]
    assert stored["metadata"]["values"] == "I value honest work."


@pytest.mark.asyncio
async def test_a_stranger_may_not_edit_or_delete(monkeypatch):
    fact = _conversation_fact()
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, items=(fact,))
    body = {"namespace": list(fact.namespace), "key": fact.key, "fact": "x"}

    for endpoint in (
        webapp_module.update_avatar_identity_fact,
        webapp_module.delete_avatar_identity_fact,
    ):
        with pytest.raises(webapp_module.HTTPException) as rejection:
            await endpoint(
                request=_request(body),
                assistant_id=ASSISTANT_ID,
                current_user=_current_user(STRANGER_ID),
            )
        assert rejection.value.status_code == 403
    assert store.puts == []
    assert store.deleted == []


def test_the_fact_routes_require_assistant_id():
    for method in ("GET", "DELETE", "PUT"):
        for route in webapp_module.app.routes:
            if getattr(route, "path", None) == "/avatar_identity_facts" and method in getattr(
                route, "methods", ()
            ):
                query_parameters = {
                    parameter.name: parameter.field_info.is_required()
                    for parameter in route.dependant.query_params
                }
                assert query_parameters.get("assistant_id") is True, method
                break
        else:
            raise AssertionError(f"route not registered: {method} /avatar_identity_facts")
