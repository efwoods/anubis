"""Unit tests for who may list and delete an avatar's source documents.

``/list_avatar_documents`` and ``/delete_avatar_document`` used to take no avatar
parameter at all: both derived their entire scope from a per-account "selected avatar"
persisted in Auth0 by ``POST /select_avatar``. That selection was a per-account
singleton shared across every device and tab, its by-id branch never verified that the
caller created the avatar, and a freshly signed-up account had no selection at all — so
both endpoints answered 400 until a selection call landed.

Both endpoints now take ``assistant_id`` explicitly and authorize the caller against the
avatar's ``metadata.user_id`` through ``resolve_assistant_for_creator``, the same check
``/update_avatar_identity_with_media`` applies when writing the very rows these two
endpoints read and delete.
"""

from types import SimpleNamespace

import pytest

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
    """Stands in for ``app.state.store``, recording the namespace searched."""

    def __init__(self, items):
        self._items = items
        self.searched_namespaces = []

    async def asearch(self, namespace, limit=None):
        self.searched_namespaces.append(namespace)
        return self._items


def _document_item(filename):
    """A store row shaped like the serialized LangChain Document the endpoints read."""
    return {
        "value": {
            "document": {"kwargs": {"metadata": {"filename": filename}}},
        }
    }


def _install(monkeypatch, metadata, documents=()):
    assistants_api = _AssistantsAPI(metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    store = _Store([_document_item(name) for name in documents])
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    return store


@pytest.mark.asyncio
async def test_the_creator_may_list_their_own_avatars_documents(monkeypatch):
    store = _install(
        monkeypatch, {"user_id": CREATOR_ID}, documents=("Mom.m4a", "resume.pdf")
    )

    response = await webapp_module.list_avatar_documents(
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
    )

    assert response == {"uploaded_documents": ["Mom.m4a", "resume.pdf"]}
    # The store namespace is scoped by the creator's user id, not by any selection.
    assert store.searched_namespaces == [(CREATOR_ID, ASSISTANT_ID)]


@pytest.mark.asyncio
async def test_a_stranger_may_not_list_someone_elses_avatars_documents(monkeypatch):
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, documents=("Mom.m4a",))

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.list_avatar_documents(
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(STRANGER_ID),
        )

    assert rejection.value.status_code == 403
    # Rejected before any store read — a non-creator learns nothing about the avatar.
    assert store.searched_namespaces == []


@pytest.mark.asyncio
async def test_a_stranger_may_not_delete_someone_elses_avatars_document(monkeypatch):
    store = _install(monkeypatch, {"user_id": CREATOR_ID}, documents=("Mom.m4a",))

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.delete_avatar_documents(
            assistant_id=ASSISTANT_ID,
            source_document_name="Mom.m4a",
            current_user=_current_user(STRANGER_ID),
        )

    assert rejection.value.status_code == 403
    # Rejected before the store read that resolves labels, and so before the SQL delete.
    assert store.searched_namespaces == []


@pytest.mark.asyncio
async def test_an_avatar_with_no_recorded_creator_denies_both_endpoints(monkeypatch):
    """Absent ownership fails closed — an unattributed avatar is nobody's to read."""
    _install(monkeypatch, {}, documents=("Mom.m4a",))

    with pytest.raises(webapp_module.HTTPException) as list_rejection:
        await webapp_module.list_avatar_documents(
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
        )
    assert list_rejection.value.status_code == 400

    with pytest.raises(webapp_module.HTTPException) as delete_rejection:
        await webapp_module.delete_avatar_documents(
            assistant_id=ASSISTANT_ID,
            source_document_name="Mom.m4a",
            current_user=_current_user(CREATOR_ID),
        )
    assert delete_rejection.value.status_code == 400


@pytest.mark.asyncio
async def test_an_unloadable_avatar_is_rejected(monkeypatch):
    class _FailingAssistantsAPI:
        async def get(self, assistant_id):
            raise RuntimeError("no such assistant")

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=_FailingAssistantsAPI()),
    )

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.list_avatar_documents(
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
        )

    assert rejection.value.status_code == 400


def _query_parameter_names(path, method):
    for route in webapp_module.app.routes:
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", ()
        ):
            return {
                parameter.name: parameter.field_info.is_required()
                for parameter in route.dependant.query_params
            }
    raise AssertionError(f"route not registered: {method} {path}")


def test_both_document_routes_require_assistant_id():
    """Omitting assistant_id is a 422 from FastAPI, not a silent fallback."""
    listing = _query_parameter_names("/list_avatar_documents", "GET")
    assert listing.get("assistant_id") is True

    deletion = _query_parameter_names("/delete_avatar_document", "DELETE")
    assert deletion.get("assistant_id") is True
    assert deletion.get("source_document_name") is True


def test_the_avatar_selection_endpoints_are_gone():
    """/select_avatar and the selection-only /message must no longer be routable."""
    registered_paths = {
        getattr(route, "path", None) for route in webapp_module.app.routes
    }
    assert "/select_avatar" not in registered_paths
    assert "/message" not in registered_paths
    assert "/message/{assistant_id}" in registered_paths
