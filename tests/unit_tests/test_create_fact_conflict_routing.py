"""Unit tests for the vectorstore-aware routing in ``update_self_identity_mem_from_user_txt``.

The create tool must route by whether the proposed fact already exists in the vectorstore:
an identical stored fact is refused as previously learned, a CONFLICTING stored fact (same
attribute, different claim — e.g. a stored "My favorite color is red." when the user now says
the favorite color is blue) redirects the model to ``edit_identity_fact`` instead of storing a
contradiction or silently dropping the new information, and genuinely new information is
stored. The relationship decision itself is made by
``_classify_proposed_fact_against_stored_facts``; both the classifier plumbing (with a stubbed
model) and the tool-level routing (with a canned classifier verdict) are exercised without a
live model.
"""

import math
import re
import uuid

import pytest
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    _classify_proposed_fact_against_stored_facts,
    _ProposedFactStoredFactRelationship,
    update_self_identity_mem_from_user_txt,
    wrap_fact_with_context,
)

CREATOR = "creator-1"
ASSISTANT = "asst-1"
IDENTITY_MEMORY_NAMESPACE = (CREATOR, ASSISTANT, "identity_memory")

PROPOSED_FACT = "My favorite color is blue."
CONFLICTING_STORED_FACT = "My favorite color is red."
UNRELATED_STORED_FACT = "I love hockey."

# Deterministic keyword embedding for the STORE INDEX: each dim is 1.0 iff the keyword is
# present. Used only so ``store.asearch`` returns candidate documents; the relatedness gate
# is the monkeypatched sentence scorer below.
_KEYWORDS = ["color", "blue", "red", "hockey", "glasses"]


def _embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        low = (text or "").lower()
        raw = [1.0 if keyword in low else 0.0 for keyword in _KEYWORDS]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        vectors.append([value / norm for value in raw])
    return vectors


def _make_store() -> InMemoryStore:
    return InMemoryStore(
        index={
            "dims": len(_KEYWORDS),
            "embed": _embed,
            "fields": ["document.kwargs.page_content"],
        }
    )


async def _fake_score_sentences(query: str, sentences: list[str]) -> list[float]:
    """Deterministic stand-in for the SentenceTransformer scorer: set-cosine over word
    tokens, crossing ``_CORRECTION_MATCH_THRESHOLD`` (0.65) exactly where a real model
    would — "My favorite color is red." scores 0.8 against the blue proposal, "I love
    hockey." scores near zero."""
    query_tokens = set(re.findall(r"[a-z']+", (query or "").lower()))
    scores: list[float] = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"[a-z']+", (sentence or "").lower()))
        if not query_tokens or not sentence_tokens:
            scores.append(0.0)
            continue
        shared = len(query_tokens & sentence_tokens)
        scores.append(
            shared / (math.sqrt(len(query_tokens)) * math.sqrt(len(sentence_tokens)))
        )
    return scores


@pytest.fixture(autouse=True)
def _patch_sentence_scorer(monkeypatch):
    monkeypatch.setattr(identity_tools, "_score_sentences", _fake_score_sentences)


@pytest.fixture(autouse=True)
def _patch_message_fact_similarity(monkeypatch):
    """Skip the message-grounding embedder (a live SentenceTransformer) by making the
    threaded similarity computation report a perfect match — grounding is covered by
    ``test_self_identity_fact_grounding.py``; these tests target the routing AFTER the
    grounding guard passes."""

    async def _fake_to_thread(function, *args, **kwargs):
        return 1.0

    monkeypatch.setattr(identity_tools.asyncio, "to_thread", _fake_to_thread)


@pytest.fixture(autouse=True)
def _patch_id_extraction(monkeypatch):
    async def _fake_extract(config):
        return {"user_id": CREATOR}, {"assistant_id": ASSISTANT}

    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)


def _patch_relationship(monkeypatch, result):
    """Replace the relationship classifier with a canned verdict (or ``None`` for the
    model-error fallback path). Records each call's arguments."""
    calls: list = []

    async def _fake_classify(proposed_fact, stored_facts):
        calls.append((proposed_fact, list(stored_facts)))
        return result

    monkeypatch.setattr(
        identity_tools, "_classify_proposed_fact_against_stored_facts", _fake_classify
    )
    return calls


async def _seed_fact(store, fact):
    document_id = str(uuid.uuid4())
    metadata = {
        "user_id": CREATOR,
        "assistant_id": ASSISTANT,
        "document_id": document_id,
        "fact": fact,
        "fact_context": "ctx",
    }
    document = Document(
        page_content=wrap_fact_with_context(fact, "ctx"), metadata=metadata
    )
    await store.aput(
        IDENTITY_MEMORY_NAMESPACE, key=document_id, value={"document": document.to_json()}
    )
    return document_id


class _FakeRuntime:
    def __init__(self, store, *, state_documents=None):
        self.store = store
        self.tool_call_id = "tc-1"
        self.state = {
            "messages": [HumanMessage(content="Your favorite color is blue.")],
            "assistant_identity_documents": state_documents or [],
        }
        self.config = {
            "configurable": {
                "user_id": CREATOR,
                "assistant_id": ASSISTANT,
                "assistant_ctx": {"metadata": {"user_id": CREATOR}},
            }
        }


def _create_coroutine():
    # The @tool wrapper exposes the async implementation on ``.coroutine``.
    return update_self_identity_mem_from_user_txt.coroutine


async def _stored_fact_count(store):
    items = await store.asearch(IDENTITY_MEMORY_NAMESPACE, query=PROPOSED_FACT, limit=50)
    return len(items)


def _tool_message_content(command):
    return command.update["messages"][0].content


# --------------------------------------------------------------------------------------
# Classifier plumbing (stubbed model)
# --------------------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, result):
        self._result = result
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._result


@pytest.mark.asyncio
async def test_classifier_hands_proposed_and_stored_facts_to_the_model(monkeypatch):
    verdict = _ProposedFactStoredFactRelationship(
        relationship="conflicts_with_stored_fact",
        conflicting_stored_fact=CONFLICTING_STORED_FACT,
        reason="stub",
    )
    model = _FakeModel(verdict)
    monkeypatch.setattr(identity_tools, "init_model", lambda **kwargs: model)

    result = await _classify_proposed_fact_against_stored_facts(
        PROPOSED_FACT, [CONFLICTING_STORED_FACT, UNRELATED_STORED_FACT]
    )

    assert result is verdict
    (system_message, human_message) = model.calls[0]
    assert isinstance(system_message, SystemMessage)
    assert PROPOSED_FACT in human_message.content
    assert f"1. {CONFLICTING_STORED_FACT}" in human_message.content
    assert f"2. {UNRELATED_STORED_FACT}" in human_message.content


@pytest.mark.asyncio
async def test_classifier_returns_none_on_model_error(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(identity_tools, "init_model", _boom)
    assert (
        await _classify_proposed_fact_against_stored_facts(
            PROPOSED_FACT, [CONFLICTING_STORED_FACT]
        )
        is None
    )


# --------------------------------------------------------------------------------------
# Tool-level routing (canned classifier verdict)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conflicting_stored_fact_redirects_to_edit(monkeypatch):
    store = _make_store()
    await _seed_fact(store, CONFLICTING_STORED_FACT)
    calls = _patch_relationship(
        monkeypatch,
        _ProposedFactStoredFactRelationship(
            relationship="conflicts_with_stored_fact",
            conflicting_stored_fact=CONFLICTING_STORED_FACT,
            reason="stub",
        ),
    )

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context='The user said: "Your favorite color is blue."',
        runtime=_FakeRuntime(store),
    )

    content = _tool_message_content(command)
    assert "edit_identity_fact" in content
    assert f'inaccurate_information="{CONFLICTING_STORED_FACT}"' in content
    assert f'corrected_information="{PROPOSED_FACT}"' in content
    # The conflicting stored fact was handed to the classifier, and nothing was stored.
    assert calls == [(PROPOSED_FACT, [CONFLICTING_STORED_FACT])]
    assert await _stored_fact_count(store) == 1


@pytest.mark.asyncio
async def test_already_stored_verdict_refuses_as_previously_learned(monkeypatch):
    store = _make_store()
    await _seed_fact(store, CONFLICTING_STORED_FACT)
    _patch_relationship(
        monkeypatch,
        _ProposedFactStoredFactRelationship(
            relationship="already_stored", conflicting_stored_fact="", reason="stub"
        ),
    )

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context="ctx",
        runtime=_FakeRuntime(store),
    )

    assert "previously learned" in _tool_message_content(command)
    assert await _stored_fact_count(store) == 1


@pytest.mark.asyncio
async def test_new_information_verdict_stores_the_fact(monkeypatch):
    store = _make_store()
    await _seed_fact(store, CONFLICTING_STORED_FACT)
    _patch_relationship(
        monkeypatch,
        _ProposedFactStoredFactRelationship(
            relationship="new_information", conflicting_stored_fact="", reason="stub"
        ),
    )

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context="ctx",
        runtime=_FakeRuntime(store),
    )

    assert f"Learned: {PROPOSED_FACT}" in _tool_message_content(command)
    assert await _stored_fact_count(store) == 2


@pytest.mark.asyncio
async def test_unrelated_stored_facts_skip_the_classifier_and_store(monkeypatch):
    """No stored fact clears the relatedness gate -> the classifier is never consulted and
    the fact is stored directly."""
    store = _make_store()
    await _seed_fact(store, UNRELATED_STORED_FACT)
    calls = _patch_relationship(monkeypatch, None)

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context="ctx",
        runtime=_FakeRuntime(store),
    )

    assert f"Learned: {PROPOSED_FACT}" in _tool_message_content(command)
    assert calls == []


@pytest.mark.asyncio
async def test_classifier_error_falls_back_to_similarity_duplicate_rule(monkeypatch):
    """Classifier returns ``None`` (model outage): an identical stored fact (clean-fact
    score 1.0 > 0.8) is still refused as previously learned — the pre-existing behavior."""
    store = _make_store()
    await _seed_fact(store, PROPOSED_FACT)
    _patch_relationship(monkeypatch, None)

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context="ctx",
        runtime=_FakeRuntime(store),
    )

    assert "previously learned" in _tool_message_content(command)
    assert await _stored_fact_count(store) == 1


@pytest.mark.asyncio
async def test_exact_state_duplicate_short_circuits_without_classifier(monkeypatch):
    store = _make_store()
    state_document = Document(
        page_content=wrap_fact_with_context(PROPOSED_FACT, "ctx"),
        metadata={"fact": PROPOSED_FACT},
    )
    calls = _patch_relationship(monkeypatch, None)

    command = await _create_coroutine()(
        fact_shared_about_the_assistant_from_the_user=PROPOSED_FACT,
        fact_context="ctx",
        runtime=_FakeRuntime(store, state_documents=[state_document]),
    )

    assert "previously learned" in _tool_message_content(command)
    assert calls == []
    assert await _stored_fact_count(store) == 0
