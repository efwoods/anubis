"""Unit tests for the conversational fact-correction tool.

Covers the deterministic core — multi-namespace search (whole-document for atomic facts,
sentence-level for long verbatim documents), in-place rewrite, sentence redaction, and
deletion — plus the tool's human-in-the-loop decision handling (approve / edit / reject +
owner guard). The LangGraph ``interrupt`` is monkeypatched to a canned decision, and the
runtime sentence embedder is monkeypatched to a deterministic bag-of-words scorer, so the
tool can be exercised without a live agent/checkpointer or a model download.
"""

import math
import re
import uuid

import pytest
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    apply_fact_correction,
    correct_identity_fact,
    find_fact_matches,
    wrap_fact_with_context,
)

CREATOR = "creator-1"
ASSISTANT = "asst-1"
WRONG_FACT = "I was born in Toronto."
RIGHT_FACT = "I was born in Ottawa."

# Deterministic keyword embedding for the STORE INDEX: each dim is 1.0 iff the keyword is
# present. Used only so `store.asearch` returns the right candidate documents; the actual
# match decision for long-text namespaces is made by the monkeypatched sentence scorer.
_KEYWORDS = ["toronto", "ottawa", "montreal", "hockey", "glasses", "alberta"]


def _embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        low = (text or "").lower()
        raw = [1.0 if kw in low else 0.0 for kw in _KEYWORDS]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        vectors.append([x / norm for x in raw])
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
    tokens. A near-restatement of the claim scores high; an unrelated sentence scores low —
    crossing _SENTENCE_MATCH_THRESHOLD (0.45) exactly where a real model would."""
    query_tokens = set(re.findall(r"[a-z']+", (query or "").lower()))
    scores: list[float] = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"[a-z']+", (sentence or "").lower()))
        if not query_tokens or not sentence_tokens:
            scores.append(0.0)
            continue
        shared = len(query_tokens & sentence_tokens)
        scores.append(shared / (math.sqrt(len(query_tokens)) * math.sqrt(len(sentence_tokens))))
    return scores


@pytest.fixture(autouse=True)
def _patch_sentence_scorer(monkeypatch):
    monkeypatch.setattr(identity_tools, "_score_sentences", _fake_score_sentences)


async def _fake_suggest_correction(
    match, *, inaccurate_information, corrected_information, correction_context, is_deletion
):
    """Deterministic stand-in for the per-document LLM suggestion: echo the global
    correction (empty on delete) and mark every match as asserting the fact. Lets the tool's
    HITL flow run without a model download or network call."""
    return ("" if is_deletion else corrected_information, correction_context, True)


@pytest.fixture(autouse=True)
def _patch_suggestion(monkeypatch):
    monkeypatch.setattr(identity_tools, "_suggest_correction", _fake_suggest_correction)


async def _seed(store, namespace, fact, *, id_field="document_id", extra_meta=None):
    """Seed an ATOMIC fact document (identity/identity_memory/memory shape)."""
    doc_id = str(uuid.uuid4())
    metadata = {
        "user_id": CREATOR,
        "assistant_id": ASSISTANT,
        id_field: doc_id,
        "fact": fact,
        "fact_context": "ctx",
    }
    if extra_meta:
        metadata.update(extra_meta)
    doc = Document(page_content=wrap_fact_with_context(fact, "ctx"), metadata=metadata)
    await store.aput(namespace, key=doc_id, value={"document": doc.to_json()})
    return doc_id


async def _seed_long_text(store, namespace, page_content, *, extra_meta=None):
    """Seed a long verbatim document (quote/document/analysis shape: raw page_content,
    no ``fact`` metadata key)."""
    doc_id = str(uuid.uuid4())
    metadata = {"user_id": CREATOR, "assistant_id": ASSISTANT, "document_id": doc_id}
    if extra_meta:
        metadata.update(extra_meta)
    doc = Document(page_content=page_content, metadata=metadata)
    await store.aput(namespace, key=doc_id, value={"document": doc.to_json()})
    return doc_id


# A multi-sentence verbatim quote with exactly one sentence asserting the wrong fact.
QUOTE_TEXT = "I love hockey. I was born in Toronto. The weather is nice."
QUOTE_OFFENDING = "I was born in Toronto."


class _FakeRuntime:
    def __init__(self, store, *, requester=CREATOR):
        self.store = store
        self.tool_call_id = "tc-1"
        self.state = {
            "assistant_identity_documents": [],
            "recalled_memory_documents": [],
            "user_identity_documents": [],
        }
        self.config = {
            "configurable": {
                "user_id": requester,
                "assistant_id": ASSISTANT,
                "assistant_ctx": {"metadata": {"user_id": CREATOR}},
            }
        }


def _tool_coroutine():
    # The @tool wrapper exposes the async implementation on ``.coroutine``.
    return correct_identity_fact.coroutine


async def _read_doc(store, namespace, key):
    item = await store.aget(namespace, key)
    return item.value["document"]["kwargs"]


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_fact_matches_spans_atomic_namespaces():
    store = _make_store()
    media_uuid5 = str(uuid.uuid5(uuid.NAMESPACE_URL, "resume.pdf"))
    await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    await _seed(
        store, (CREATOR, ASSISTANT, "identity", media_uuid5), WRONG_FACT,
        id_field="id", extra_meta={"filename": "resume.pdf"},
    )
    await _seed(store, (CREATOR, ASSISTANT, "identity_memory"), WRONG_FACT)
    await _seed(store, (CREATOR, ASSISTANT, "memory"), WRONG_FACT, id_field="id")
    await _seed(store, (CREATOR, ASSISTANT, "identity"), "I love hockey.")  # unrelated

    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    fact_matches = [m for m in matches if m.kind == "fact"]
    matched = [m.matched_text for m in fact_matches]
    assert matched.count(WRONG_FACT) == 4  # incl. the media sub-namespace
    assert "I love hockey." not in matched


@pytest.mark.asyncio
async def test_find_fact_matches_locates_sentence_in_quote():
    store = _make_store()
    await _seed_long_text(store, (CREATOR, ASSISTANT, "quote"), QUOTE_TEXT)
    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    sentence_matches = [m for m in matches if m.kind == "sentence"]
    assert [m.matched_text for m in sentence_matches] == [QUOTE_OFFENDING]


@pytest.mark.asyncio
async def test_analysis_namespace_is_swept():
    store = _make_store()
    await _seed_long_text(
        store, (CREATOR, ASSISTANT, "analysis"),
        "Openness is high. I was born in Toronto. Conscientiousness is moderate.",
        extra_meta={"analysis_type": "biography"},
    )
    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    assert any(
        m.kind == "sentence" and m.namespace[2] == "analysis" and m.matched_text == QUOTE_OFFENDING
        for m in matches
    )


@pytest.mark.asyncio
async def test_find_fact_matches_catches_paraphrase_at_production_scale():
    """An atomic fact phrased differently from the query still matches (regression guard
    for the threshold: the old 0.8 gate dropped real matches that score sub-0.8)."""
    vocab = ["i", "was", "born", "in", "toronto", "grew", "up", "raised"]

    def bow_embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            tokens = re.findall(r"[a-z]+", (text or "").lower())
            raw = [float(tokens.count(w)) for w in vocab]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors

    store = InMemoryStore(
        index={"dims": len(vocab), "embed": bow_embed, "fields": ["document.kwargs.page_content"]}
    )
    paraphrase = "I grew up in Toronto."
    await _seed(store, (CREATOR, ASSISTANT, "identity"), paraphrase)
    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    assert [m.matched_text for m in matches] == [paraphrase]


# --------------------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_fact_correction_rewrites_in_place():
    store = _make_store()
    await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    original_key = matches[0].key

    changes = await apply_fact_correction(
        store,
        corrected_information=RIGHT_FACT,
        correction_context="I was told I was born in Ottawa.",
        matches=matches,
    )
    assert len(changes) == 1 and changes[0].action == "rewrite"
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), original_key))["metadata"]
    assert meta["fact"] == RIGHT_FACT
    assert meta["corrected_from"] == WRONG_FACT
    page_content = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), original_key))["page_content"]
    assert RIGHT_FACT in page_content


@pytest.mark.asyncio
async def test_apply_sentence_redaction_preserves_rest():
    store = _make_store()
    key = await _seed_long_text(store, (CREATOR, ASSISTANT, "quote"), QUOTE_TEXT)
    matches = await find_fact_matches(
        store, creator_id=CREATOR, assistant_id=ASSISTANT, user_id=CREATOR, query=WRONG_FACT
    )
    changes = await apply_fact_correction(
        store, corrected_information="", correction_context="", matches=matches, is_deletion=True
    )
    assert len(changes) == 1 and changes[0].action == "redact"
    kwargs = await _read_doc(store, (CREATOR, ASSISTANT, "quote"), key)
    assert QUOTE_OFFENDING not in kwargs["page_content"]
    assert "I love hockey." in kwargs["page_content"]
    assert "The weather is nice." in kwargs["page_content"]
    assert QUOTE_OFFENDING in kwargs["metadata"]["redacted_sentences"]


# --------------------------------------------------------------------------------------
# Tool (HITL)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_approve_corrects_fact_and_redacts_quote(monkeypatch):
    store = _make_store()
    await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    quote_key = await _seed_long_text(store, (CREATOR, ASSISTANT, "quote"), QUOTE_TEXT)
    captured = {}

    def _fake_interrupt(payload):
        captured["payload"] = payload
        # Accept the model's suggested edit on every matched document.
        return {
            "type": "apply",
            "items": [{"index": m["index"], "action": "accept"} for m in payload["matches"]],
        }

    monkeypatch.setattr(identity_tools, "interrupt", _fake_interrupt)

    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="I was told I was born in Ottawa.",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )

    assert captured["payload"]["kind"] == "fact_correction"
    kinds = {m["kind"] for m in captured["payload"]["matches"]}
    assert kinds == {"fact", "sentence"}

    # The atomic fact is rewritten; the quote sentence is replaced with the correction.
    quote_kwargs = await _read_doc(store, (CREATOR, ASSISTANT, "quote"), quote_key)
    assert QUOTE_OFFENDING not in quote_kwargs["page_content"]
    assert RIGHT_FACT in quote_kwargs["page_content"]
    assert "Applied 2 change(s)" in cmd.update["messages"][0].content


@pytest.mark.asyncio
async def test_tool_delete_removes_atomic_fact(monkeypatch):
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(
        identity_tools, "interrupt",
        lambda payload: {
            "type": "apply",
            "items": [{"index": m["index"], "action": "remove"} for m in payload["matches"]],
        },
    )

    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information="",
        correction_context="That never happened.",
        correction_kind="delete",
        runtime=_FakeRuntime(store),
    )
    assert await store.aget((CREATOR, ASSISTANT, "identity"), key) is None
    assert "Deleted" in cmd.update["messages"][0].content


@pytest.mark.asyncio
async def test_tool_edit_uses_owner_revision(monkeypatch):
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(
        identity_tools, "interrupt",
        lambda payload: {
            "type": "apply",
            "items": [
                {
                    "index": m["index"],
                    "action": "edit",
                    "corrected_text": "I was born in Montreal.",
                    "correction_context": "edited",
                }
                for m in payload["matches"]
            ],
        },
    )
    await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key))["metadata"]
    assert meta["fact"] == "I was born in Montreal."


@pytest.mark.asyncio
async def test_tool_reject_leaves_store_untouched(monkeypatch):
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(identity_tools, "interrupt", lambda payload: {"type": "reject"})
    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key))["metadata"]
    assert meta["fact"] == WRONG_FACT  # unchanged
    assert "unchanged" in cmd.update["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_tool_owner_guard_blocks_non_creator(monkeypatch):
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)

    def _should_not_run(payload):  # pragma: no cover
        raise AssertionError("interrupt must not be reached for a non-owner")

    monkeypatch.setattr(identity_tools, "interrupt", _should_not_run)
    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store, requester="someone-else"),
    )
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key))["metadata"]
    assert meta["fact"] == WRONG_FACT  # unchanged
    assert "creator" in cmd.update["messages"][0].content.lower()


# --------------------------------------------------------------------------------------
# Per-document HITL (each match edited independently; one approve applies all)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_exposes_one_editable_item_per_match(monkeypatch):
    """The interrupt payload carries each matched document as its own editable item with an
    index, current text, a suggested edit, and a recommended/default action (no global field)."""
    store = _make_store()
    await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    await _seed(store, (CREATOR, ASSISTANT, "identity"), "I grew up in Toronto.")
    captured = {}

    def _fake_interrupt(payload):
        captured["payload"] = payload
        return {"type": "cancel"}

    monkeypatch.setattr(identity_tools, "interrupt", _fake_interrupt)
    await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    payload = captured["payload"]
    assert payload["default_action"] == "skip"
    assert payload["actions"] == ["skip", "accept", "edit", "remove"]
    items = payload["matches"]
    assert len(items) == 2
    assert {m["index"] for m in items} == {0, 1}
    for m in items:
        assert set(m) >= {
            "index", "current_text", "suggested_text", "suggested_context",
            "default_action", "recommended_action",
        }
        assert m["default_action"] == "skip"
    assert "proposed" not in payload  # no single global correction field


@pytest.mark.asyncio
async def test_per_document_edits_apply_distinct_text(monkeypatch):
    """Each matched document is rewritten with ITS OWN corrected text, not one shared value."""
    store = _make_store()
    key_a = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    key_b = await _seed(store, (CREATOR, ASSISTANT, "identity"), "I grew up in Toronto.")

    def _fake_interrupt(payload):
        return {
            "type": "apply",
            "items": [
                {
                    "index": m["index"],
                    "action": "edit",
                    "corrected_text": f"corrected-{m['key']}",
                    "correction_context": "ctx",
                }
                for m in payload["matches"]
            ],
        }

    monkeypatch.setattr(identity_tools, "interrupt", _fake_interrupt)
    await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    meta_a = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key_a))["metadata"]
    meta_b = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key_b))["metadata"]
    assert meta_a["fact"] == f"corrected-{key_a}"
    assert meta_b["fact"] == f"corrected-{key_b}"


@pytest.mark.asyncio
async def test_excluded_document_is_left_untouched(monkeypatch):
    """A document the owner skips is not modified, even though it matched."""
    store = _make_store()
    await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    await _seed(store, (CREATOR, ASSISTANT, "identity"), "I grew up in Toronto.")
    captured = {}

    def _fake_interrupt(payload):
        captured["payload"] = payload
        first, second = payload["matches"]
        return {
            "type": "apply",
            "items": [
                {"index": first["index"], "action": "edit", "corrected_text": "kept", "correction_context": "c"},
                {"index": second["index"], "action": "skip", "corrected_text": "ignored", "correction_context": "c"},
            ],
        }

    monkeypatch.setattr(identity_tools, "interrupt", _fake_interrupt)
    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    included_key = captured["payload"]["matches"][0]["key"]
    excluded_key = captured["payload"]["matches"][1]["key"]
    included_meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), included_key))["metadata"]
    excluded_meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), excluded_key))["metadata"]
    assert included_meta["fact"] == "kept"
    assert excluded_meta["fact"] in (WRONG_FACT, "I grew up in Toronto.")  # unchanged
    assert "corrected_from" not in excluded_meta
    assert "Applied 1 change(s)" in cmd.update["messages"][0].content


@pytest.mark.asyncio
async def test_default_skip_leaves_everything_untouched(monkeypatch):
    """A bare apply with no per-item actions changes nothing — the safe default is 'skip', so
    falsely-retrieved documents are never silently edited or deleted."""
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(identity_tools, "interrupt", lambda payload: {"type": "apply"})
    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key))["metadata"]
    assert meta["fact"] == WRONG_FACT  # untouched
    assert "corrected_from" not in meta
    assert "unchanged" in cmd.update["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_accept_applies_model_suggestion(monkeypatch):
    """'accept' applies the per-document suggested edit without the owner retyping it."""
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(
        identity_tools, "interrupt",
        lambda payload: {
            "type": "apply",
            "items": [{"index": m["index"], "action": "accept"} for m in payload["matches"]],
        },
    )
    await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,  # echoed by the fake per-document suggester
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    meta = (await _read_doc(store, (CREATOR, ASSISTANT, "identity"), key))["metadata"]
    assert meta["fact"] == RIGHT_FACT


@pytest.mark.asyncio
async def test_remove_on_update_deletes_the_fact(monkeypatch):
    """'remove' deletes the matched fact even on an 'update'-kind correction (the owner can
    choose to drop a document rather than rewrite it)."""
    store = _make_store()
    key = await _seed(store, (CREATOR, ASSISTANT, "identity"), WRONG_FACT)
    monkeypatch.setattr(
        identity_tools, "interrupt",
        lambda payload: {
            "type": "apply",
            "items": [{"index": m["index"], "action": "remove"} for m in payload["matches"]],
        },
    )
    cmd = await _tool_coroutine()(
        inaccurate_information=WRONG_FACT,
        corrected_information=RIGHT_FACT,
        correction_context="ctx",
        correction_kind="update",
        runtime=_FakeRuntime(store),
    )
    assert await store.aget((CREATOR, ASSISTANT, "identity"), key) is None
    assert "Deleted" in cmd.update["messages"][0].content
