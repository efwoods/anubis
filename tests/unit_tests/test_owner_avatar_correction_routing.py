"""An owner correcting a fact about the avatar must change the avatar, for every viewer.

Regression for avatar 47cfdaa2 (production, 2026-10-14 to 2026-10-29). The owner wrote
"I need you to learn your linked in site is linkedin.com/in/evanfwoods NOT
linkedin.com/in/evanwoods" and "I need you to learn that you do NOT use ROUGE and BERT
scores". Neither correction reached the avatar's stored facts:

- The ROUGE correction was stored by ``learn_user_preference`` (and once by
  ``learn_information_about_the_user``) under the owner's own user id, which an
  anonymous visitor never reads, while the stored biography still said "rouge and BERT
  scores" inside the ``fact_context`` of eight identity facts.
- The LinkedIn correction was stored as a new fact beside the wrong URL, which lived
  only in the ``source_urls`` metadata of research facts. ``organization_links`` lifted
  that metadata URL into the prompt as a link the avatar shares, so the avatar kept
  giving out the wrong profile.

These tests pin the routing guards, the URL / context matching in
``find_fact_matches``, and the in-place rewrite of URL metadata and context sentences.
"""

import math
import re
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.tools.identity.identity_tools as identity_tools
import src.anubis.utils.tools.identity.learning_tools as learning_tools
from src.anubis.utils.organization_links import organization_links_from_identity
from src.anubis.utils.tools.identity.identity_tools import (
    edit_identity_fact,
    find_fact_matches,
    learn_information_about_the_user,
    message_corrects_a_fact_about_the_avatar,
    update_self_identity_mem_from_user_txt,
    wrap_fact_with_context,
)

CREATOR = "creator-1"
VISITOR = "visitor-1"
ASSISTANT = "asst-1"
IDENTITY_NAMESPACE = (CREATOR, ASSISTANT, "identity")
IDENTITY_MEMORY_NAMESPACE = (CREATOR, ASSISTANT, "identity_memory")

LINKEDIN_CORRECTION_MESSAGE = (
    "I need you to learn your linked in site is linkedin.com/in/evanfwoods NOT "
    "linkedin.com/in/evanwoods"
)
ROUGE_CORRECTION_MESSAGE = (
    "I need you to learn that you do NOT use ROUGE and BERT scores to determine "
    "authenticity; The neuralnexus uses stylometrics indicators such as moving average "
    "type token ratio and key phrase ratio to determine true-to-life authentic quality of "
    "the generated responses."
)
WRONG_PROFILE_URL = "https://www.linkedin.com/in/evanwoods"
RIGHT_PROFILE_URL = "https://www.linkedin.com/in/evanfwoods"
BIOGRAPHY_ROUGE_SENTENCE = (
    "Creating statistical proof of the authenticity of the responses using metrics such as "
    "rouge and BERT scores."
)
BIOGRAPHY_CONTEXT = (
    "I am working on a startup, Afterlife Systems Incorporated. "
    f"{BIOGRAPHY_ROUGE_SENTENCE} "
    "I am looking for a partner to join me in an incubator."
)

_KEYWORDS = ["linkedin", "rouge", "startup", "english", "employed"]


def _embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered_text = (text or "").lower()
        raw = [1.0 if keyword in lowered_text else 0.0 for keyword in _KEYWORDS]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        vectors.append([value / norm for value in raw])
    return vectors


def _make_store() -> InMemoryStore:
    return InMemoryStore(
        index={"dims": len(_KEYWORDS), "embed": _embed, "fields": ["document.kwargs.page_content"]}
    )


async def _fake_score_sentences(query: str, sentences: list[str]) -> list[float]:
    """Set-cosine over word tokens, a deterministic stand-in for the sentence embedder."""
    query_tokens = set(re.findall(r"[a-z']+", (query or "").lower()))
    scores: list[float] = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"[a-z']+", (sentence or "").lower()))
        if not query_tokens or not sentence_tokens:
            scores.append(0.0)
            continue
        shared_token_count = len(query_tokens & sentence_tokens)
        scores.append(
            shared_token_count / (math.sqrt(len(query_tokens)) * math.sqrt(len(sentence_tokens)))
        )
    return scores


async def _fake_suggest_correction(
    match, *, inaccurate_information, corrected_information, correction_context, is_deletion
):
    return ("" if is_deletion else corrected_information, correction_context, True)


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch):
    monkeypatch.setattr(identity_tools, "_score_sentences", _fake_score_sentences)
    monkeypatch.setattr(identity_tools, "_suggest_correction", _fake_suggest_correction)

    async def _fake_grounds_fact(*args, **kwargs):
        return True

    monkeypatch.setattr(identity_tools, "_user_message_grounds_fact", _fake_grounds_fact)

    async def _fake_extract(config):
        return (
            {"user_id": config["configurable"]["user_id"]},
            {"assistant_id": ASSISTANT},
        )

    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)
    monkeypatch.setattr(learning_tools, "extract_user_id_assistant_id", _fake_extract)


class _FakeRuntime:
    def __init__(self, store, *, requester=CREATOR, latest_user_message=""):
        self.store = store
        self.tool_call_id = "tool-call-1"
        self.state = {
            "messages": [HumanMessage(content=latest_user_message)],
            "assistant_identity_documents": [],
            "recalled_memory_documents": [],
            "user_identity_documents": [],
        }
        self.config = {
            "configurable": {
                "user_id": requester,
                "assistant_id": ASSISTANT,
                "assistant_ctx": {"name": "Evan Woods", "metadata": {"user_id": CREATOR}},
            }
        }


async def _seed_research_fact(store, fact, source_url):
    document_id = str(uuid.uuid4())
    metadata = {
        "fact": fact,
        "source": "deep_research",
        "user_id": CREATOR,
        "assistant_id": ASSISTANT,
        "document_id": document_id,
        "source_urls": [source_url],
        "fact_context": "In the LinkedIn profile header, the status is listed.",
    }
    document = Document(
        page_content=wrap_fact_with_context(fact, metadata["fact_context"]), metadata=metadata
    )
    await store.aput(IDENTITY_NAMESPACE, key=document_id, value={"document": document.to_json()})
    return document_id


async def _seed_biography_fact(store, fact):
    document_id = str(uuid.uuid4())
    metadata = {
        "fact": fact,
        "user_id": CREATOR,
        "assistant_id": ASSISTANT,
        "document_id": document_id,
        "fact_context": BIOGRAPHY_CONTEXT,
    }
    document = Document(
        page_content=wrap_fact_with_context(fact, BIOGRAPHY_CONTEXT), metadata=metadata
    )
    await store.aput(
        IDENTITY_MEMORY_NAMESPACE, key=document_id, value={"document": document.to_json()}
    )
    return document_id


async def _stored_document(store, namespace, key):
    return Document(**(await store.aget(namespace, key)).value["document"]["kwargs"])


def _accept_every_match(captured_payload):
    def _fake_interrupt(payload):
        captured_payload["payload"] = payload
        return {
            "type": "apply",
            "items": [{"index": match["index"], "action": "accept"} for match in payload["matches"]],
        }

    return _fake_interrupt


# --------------------------------------------------------------------------------------
# Detecting a correction of the avatar in the owner's own message
# --------------------------------------------------------------------------------------

AVATAR_CORRECTIONS = [
    LINKEDIN_CORRECTION_MESSAGE,
    ROUGE_CORRECTION_MESSAGE,
    # A passage pasted after a colon carries its own "you can"; only the clause that
    # holds the correction is judged.
    "I need you to learn that you do not use ROUGE or BERT Scores in the following passage: "
    "Hey - tell me all about yourself and what you can do for me",
    "Actually you were born in Ottawa, not Toronto.",
]

NOT_AVATAR_CORRECTIONS = [
    "I want you to know that I like hiking",
    "Call me Sam, keep things short, and skip the pep talks.",
    "learn that you should not ask follow-up questions",
    "Okay, you should be shorter, not longer",
    "I don't like long replies",
    "I want you to stop asking me questions, don't do that",
    "Hey - tell me all about yourself and what you can do for me",
]


@pytest.mark.parametrize("message_text", AVATAR_CORRECTIONS)
def test_owner_message_correcting_the_avatar_is_detected(message_text):
    assert message_corrects_a_fact_about_the_avatar(message_text)


@pytest.mark.parametrize("message_text", NOT_AVATAR_CORRECTIONS)
def test_preference_or_user_fact_is_not_a_correction_of_the_avatar(message_text):
    assert not message_corrects_a_fact_about_the_avatar(message_text)


# --------------------------------------------------------------------------------------
# The wrong tools refuse an owner's correction of the avatar
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preference_tool_redirects_owner_correction_and_stores_nothing(monkeypatch):
    stored_preferences: list = []

    async def _fake_store_user_preference(*args, **kwargs):
        stored_preferences.append(kwargs)
        return Document(page_content="stored")

    monkeypatch.setattr(learning_tools, "store_user_preference", _fake_store_user_preference)

    command = await learning_tools.learn_user_preference.coroutine(
        preference="The user prefers authenticity to be determined without using ROUGE or BERT scores.",
        preference_context="The user corrected the authenticity methodology.",
        runtime=_FakeRuntime(_make_store(), latest_user_message=ROUGE_CORRECTION_MESSAGE),
    )

    tool_message_content = command.update["messages"][0].content
    assert tool_message_content.startswith("Not learned:")
    assert "edit_identity_fact" in tool_message_content
    assert stored_preferences == []


@pytest.mark.asyncio
async def test_preference_tool_still_stores_a_visitor_preference(monkeypatch):
    stored_preferences: list = []

    async def _fake_store_user_preference(*args, **kwargs):
        stored_preferences.append(kwargs)
        return Document(page_content="stored")

    monkeypatch.setattr(learning_tools, "store_user_preference", _fake_store_user_preference)

    command = await learning_tools.learn_user_preference.coroutine(
        preference="The user wants short replies.",
        preference_context="The user asked for short replies.",
        runtime=_FakeRuntime(
            _make_store(), requester=VISITOR, latest_user_message=ROUGE_CORRECTION_MESSAGE
        ),
    )

    assert command.update["messages"][0].content.startswith("Learned preference:")
    assert len(stored_preferences) == 1


@pytest.mark.asyncio
async def test_user_fact_tool_redirects_owner_correction_of_the_avatar():
    store = _make_store()
    command = await learn_information_about_the_user.coroutine(
        user_fact="I do not use ROUGE or BERT scores as indicators of authenticity.",
        fact_context="The user corrected the authenticity methodology.",
        runtime=_FakeRuntime(store, latest_user_message=ROUGE_CORRECTION_MESSAGE),
    )

    assert "edit_identity_fact" in command.update["messages"][0].content
    assert await store.asearch((ASSISTANT, CREATOR, "identity"), query="rouge") == []


@pytest.mark.asyncio
async def test_create_tool_redirects_a_negated_url_that_is_still_stored():
    store = _make_store()
    await _seed_research_fact(store, "I'm self-employed.", WRONG_PROFILE_URL)

    command = await update_self_identity_mem_from_user_txt.coroutine(
        fact_shared_about_the_assistant_from_the_user=(
            "My LinkedIn site is linkedin.com/in/evanfwoods, not linkedin.com/in/evanwoods."
        ),
        fact_context="I was told my LinkedIn site is linkedin.com/in/evanfwoods.",
        runtime=_FakeRuntime(store, latest_user_message=LINKEDIN_CORRECTION_MESSAGE),
    )

    tool_message_content = command.update["messages"][0].content
    assert 'inaccurate_information="linkedin.com/in/evanwoods"' in tool_message_content
    assert await store.asearch(IDENTITY_MEMORY_NAMESPACE, query="linkedin") == []


# --------------------------------------------------------------------------------------
# edit_identity_fact finds and rewrites every copy of the wrong value
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_fact_matches_finds_a_url_stored_only_in_metadata():
    store = _make_store()
    research_key = await _seed_research_fact(store, "I'm self-employed.", WRONG_PROFILE_URL)
    await _seed_research_fact(store, "I completed a program.", RIGHT_PROFILE_URL)

    matches = await find_fact_matches(
        store,
        creator_id=CREATOR,
        assistant_id=ASSISTANT,
        user_id=CREATOR,
        query="linkedin.com/in/evanwoods",
    )

    assert [(match.kind, match.key, match.matched_text) for match in matches] == [
        ("source_url", research_key, WRONG_PROFILE_URL)
    ]


@pytest.mark.asyncio
async def test_edit_replaces_the_wrong_url_and_keeps_the_fact(monkeypatch):
    store = _make_store()
    research_key = await _seed_research_fact(store, "I'm self-employed.", WRONG_PROFILE_URL)
    captured_payload: dict = {}
    monkeypatch.setattr(identity_tools, "interrupt", _accept_every_match(captured_payload))

    await edit_identity_fact.coroutine(
        inaccurate_information="linkedin.com/in/evanwoods",
        corrected_information="My LinkedIn profile is linkedin.com/in/evanfwoods.",
        correction_context="I was told my LinkedIn profile is linkedin.com/in/evanfwoods.",
        runtime=_FakeRuntime(store, latest_user_message=LINKEDIN_CORRECTION_MESSAGE),
    )

    preview = captured_payload["payload"]["matches"][0]
    assert preview["kind"] == "source_url"
    assert preview["recommended_action"] == "accept"
    assert preview["suggested_edit_fact_content"] == RIGHT_PROFILE_URL

    corrected_document = await _stored_document(store, IDENTITY_NAMESPACE, research_key)
    assert corrected_document.metadata["source_urls"] == [RIGHT_PROFILE_URL]
    assert corrected_document.metadata["fact"] == "I'm self-employed."
    assert organization_links_from_identity(
        assistant_description="", identity_documents=[corrected_document]
    ) == [RIGHT_PROFILE_URL]


@pytest.mark.asyncio
async def test_edit_rewrites_the_wrong_claim_inside_a_shared_biography(monkeypatch):
    store = _make_store()
    first_key = await _seed_biography_fact(
        store, "I am working on a startup, Afterlife Systems Incorporated."
    )
    second_key = await _seed_biography_fact(store, "I am looking for a partner.")
    captured_payload: dict = {}
    monkeypatch.setattr(identity_tools, "interrupt", _accept_every_match(captured_payload))
    corrected_sentence = (
        "Creating statistical proof of the authenticity of the responses using stylometric "
        "indicators, not rouge and BERT scores."
    )

    await edit_identity_fact.coroutine(
        inaccurate_information=(
            "Creating statistical proof of the authenticity of the responses using rouge and "
            "BERT scores."
        ),
        corrected_information=corrected_sentence,
        correction_context="I was told I do not use ROUGE or BERT scores.",
        runtime=_FakeRuntime(store, latest_user_message=ROUGE_CORRECTION_MESSAGE),
    )

    assert {match["kind"] for match in captured_payload["payload"]["matches"]} == {
        "context_sentence"
    }
    for key, fact in (
        (first_key, "I am working on a startup, Afterlife Systems Incorporated."),
        (second_key, "I am looking for a partner."),
    ):
        corrected_document = await _stored_document(store, IDENTITY_MEMORY_NAMESPACE, key)
        assert BIOGRAPHY_ROUGE_SENTENCE not in corrected_document.metadata["fact_context"]
        assert BIOGRAPHY_ROUGE_SENTENCE not in corrected_document.page_content
        assert corrected_sentence in corrected_document.metadata["fact_context"]
        assert corrected_document.metadata["fact"] == fact


@pytest.mark.asyncio
async def test_text_restating_the_url_correction_is_replaced_by_the_corrected_fact(monkeypatch):
    """Production row 4a0745f2 said "…evanfwoods, not linkedin.com/in/evanwoods."; the
    per-match model swapped the wrong side, so the text is replaced by the corrected fact."""
    store = _make_store()
    document_id = str(uuid.uuid4())
    restated_fact = "My LinkedIn site is linkedin.com/in/evanfwoods, not linkedin.com/in/evanwoods."
    restated_context = "Evan corrected the LinkedIn site to linkedin.com/in/evanfwoods rather than linkedin.com/in/evanwoods."
    document = Document(
        page_content=wrap_fact_with_context(restated_fact, restated_context),
        metadata={"fact": restated_fact, "fact_context": restated_context, "document_id": document_id},
    )
    await store.aput(IDENTITY_MEMORY_NAMESPACE, key=document_id, value={"document": document.to_json()})

    async def _swapping_suggestion(match, **kwargs):
        raise AssertionError("the per-match model must not be asked about a restated URL correction")

    monkeypatch.setattr(identity_tools, "_suggest_correction", _swapping_suggestion)
    monkeypatch.setattr(identity_tools, "interrupt", _accept_every_match({}))
    corrected_fact = "My LinkedIn profile is https://www.linkedin.com/in/evanfwoods."

    await edit_identity_fact.coroutine(
        inaccurate_information="linkedin.com/in/evanwoods",
        corrected_information=corrected_fact,
        correction_context="I was told my LinkedIn profile is linkedin.com/in/evanfwoods.",
        runtime=_FakeRuntime(store, latest_user_message=LINKEDIN_CORRECTION_MESSAGE),
    )

    corrected_document = await _stored_document(store, IDENTITY_MEMORY_NAMESPACE, document_id)
    assert corrected_document.metadata["fact"] == corrected_fact
    assert "in/evanwoods" not in corrected_document.page_content
    assert "in/evanwoods" not in corrected_document.metadata["fact_context"]


@pytest.mark.asyncio
async def test_wrong_claim_inside_a_scraped_html_quote_is_matched():
    """A scraped page stores list items with no whitespace after the period
    ("dataset.</li>\\n<li>"); the claim inside one list item must still match."""
    store = _make_store()
    quote_key = str(uuid.uuid4())
    html_page = (
        '<p dir="auto">I am currently working on:</p>\n<ul dir="auto">\n'
        "<li>Developing pipelines to pull data.</li>\n"
        f"<li>{BIOGRAPHY_ROUGE_SENTENCE}</li>\n<li>Creating API endpoints.</li>\n</ul>"
    )
    document = Document(page_content=html_page, metadata={"document_id": quote_key})
    await store.aput((CREATOR, ASSISTANT, "quote"), key=quote_key, value={"document": document.to_json()})

    matches = await find_fact_matches(
        store,
        creator_id=CREATOR,
        assistant_id=ASSISTANT,
        user_id=CREATOR,
        query=BIOGRAPHY_ROUGE_SENTENCE,
    )

    assert [(match.kind, match.matched_text) for match in matches] == [
        ("sentence", BIOGRAPHY_ROUGE_SENTENCE)
    ]
