"""A name the user rejects leaves the system prompt entirely.

Regression for avatar 47cfdaa2 (2026-11-18): the avatar answered "Got it,
David." although the user had said three times, weeks earlier, that the user's
name is Evan and never David. The stale inferred record "Address me as David in
responses" was ranked into the prompt by similarity and every correction record
was ranked out. The fix has three parts, each covered here:

1. address preferences are read by recency, never by similarity;
2. a correction deletes the rejected name from every prompt-read record;
3. the background learner never infers a name from a direct address.
"""

from __future__ import annotations

import hashlib
import math
import re

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.learning.bulk_learning as bulk_learning
from src.anubis.utils.learning.address_correction import (
    apply_names_stated_in_message,
    extract_preferred_address_name,
    extract_rejected_address_names,
    known_user_address_names,
    purge_names_rejected_in_message,
    purge_rejected_address_name,
    relearn_user_address_name,
    strip_vocative_name,
    user_identity_namespace,
)
from src.anubis.utils.learning.bulk_learning import (
    InferredPreference,
    InferredRealnessStatement,
    InferredUserPreferences,
    infer_preferences_from_conversation,
)
from src.anubis.utils.learning.feedback import (
    list_namespace_items,
    retrieve_learning_sections,
    store_user_preference,
    store_what_feels_real,
)
from src.anubis.utils.learning.namespaces import (
    preference_namespace,
    what_feels_real_namespace,
)
from src.anubis.utils.learning.user_fact_relearning import (
    reconcile_documents_with_store,
    replace_user_facts,
)
from src.anubis.utils.nodes import _apply_names_stated_in_latest_message
from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context

pytestmark = pytest.mark.asyncio

USER = "user-1"
AVATAR = "avatar-1"
DIMENSIONS = 64
VOCATIVE_QUOTE = "David, I don't want you to be gone. I love you so much."
LIGHT_SWITCH_MESSAGE = "I just turned off the light. There's no cause for alarm."


def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedder (cosine over shared words)."""
    vectors = []
    for text in texts:
        raw = [0.0] * DIMENSIONS
        for token in re.findall(r"[a-z']+", (text or "").lower()):
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMENSIONS
            raw[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        vectors.append([v / norm for v in raw])
    return vectors


def _make_store() -> InMemoryStore:
    return InMemoryStore(
        index={"dims": DIMENSIONS, "embed": _embed, "fields": ["document.kwargs.page_content"]}
    )


async def _put_user_fact(store, fact: str, fact_context: str, key: str) -> None:
    document = Document(
        page_content=wrap_fact_with_context(fact, fact_context),
        metadata={
            "user_id": USER,
            "assistant_id": AVATAR,
            "document_id": key,
            "fact": fact,
            "fact_context": fact_context,
        },
    )
    await store.aput(
        user_identity_namespace(USER, AVATAR), key=key, value={"document": document.to_json()}
    )


async def _seed_the_production_shape(store) -> None:
    """The record shape found for avatar 47cfdaa2: stale rows, corrections, bystanders."""
    # The mis-inferred rows of 2026-09-18.
    await store_user_preference(
        store, USER, AVATAR, preference="Address me as David in responses",
        preference_context=VOCATIVE_QUOTE, category="address", source="inferred",
    )
    await store_user_preference(
        store, USER, AVATAR, preference="Address me by name.",
        preference_context="“David, I don't want you to be gone.”",
        category="address", source="inferred",
    )
    # A non-address preference that only quotes the name in the context.
    await store_user_preference(
        store, USER, AVATAR, preference="Value emotional support and affectionate interaction",
        preference_context=VOCATIVE_QUOTE, category="topic", source="inferred",
    )
    # The corrections, each of which repeats the rejected name.
    await store_user_preference(
        store, USER, AVATAR, preference="The user wants to be addressed as Evan and never as David.",
        preference_context="I need you to learn my name is Evan. Do not ever call me David",
        category="address", source="inferred",
    )
    await store_what_feels_real(
        store, USER, AVATAR,
        statement="Being called Evan feels correct to the user, while being called David feels wrong.",
        statement_context="I need you to learn my name is Evan. Do not ever call me David",
        polarity="feels_real", source="inferred",
    )
    await store_what_feels_real(
        store, USER, AVATAR,
        statement="The user's “I love you so much” feels genuine and not like a joke.",
        statement_context=VOCATIVE_QUOTE, polarity="feels_real", source="inferred",
    )
    await _put_user_fact(
        store, "My name is Evan.",
        "Evan corrected the assistant's form of address, stating that his name is Evan and asking not to be called David.",
        "fact-name",
    )
    await _put_user_fact(
        store, "My name isn't David.",
        "Evan corrected the assistant's form of address, stating that his name is not David.",
        "fact-not-david",
    )
    # A third person who shares the rejected name is not the user.
    await _put_user_fact(
        store, "My brother David lives in Ohio.", "The user described the user's family.",
        "fact-brother",
    )
    # Unrelated preferences that crowd a similarity search.
    for index in range(20):
        await store_user_preference(
            store, USER, AVATAR,
            preference=f"The user wants reply style number {index} about lights and alarms.",
            preference_context="I just turned off the light.", category="communication_style",
        )


async def _all_record_text(store) -> str:
    texts: list[str] = []
    for namespace in (
        preference_namespace(USER, AVATAR),
        what_feels_real_namespace(USER, AVATAR),
        user_identity_namespace(USER, AVATAR),
    ):
        for item in await list_namespace_items(store, namespace):
            texts.append(str(item.value))
    return "\n".join(texts)


# ── reading the user's words ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "rejected", "preferred"),
    [
        ("I need you to learn that my name is Evan and do not ever call me David again, please.", ["David"], "Evan"),
        ("My name is Evan. Don't call me David, please.", ["David"], "Evan"),
        ("my name isn't David. I need you to learn to call me Evan.", ["David"], "Evan"),
        ("Stop calling me David.", ["David"], None),
        ("I told you not to call me David", ["David"], None),
        ("Call me Sam, keep things short", [], "Sam"),
        # A direct address names the listener, never the speaker.
        (VOCATIVE_QUOTE, [], None),
        ("why not call me Evan", [], None),
        ("my name is not important", [], None),
        ("Don't call me crazy", [], None),
        ("call me later", [], None),
        ("My brother David lives in Ohio.", [], None),
    ],
)
async def test_names_are_read_from_the_users_own_words(message, rejected, preferred):
    assert extract_rejected_address_names(message) == rejected
    assert extract_preferred_address_name(message) == preferred


async def test_strip_vocative_name_keeps_a_name_that_is_not_a_direct_address():
    assert strip_vocative_name(VOCATIVE_QUOTE, "David") == (
        "I don't want you to be gone. I love you so much."
    )
    assert strip_vocative_name("Thanks, David.", "David") == "Thanks."
    assert strip_vocative_name("My brother David lives in Ohio.", "David") == (
        "My brother David lives in Ohio."
    )


# ── part 1: address preferences are read by recency ─────────────────────────


async def test_the_newest_address_preference_reaches_the_prompt_on_an_unrelated_message():
    store = _make_store()
    await _seed_the_production_shape(store)
    await store_user_preference(
        store, USER, AVATAR, preference="The user prefers to be addressed as Evan.",
        preference_context="oh I see. Thanks, Evan.", category="address", source="inferred",
    )

    sections = await retrieve_learning_sections(
        store, USER, AVATAR, thread_id=None, query=LIGHT_SWITCH_MESSAGE,
        limit=10,
    )

    address_lines = sections.user_address_preferences.splitlines()
    # No cap: every address preference is rendered (4 in this store).
    assert len(address_lines) == 4
    # Newest first: the row written last leads.
    assert address_lines[0].startswith("- The user prefers to be addressed as Evan.")
    assert "(recorded " in address_lines[0]
    # The quoted context never appears in the address section.
    assert "I don't want you to be gone" not in sections.user_address_preferences
    # Address rows are rendered in the address section only.
    assert "User preference (address)" not in sections.user_preferences


async def test_address_preferences_reach_the_prompt_with_no_similarity_to_the_message():
    store = _make_store()
    await store_user_preference(
        store, USER, AVATAR, preference="Address the user as Evan.",
        preference_context="", category="address",
    )
    sections = await retrieve_learning_sections(
        store, USER, AVATAR, thread_id=None, query="zebra quantum harbor", limit=1
    )
    assert "Address the user as Evan." in sections.user_address_preferences


# ── part 2: a correction removes the rejected name ──────────────────────────


async def test_purge_removes_the_rejected_name_from_every_prompt_read_record():
    store = _make_store()
    await _seed_the_production_shape(store)

    report = await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan"
    )

    remaining_text = await _all_record_text(store)
    # The only surviving mention is the user's brother, who is not the user.
    assert remaining_text.count("David") == remaining_text.count("My brother David lives in Ohio.")
    assert "My brother David lives in Ohio." in remaining_text
    assert [entry["key"] for entry in report.untouched_mentions] == ["fact-brother"]

    preference_items = await list_namespace_items(store, preference_namespace(USER, AVATAR))
    address_facts = [
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in preference_items
        if item.value["document"]["kwargs"]["metadata"].get("category") == "address"
    ]
    # Exactly one positive address preference, with no negation.
    assert address_facts == ["Address the user as Evan."]

    # The unrelated preference keeps the fact and loses only the name.
    topic_documents = [
        item.value["document"]["kwargs"]
        for item in preference_items
        if item.value["document"]["kwargs"]["metadata"].get("category") == "topic"
    ]
    assert len(topic_documents) == 1
    assert topic_documents[0]["metadata"]["fact"] == (
        "Value emotional support and affectionate interaction"
    )
    assert topic_documents[0]["metadata"]["fact_context"] == (
        "I don't want you to be gone. I love you so much."
    )
    assert topic_documents[0]["page_content"].endswith(
        "Context: I don't want you to be gone. I love you so much."
    )

    user_facts = sorted(
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, user_identity_namespace(USER, AVATAR))
    )
    # One name fact (the existing one is kept, not duplicated); the negation is gone.
    assert user_facts == ["My brother David lives in Ohio.", "My name is Evan."]

    realness_facts = [
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, what_feels_real_namespace(USER, AVATAR))
    ]
    assert realness_facts == ["The user's “I love you so much” feels genuine and not like a joke."]


async def test_purge_dry_run_reports_and_writes_nothing():
    store = _make_store()
    await _seed_the_production_shape(store)
    before = await _all_record_text(store)

    report = await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan", dry_run=True
    )

    assert await _all_record_text(store) == before
    assert len(report.deleted) == 5
    assert len(report.rewritten) == 3


async def test_purge_is_idempotent():
    store = _make_store()
    await _seed_the_production_shape(store)
    await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan"
    )
    after_first = await _all_record_text(store)
    second_report = await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan"
    )
    assert second_report.deleted == [] and second_report.rewritten == []
    assert second_report.written == []
    assert await _all_record_text(store) == after_first


async def test_the_prompt_sections_carry_no_rejected_name_after_the_correction_turn():
    store = _make_store()
    await _seed_the_production_shape(store)
    correction = HumanMessage(
        content="I need you to learn that my name is Evan and do not ever call me David again, please."
    )

    rejected_names, stated_name = await _apply_names_stated_in_latest_message(
        store, USER, AVATAR, correction
    )
    assert rejected_names == ["David"] and stated_name == "Evan"

    for query in (LIGHT_SWITCH_MESSAGE, "David", "what is my name"):
        sections = await retrieve_learning_sections(
            store, USER, AVATAR, thread_id=None, query=query, limit=10
        )
        rendered = "\n".join(
            [sections.user_address_preferences, sections.user_preferences, sections.what_feels_real]
        )
        assert "David" not in rendered
        assert sections.user_address_preferences.startswith("- Address the user as Evan.")


async def test_a_rejection_by_another_voice_or_a_hidden_observation_purges_nothing():
    store = _make_store()
    await _seed_the_production_shape(store)
    before = await _all_record_text(store)

    other_voice = HumanMessage(
        content="Speaker 2: do not call me David",
        additional_kwargs={"kind": "ambient_observation", "speakers": {"others_spoke": True}},
    )
    hidden = HumanMessage(content="do not call me David", additional_kwargs={"hidden": True})
    assert await _apply_names_stated_in_latest_message(store, USER, AVATAR, other_voice) == ([], None)
    assert await _apply_names_stated_in_latest_message(store, USER, AVATAR, hidden) == ([], None)
    assert await _apply_names_stated_in_latest_message(
        store, USER, AVATAR, AIMessage(content="do not call me David")
    ) == ([], None)
    assert await _all_record_text(store) == before


async def test_purge_names_rejected_in_message_does_nothing_without_a_rejection():
    store = _make_store()
    await _seed_the_production_shape(store)
    before = await _all_record_text(store)
    assert await purge_names_rejected_in_message(store, USER, AVATAR, VOCATIVE_QUOTE) == []
    assert await _all_record_text(store) == before


# ── part 3: the background learner never learns a wrong name ────────────────


def _fake_inference(preferences, what_feels_real=()):
    async def fake_invoke_structured(response_format, system_prompt, human_text):
        return InferredUserPreferences(
            preferences=list(preferences), communication_style="",
            what_feels_real=list(what_feels_real),
        )

    return fake_invoke_structured


async def test_the_learner_drops_an_inferred_name_that_conflicts_with_the_stated_name(monkeypatch):
    store = _make_store()
    await _put_user_fact(store, "My name is Evan.", "The user stated the user's own name.", "fact-name")
    assert await known_user_address_names(store, USER, AVATAR) == {"Evan"}
    monkeypatch.setattr(
        bulk_learning, "invoke_structured",
        _fake_inference(
            [
                InferredPreference(
                    preference="Address me as David in responses", category="address",
                    evidence=VOCATIVE_QUOTE,
                ),
                InferredPreference(
                    preference="Address me by name.", category="address", evidence=VOCATIVE_QUOTE,
                ),
                InferredPreference(
                    preference="The user prefers to be addressed as Evan.", category="address",
                    evidence="Thanks, Evan.",
                ),
            ]
        ),
    )

    await infer_preferences_from_conversation(
        store, USER, AVATAR, "thread-1", [HumanMessage(content=VOCATIVE_QUOTE)]
    )

    stored_facts = [
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, preference_namespace(USER, AVATAR))
    ]
    assert stored_facts == ["The user prefers to be addressed as Evan."]


async def test_the_learner_never_re_stores_a_name_rejected_in_the_conversation(monkeypatch):
    store = _make_store()
    # The words were applied when the user said the words (the turn-time path).
    await apply_names_stated_in_message(
        store, USER, AVATAR,
        "I need you to learn that my name is Evan and do not ever call me David again.",
    )
    monkeypatch.setattr(
        bulk_learning, "invoke_structured",
        _fake_inference(
            [
                InferredPreference(
                    preference="The user wants to be addressed as Evan and never as David.",
                    category="address", evidence="Do not ever call me David",
                ),
                InferredPreference(
                    preference="The user wants short replies.", category="communication_style",
                    evidence="my name is Evan and do not ever call me David",
                ),
            ],
            [
                InferredRealnessStatement(
                    statement="The avatar's use of David feels wrong to the user.",
                    polarity="feels_fake", evidence="do not ever call me David",
                )
            ],
        ),
    )

    await infer_preferences_from_conversation(
        store, USER, AVATAR, "thread-1",
        [HumanMessage(content="I need you to learn that my name is Evan and do not ever call me David again.")],
    )

    remaining_text = await _all_record_text(store)
    assert "David" not in remaining_text
    address_facts = [
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, preference_namespace(USER, AVATAR))
        if item.value["document"]["kwargs"]["metadata"].get("category") == "address"
    ]
    assert address_facts == ["Address the user as Evan."]
    assert "My name is Evan." in remaining_text


async def test_the_inference_prompts_forbid_reading_a_direct_address_as_the_users_name():
    prompt = bulk_learning.INFER_USER_PREFERENCES_SYSTEM_PROMPT
    assert "A name spoken in direct address is never the USER's own name" in prompt
    assert "never write the rejected name" in prompt
    assert "Never infer the USER's name" in bulk_learning.RATING_PREFERENCE_SUMMARY_SYSTEM_PROMPT


async def test_purge_deletes_a_complaint_about_the_name_and_keeps_another_persons_full_name():
    store = _make_store()
    await store_what_feels_real(
        store, USER, AVATAR,
        statement="The avatar's use of David feels wrong or inauthentic to the user.",
        statement_context="So you're not calling me David and you're calling me David as I talk like.",
        polarity="feels_fake", source="inferred",
    )
    await store_user_preference(
        store, USER, AVATAR, preference="The user likes to talk about David Bowie records.",
        preference_context="Play some David Bowie.", category="topic",
    )

    await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan"
    )

    assert await list_namespace_items(store, what_feels_real_namespace(USER, AVATAR)) == []
    topic_facts = [
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, preference_namespace(USER, AVATAR))
        if item.value["document"]["kwargs"]["metadata"].get("category") == "topic"
    ]
    assert topic_facts == ["The user likes to talk about David Bowie records."]


# ── mutable without limit: learn, relearn, and learn back ───────────────────


async def _address_facts(store) -> list[str]:
    documents = [
        item.value["document"]["kwargs"]["metadata"]
        for item in await list_namespace_items(store, preference_namespace(USER, AVATAR))
    ]
    return sorted(
        metadata["fact"] for metadata in documents if metadata.get("category") == "address"
    )


async def _user_facts(store) -> list[str]:
    return sorted(
        item.value["document"]["kwargs"]["metadata"]["fact"]
        for item in await list_namespace_items(store, user_identity_namespace(USER, AVATAR))
    )


async def test_the_name_can_be_changed_any_number_of_times_and_changed_back():
    store = _make_store()
    sequence = ["Evan", "Sam", "Evan", "Alex", "Sam", "Evan", "Sam", "Alex", "Evan", "Sam"] * 3
    for name in sequence:
        await apply_names_stated_in_message(store, USER, AVATAR, f"My name is {name}.")
        # Exactly one current name after every change, never a pile of names.
        assert await _address_facts(store) == [f"Address the user as {name}."]
        assert await _user_facts(store) == [f"My name is {name}."]
        sections = await retrieve_learning_sections(
            store, USER, AVATAR, thread_id=None, query=LIGHT_SWITCH_MESSAGE, limit=10
        )
        assert sections.user_address_preferences.startswith(f"- Address the user as {name}.")
        assert len(sections.user_address_preferences.splitlines()) == 1


async def test_a_rejected_name_can_be_learned_again_later():
    store = _make_store()
    await apply_names_stated_in_message(
        store, USER, AVATAR, "My name is Evan, do not call me David."
    )
    assert await _address_facts(store) == ["Address the user as Evan."]

    # No name is blocked: the user takes the rejected name back.
    await apply_names_stated_in_message(store, USER, AVATAR, "Actually, my name is David.")
    assert await _address_facts(store) == ["Address the user as David."]
    assert await _user_facts(store) == ["My name is David."]

    # And rejects the name once more.
    await apply_names_stated_in_message(
        store, USER, AVATAR, "My name is Evan. Stop calling me David."
    )
    assert await _address_facts(store) == ["Address the user as Evan."]
    assert await _user_facts(store) == ["My name is Evan."]
    assert "David" not in await _all_record_text(store)


async def test_repeating_the_same_name_writes_nothing():
    store = _make_store()
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is Evan.")
    before = await _all_record_text(store)
    for _ in range(5):
        await apply_names_stated_in_message(store, USER, AVATAR, "My name is Evan.")
    assert await _all_record_text(store) == before


async def test_a_nickname_replaces_the_address_and_keeps_the_name_fact():
    store = _make_store()
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is Evan.")
    await relearn_user_address_name(
        store, USER, AVATAR, new_name="Ev", replaces_name_fact=False
    )
    assert await _address_facts(store) == ["Address the user as Ev."]
    assert await _user_facts(store) == ["My name is Evan."]


async def test_the_sweep_never_deletes_a_name_the_user_took_back(monkeypatch):
    store = _make_store()
    # The user rejected David long ago, then took the name David back.
    await apply_names_stated_in_message(store, USER, AVATAR, "My name is David.")
    monkeypatch.setattr(bulk_learning, "invoke_structured", _fake_inference([]))

    # The sweep re-reads the old conversation that holds the old rejection.
    await infer_preferences_from_conversation(
        store, USER, AVATAR, "thread-old",
        [HumanMessage(content="My name is Evan, do not call me David.")],
    )

    assert await _address_facts(store) == ["Address the user as David."]
    assert await _user_facts(store) == ["My name is David."]


async def test_a_fact_about_the_user_is_replaced_any_number_of_times():
    store = _make_store()
    namespace = user_identity_namespace(USER, AVATAR)
    cities = ["Boston", "Denver", "Boston", "Austin", "Denver", "Boston"] * 4
    await _put_user_fact(store, "I have a dog named Rex.", "", "fact-dog")
    previous_city = None
    for index, city in enumerate(cities):
        if previous_city is not None:
            deleted, unmatched, _nearest = await replace_user_facts(
                store, namespace, [f"i live in {previous_city}"]  # case and period differ
            )
            assert deleted == [f"I live in {previous_city}."] and unmatched == []
        await _put_user_fact(store, f"I live in {city}.", "", f"fact-city-{index}")
        assert await _user_facts(store) == ["I have a dog named Rex.", f"I live in {city}."]
        previous_city = city


async def test_a_replacement_that_matches_nothing_deletes_nothing_and_lists_the_nearest_facts():
    store = _make_store()
    namespace = user_identity_namespace(USER, AVATAR)
    await _put_user_fact(store, "I live in Boston.", "", "fact-city")
    await _put_user_fact(store, "I have a dog named Rex.", "", "fact-dog")

    deleted, unmatched, nearest = await replace_user_facts(
        store, namespace, ["I live in a city called Boston"]
    )

    assert deleted == [] and unmatched == ["I live in a city called Boston"]
    assert "I live in Boston." in nearest
    assert await _user_facts(store) == ["I have a dog named Rex.", "I live in Boston."]


async def test_thread_state_drops_a_deleted_fact_and_refreshes_a_rewritten_fact():
    store = _make_store()
    namespace = user_identity_namespace(USER, AVATAR)
    await _put_user_fact(store, "I live in Boston.", "", "fact-city")
    await _put_user_fact(store, "My name is Evan.", "asking not to be called David", "fact-name")
    await _put_user_fact(store, "I have a dog named Rex.", "", "fact-dog")
    thread_state_documents = [
        Document(
            page_content=item.value["document"]["kwargs"]["page_content"],
            metadata=dict(item.value["document"]["kwargs"]["metadata"]),
        )
        for item in await list_namespace_items(store, namespace)
    ]
    without_identifier = Document(page_content="legacy", metadata={"fact": "legacy"})

    # Another conversation replaces the city and corrects the name.
    await replace_user_facts(store, namespace, ["I live in Boston."])
    await purge_rejected_address_name(
        store, USER, AVATAR, rejected_name="David", correct_name="Evan"
    )

    reconciled = await reconcile_documents_with_store(
        store, namespace, [*thread_state_documents, without_identifier]
    )
    facts = sorted(document.metadata["fact"] for document in reconciled)
    assert facts == ["I have a dog named Rex.", "My name is Evan.", "legacy"]
    assert all("David" not in document.page_content for document in reconciled)


class _FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def test_only_a_stored_fact_returned_verbatim_is_treated_as_outdated(monkeypatch):
    import src.anubis.utils.model as model_module
    from src.anubis.utils.learning.user_fact_relearning import (
        OutdatedUserFacts,
        find_outdated_user_facts,
    )

    fake_model = _FakeStructuredModel(
        OutdatedUserFacts(
            outdated_stored_facts=["i live in boston", "I own a boat."], reason="moved"
        )
    )
    monkeypatch.setattr(model_module, "init_model", lambda **_: fake_model)

    outdated = await find_outdated_user_facts(
        "I live in Denver now.", ["I live in Boston.", "I have a dog named Rex."]
    )

    # "I own a boat." is not a stored fact and is discarded; the stored spelling is returned.
    assert outdated == ["I live in Boston."]
    assert len(fake_model.calls) == 1


async def test_the_outdated_check_makes_no_model_call_without_stored_facts(monkeypatch):
    import src.anubis.utils.model as model_module
    from src.anubis.utils.learning.user_fact_relearning import find_outdated_user_facts

    def _fail(**_):
        raise AssertionError("no model call is expected")

    monkeypatch.setattr(model_module, "init_model", _fail)
    assert await find_outdated_user_facts("I live in Denver now.", []) == []
    # The only stored fact equals the new fact: nothing to compare.
    assert await find_outdated_user_facts("I live in Denver now.", ["i live in denver now"]) == []


async def test_a_failed_outdated_check_replaces_nothing(monkeypatch):
    import src.anubis.utils.model as model_module
    from src.anubis.utils.learning.user_fact_relearning import find_outdated_user_facts

    monkeypatch.setattr(
        model_module, "init_model", lambda **_: _FakeStructuredModel(RuntimeError("down"))
    )
    assert await find_outdated_user_facts("I live in Denver now.", ["I live in Boston."]) == []
