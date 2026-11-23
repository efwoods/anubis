"""The context stored beside a fact learned in conversation is the user's own message.

The context used to be a summary written by the model that extracted the fact. The owner of
the avatar "Mom" wrote "we went to get dirt at the old house. You asked me to come with you."
and the summary stored on every fact read "I asked the user to come with me to the old
house", which the avatar then recounted (avatar 681fe8da, 2026-11-18). The old house was only
where the dirt was. These tests pin the replacement: the stored ``fact_context`` is the
user's most recent message, word for word, whatever ``fact_context`` argument the model sends.
"""

import pytest
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.learning.user_fact_relearning as user_fact_relearning
import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    learn_information_about_the_user,
    update_self_identity_mem_from_user_txt,
    verbatim_fact_context,
)

CREATOR = "creator-1"
ASSISTANT = "asst-1"
DIRT_MESSAGE = (
    "There was this one time we went to get dirt at the old house. You asked me to come "
    "with you. We went to the forest that was next to the house."
)
MODEL_WRITTEN_SUMMARY = (
    "I was told that I asked the user to come with me to the old house, where we went into "
    "the neighboring forest."
)


def test_avatar_fact_context_quotes_the_message_word_for_word():
    assert (
        verbatim_fact_context(DIRT_MESSAGE, told_to_the_avatar=True)
        == f'The user told me: "{DIRT_MESSAGE}"'
    )


def test_user_fact_context_quotes_the_message_word_for_word():
    assert (
        verbatim_fact_context(DIRT_MESSAGE, told_to_the_avatar=False)
        == f'The user said: "{DIRT_MESSAGE}"'
    )


def test_line_breaks_of_the_message_are_kept():
    message = "You played French horn.\n\nYou taught me to write."
    assert message in verbatim_fact_context(message, told_to_the_avatar=True)


def test_an_empty_message_gives_an_empty_context():
    assert verbatim_fact_context("   ", told_to_the_avatar=True) == ""


def test_a_sentence_that_carries_a_rejected_name_is_left_out():
    fact_context = verbatim_fact_context(
        "My name is Evan. Do not call me David. I live in Denver.",
        told_to_the_avatar=False,
        rejected_values=["David"],
    )
    assert "David" not in fact_context
    assert fact_context == 'The user said: "My name is Evan. I live in Denver."'


def test_a_rejected_name_inside_a_longer_word_does_not_remove_the_sentence():
    message = "I visited Davidson College."
    assert message in verbatim_fact_context(
        message, told_to_the_avatar=False, rejected_values=["David"]
    )


class _FakeRuntime:
    def __init__(self, store, message_text):
        self.store = store
        self.tool_call_id = "tc-1"
        self.state = {
            "messages": [HumanMessage(content=message_text)],
            "assistant_identity_documents": [],
            "user_identity_documents": [],
        }
        self.config = {
            "configurable": {
                "user_id": CREATOR,
                "assistant_id": ASSISTANT,
                "assistant_ctx": {"metadata": {"user_id": CREATOR}, "name": "Mom"},
            }
        }


@pytest.fixture
def _patched_models(monkeypatch):
    async def _fake_judge(*args, **kwargs):
        return identity_tools.FactGroundingVerdict(user_message_states_fact=True)

    async def _fake_extract(config):
        return {"user_id": CREATOR}, {"assistant_id": ASSISTANT}

    async def _fake_score_sentences(query, sentences):
        return [0.0 for _ in sentences]

    async def _fake_find_outdated_user_facts(user_fact, stored_facts):
        return []

    monkeypatch.setattr(identity_tools, "_judge_fact_against_user_message", _fake_judge)
    monkeypatch.setattr(identity_tools, "extract_user_id_assistant_id", _fake_extract)
    monkeypatch.setattr(identity_tools, "_score_sentences", _fake_score_sentences)
    monkeypatch.setattr(
        user_fact_relearning, "find_outdated_user_facts", _fake_find_outdated_user_facts
    )


async def _only_stored_document(store, namespace):
    items = await store.asearch(namespace)
    assert len(items) == 1
    return items[0].value["document"]["kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_sent_fact_context", [MODEL_WRITTEN_SUMMARY, ""])
async def test_avatar_fact_is_stored_with_the_users_own_message(
    _patched_models, model_sent_fact_context
):
    store = InMemoryStore()
    await update_self_identity_mem_from_user_txt.coroutine(
        fact_shared_about_the_assistant_from_the_user="I asked the user to come with me.",
        fact_context=model_sent_fact_context,
        runtime=_FakeRuntime(store, DIRT_MESSAGE),
    )
    stored = await _only_stored_document(store, (CREATOR, ASSISTANT, "identity_memory"))
    expected_fact_context = f'The user told me: "{DIRT_MESSAGE}"'
    assert stored["metadata"]["fact_context"] == expected_fact_context
    assert stored["metadata"]["fact"] == "I asked the user to come with me."
    assert f"<FACT_CONTEXT>{expected_fact_context}</FACT_CONTEXT>" in stored["page_content"]
    assert "come with me to the old house" not in stored["page_content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_sent_fact_context", [MODEL_WRITTEN_SUMMARY, ""])
async def test_user_fact_is_stored_with_the_users_own_message(
    _patched_models, model_sent_fact_context
):
    store = InMemoryStore()
    await learn_information_about_the_user.coroutine(
        user_fact="I did not see the snake next to me.",
        fact_context=model_sent_fact_context,
        runtime=_FakeRuntime(store, DIRT_MESSAGE),
    )
    stored = await _only_stored_document(store, (ASSISTANT, CREATOR, "identity"))
    expected_fact_context = f'The user said: "{DIRT_MESSAGE}"'
    assert stored["metadata"]["fact_context"] == expected_fact_context
    assert f"<FACT_CONTEXT>{expected_fact_context}</FACT_CONTEXT>" in stored["page_content"]
    assert "Mom asked him to come" not in stored["page_content"]


def test_system_prompt_explains_how_to_read_a_quoted_message():
    from src.anubis.utils.prompts.system_prompts import IDENTITY_SYSTEM_PROMPT_TEMPLATE

    assert "How to recount a memory the user told you" in IDENTITY_SYSTEM_PROMPT_TEMPLATE
