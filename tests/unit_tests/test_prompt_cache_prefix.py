"""The system prompt's opening stretch must not move from one turn to the next.

OpenAI charges the longest identical opening stretch of a request — tool
definitions first, then the system prompt — at the cached rate, roughly a tenth
of the uncached rate, and answers a cached request sooner. The saving is
decided entirely by ORDER: one per-turn value early in the prompt pushes every
fixed token behind that value out of the cached stretch.

``IDENTITY_SYSTEM_PROMPT_TEMPLATE`` used to put the ``<ROLE>`` block — identity
facts retrieved by similarity, recalled memories, direct quotes, the avatar's
current emotion, the system time — in the MIDDLE, with a second copy of the
learn-information text and a repeated instruction block behind it. Those
trailing thousands of fixed tokens were re-billed in full on every message.
``_build_consciousness_system_message_update`` now assembles the prompt in
order of how often each part changes: fixed instructions, then the fixed
capability sections, then this turn's status blocks, then ``<ROLE>``.

These tests pin that order. A future edit that moves a per-turn value back
above the fixed text will not fail a behavioural test — the avatar answers
exactly as well either way — so the cost regression would otherwise be
invisible until the invoice arrives.
"""

from os.path import commonprefix
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

import src.anubis.utils.nodes as nodes
from src.anubis.utils.model import prompt_cache_key_for, prompt_cache_key_is_enabled
from src.anubis.utils.prompts.system_prompts import (
    IDENTITY_SYSTEM_PROMPT_TEMPLATE,
    ROLE_SECTION_OPENING_TAG,
)

CREATOR_ID = "creator-1"
ASSISTANT_ID = "assistant-1"
VISITOR_ID = "visitor-1"


class _EmptyStore:
    """A store that answers every read with nothing.

    The prompt's ORDER is what is under test, not what the store holds, and an
    empty store keeps the two renders below differing only in the ways the test
    makes them differ.
    """

    async def asearch(self, namespace, query=None, limit=None):
        return []

    async def aget(self, namespace, key):
        return None


async def _render_prompt(*, message: str, live_shares=None, user_id=VISITOR_ID):
    """Drive the real consciousness builder and return the rendered prompt."""
    assistant_ctx = {
        "name": "Grant Imahara",
        "metadata": {"user_id": CREATOR_ID},
    }
    state = {
        "messages": [HumanMessage(content=message)],
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
            "live_shares": live_shares,
        }
    }
    runtime = SimpleNamespace(
        store=_EmptyStore(),
        context=SimpleNamespace(assistant_ctx=assistant_ctx, user_ctx={}),
    )
    update = await nodes._build_consciousness_system_message_update(
        state, config, runtime
    )
    return update["system_message"][0].content


# --- The template's own order -------------------------------------------------


def test_the_role_block_is_the_end_of_the_template():
    """Nothing fixed may sit behind the per-turn ROLE block.

    The closing reminder after ROLE is deliberately short: the recency of a
    final instruction is worth keeping, a second full copy of every rule is
    not, and everything behind ROLE is billed at the uncached rate.
    """
    role_start = IDENTITY_SYSTEM_PROMPT_TEMPLATE.index(ROLE_SECTION_OPENING_TAG)
    behind_the_role_block = IDENTITY_SYSTEM_PROMPT_TEMPLATE[role_start:]
    after_the_role_block = behind_the_role_block.split("</ROLE>", 1)[1]
    assert len(after_the_role_block) < 1000, after_the_role_block[:2000]


def test_the_learn_information_text_is_rendered_once():
    """The template carried two full copies, the second one behind ROLE.

    That second copy is between 8,500 and 15,400 characters depending on
    whether the person talking is the avatar's creator, and every turn paid
    the uncached rate for it.
    """
    assert IDENTITY_SYSTEM_PROMPT_TEMPLATE.count("{learn_information_prompt_str}") == 1


def test_the_per_turn_values_are_all_inside_the_role_block():
    """Every value that changes turn to turn belongs behind the fixed text."""
    role_start = IDENTITY_SYSTEM_PROMPT_TEMPLATE.index(ROLE_SECTION_OPENING_TAG)
    fixed_half = IDENTITY_SYSTEM_PROMPT_TEMPLATE[:role_start]
    for placeholder in (
        "{retrieved_memories}",
        "{retrieved_knowledge}",
        "{direct_quotes}",
        "{assistant_emotions}",
        "{user_emotions}",
        "{current_conversation_sentiment}",
        "{system_time}",
        # The what-feels-real request flips on and off with the message count,
        # and used to sit in CONTINUOUS_LEARNING near the top, where the flip
        # invalidated every cached token behind it.
        "{what_feels_real_request}",
    ):
        assert placeholder not in fixed_half, placeholder


# --- The assembled prompt -----------------------------------------------------


@pytest.mark.asyncio
async def test_two_turns_share_the_whole_fixed_half_of_the_prompt():
    """Two different turns must agree, byte for byte, up to the first per-turn text."""
    first = await _render_prompt(message="Tell me about yourself.")
    second = await _render_prompt(
        message="What is on my screen?", live_shares='["screen"]'
    )

    shared_opening = commonprefix([first, second])
    # The whole template ahead of ROLE is inside the shared opening: the
    # instructions, the restrictions, the style guidance and the single copy
    # of the learn-information text.
    assert "</CONTINUOUS_LEARNING>" in shared_opening
    assert "<YOUR ORGANIZATION LINKS>" in shared_opening
    assert "<CLOSING_REMINDER>" not in shared_opening
    # And the shared opening is the bulk of the prompt rather than a scrap of
    # the first section, which is what a per-turn value near the top produces.
    assert len(shared_opening) > 0.5 * min(len(first), len(second))


@pytest.mark.asyncio
async def test_what_is_being_shared_lands_behind_the_fixed_half():
    """The LIVE_SHARES block is per-turn text and belongs after the fixed text."""
    prompt = await _render_prompt(
        message="What is on my screen?", live_shares='["screen"]'
    )
    assert "<LIVE_SHARES>" in prompt
    assert prompt.index("<YOUR ORGANIZATION LINKS>") < prompt.index("<LIVE_SHARES>")
    assert prompt.index("<LIVE_SHARES>") < prompt.index(ROLE_SECTION_OPENING_TAG)


@pytest.mark.asyncio
async def test_the_role_block_ends_the_assembled_prompt():
    """ROLE is re-attached last, after every capability and status section."""
    prompt = await _render_prompt(message="Tell me about yourself.")
    assert ROLE_SECTION_OPENING_TAG in prompt
    assert "=== RETRIEVED MEMORIES ===" in prompt
    assert prompt.index(ROLE_SECTION_OPENING_TAG) > 0.5 * len(prompt)


# --- Routing one avatar's requests to one cache -------------------------------


def test_the_cache_key_separates_avatars_and_audiences():
    """The creator and the public are given different text early in the prompt.

    The learn-information block differs between the two, high enough up that
    the two prefixes diverge almost immediately, so the two audiences have no
    cached stretch to share and are kept in separate caches.
    """
    creator = prompt_cache_key_for(ASSISTANT_ID, True)
    public = prompt_cache_key_for(ASSISTANT_ID, False)
    assert creator != public
    assert prompt_cache_key_for("other-avatar", True) != creator
    # A missing avatar id still yields a usable key rather than raising.
    assert prompt_cache_key_for(None, False)


def test_the_cache_key_is_on_unless_a_deployment_turns_it_off():
    """Unset means on: the saving should not wait on an env edit."""
    assert prompt_cache_key_is_enabled(SimpleNamespace(prompt_cache_key_enabled=None))
    assert prompt_cache_key_is_enabled(SimpleNamespace(prompt_cache_key_enabled=""))
    assert prompt_cache_key_is_enabled(
        SimpleNamespace(prompt_cache_key_enabled="TRUE")
    )
    assert not prompt_cache_key_is_enabled(
        SimpleNamespace(prompt_cache_key_enabled="FALSE")
    )


# --- Trimming an over-long prompt ---------------------------------------------


def test_an_over_long_prompt_is_trimmed_at_the_fixed_half_not_the_role_block():
    """The trim must fall on the instructions, never on the identity facts.

    ``DynamicConsciousnessPrompt`` trims the system prompt when the prompt
    outgrows the model's window. The trim cut from the END, which was harmless
    while ROLE sat in the middle and only repeated instructions followed it.
    With ROLE moved last, trimming from the end would cut exactly the identity
    facts, recalled memories and quotes the reply is built from and keep the
    boilerplate — so the trim now falls on the fixed half instead.
    """
    from src.anubis.utils.middleware.dynamic_consciousness_prompt import (
        DynamicConsciousnessPrompt,
    )

    fixed_half = "FIXED INSTRUCTION LINE.\n" * 4000
    role_section = (
        ROLE_SECTION_OPENING_TAG
        + "=== YOUR NAME ===\nGrant Imahara\n\n"
        + "=== RETRIEVED MEMORIES ===\nthe memory that answers the question\n"
        + "</ROLE>\n"
    )
    trimmed = DynamicConsciousnessPrompt._truncate_preserving_role(
        fixed_half + role_section, 4096
    )

    assert "Grant Imahara" in trimmed
    assert "the memory that answers the question" in trimmed
    assert trimmed.endswith("</ROLE>\n")
    assert len(trimmed) < len(fixed_half + role_section)


def test_a_prompt_with_no_role_block_is_still_trimmed():
    """A template edit that drops the tag must not stop the trim working."""
    from src.anubis.utils.middleware.dynamic_consciousness_prompt import (
        DynamicConsciousnessPrompt,
    )

    text = "LINE.\n" * 8000
    trimmed = DynamicConsciousnessPrompt._truncate_preserving_role(text, 1024)
    assert len(trimmed) < len(text)


# --- Recording the cache hit in the ledger ------------------------------------


class _RecordingCursor:
    def __init__(self, executed: list):
        self.executed = executed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_details):
        return False

    async def execute(self, statement, parameters=None):
        self.executed.append((statement, parameters))


class _RecordingConnection:
    def __init__(self, executed: list):
        self.executed = executed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_details):
        return False

    def cursor(self):
        return _RecordingCursor(self.executed)


class _RecordingPool:
    """A database pool that records every statement instead of running it."""

    def __init__(self):
        self.executed: list = []

    def connection(self):
        return _RecordingConnection(self.executed)


@pytest.mark.asyncio
async def test_the_ledger_row_carries_the_cached_and_written_token_counts():
    """The cache hit rate has to be readable from ``api_metrics``.

    Before these columns the cached counts were read from the provider, priced
    into ``cost_usd`` and thrown away, so whether a prompt change moved the hit
    rate could only be found out by calling the provider directly.
    """
    from src.anubis.utils.billing.metering import persist_api_metrics_row

    pool = _RecordingPool()
    written = await persist_api_metrics_row(
        pool,
        inference_type="message",
        prompt_tokens=8328,
        completion_tokens=40,
        total_tokens=8368,
        cached_prompt_tokens=8064,
        cache_write_tokens=0,
    )
    assert written is True
    statement, parameters = pool.executed[-1]
    assert "cached_prompt_tokens" in statement and "cache_write_tokens" in statement
    assert parameters[-2:] == (8064, 0)
    # One placeholder per column, so the insert cannot silently misalign.
    assert statement.count("%s") == len(parameters)


@pytest.mark.asyncio
async def test_an_existing_ledger_table_is_given_the_two_columns():
    """A deployment whose table predates the columns gets them added on boot."""
    from src.anubis.utils.billing.metering import ensure_api_metrics_table

    pool = _RecordingPool()
    await ensure_api_metrics_table(pool)
    statements = " ".join(statement for statement, _parameters in pool.executed)
    assert "ADD COLUMN IF NOT EXISTS cached_prompt_tokens" in statements
    assert "ADD COLUMN IF NOT EXISTS cache_write_tokens" in statements


def test_a_reply_served_from_the_cache_is_priced_at_the_cached_rate():
    """The reply's recorded cost must match what the provider billed.

    Every prompt token used to be priced at ``MODEL_PROMPT_COST``, so a reply
    whose prompt was 97% cached was recorded at roughly three and a half times
    its real cost, and the spend ledger could never show a cache paying off.
    """
    from langchain_core.messages import AIMessage

    from src.anubis.graph import _attach_token_usage_metadata

    planning_call = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 8328,
            "output_tokens": 40,
            "total_tokens": 8368,
            "input_token_details": {"cache_read": 8064},
        },
    )
    reply = AIMessage(content="I build things.")
    context = SimpleNamespace(
        model="gpt-5.6-luna",
        image_model=None,
        classification_model=None,
        llama_model=None,
        model_prompt_cost=0.0000002,
        model_completion_cost=0.0000008,
        model_cached_prompt_cost=0.00000002,
        model_cache_write_cost=0.0,
    )
    _attach_token_usage_metadata(reply, [planning_call], context)

    token_usage = reply.response_metadata["token_usage"]
    assert token_usage["prompt_tokens"] == 8328
    assert token_usage["cached_prompt_tokens"] == 8064
    assert token_usage["cache_write_tokens"] == 0
    expected_cost = (
        (8328 - 8064) * 0.0000002 + 8064 * 0.00000002 + 40 * 0.0000008
    )
    assert reply.response_metadata["total_cost"] == pytest.approx(expected_cost)


# --- Two messages: the fixed text and the per-turn text -----------------------
#
# gpt-5.6-luna keeps a prompt cache only at message boundaries. Measured on
# 2026-09-20: with the per-turn text inside the one system message, six turns of
# one conversation each WROTE ~8.3k tokens into the cache (billed above the
# plain input rate) and read back none; with the fixed text as its own system
# message, turns three onward read 7,946 tokens back and wrote about 400.


def _middleware_request(system_message, messages):
    from langchain.agents.middleware.types import ModelRequest

    return ModelRequest(
        model=None,
        messages=messages,
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"system_message": [system_message]},
        runtime=None,
    )


def _use_provider(monkeypatch, model_provider):
    import src.anubis.utils.middleware.dynamic_consciousness_prompt as middleware

    monkeypatch.setattr(
        middleware,
        "GlobalContext",
        lambda: SimpleNamespace(model_provider=model_provider, model_token_limit=400000),
    )
    return middleware.DynamicConsciousnessPrompt()


def _consciousness_message(fixed_text, per_turn_text):
    from langchain_core.messages import SystemMessage

    from src.anubis.utils.prompts.system_prompts import PER_TURN_SECTION_START_KEY

    return SystemMessage(
        content=fixed_text + per_turn_text,
        id="00000000-0000-0000-0000-0000000000000",
        additional_kwargs={PER_TURN_SECTION_START_KEY: len(fixed_text)},
    )


def test_the_fixed_text_and_the_per_turn_text_are_sent_as_two_messages(monkeypatch):
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

    middleware = _use_provider(monkeypatch, "OPEN_AI")
    fixed_text = "FIXED INSTRUCTIONS\n"
    per_turn_text = ROLE_SECTION_OPENING_TAG + "=== YOUR NAME ===\nGrant\n</ROLE>\n"
    earlier_question = HumanMessage(content="Tell me about yourself.")
    earlier_reply = AIMessage(content="I build things.")
    newest_question = HumanMessage(content="What did you build?")
    tool_round = AIMessage(
        content="", tool_calls=[{"name": "recall", "args": {}, "id": "call-1"}]
    )
    tool_answer = ToolMessage(content="a robot", tool_call_id="call-1")

    request = middleware._apply(
        _middleware_request(
            _consciousness_message(fixed_text, per_turn_text),
            [earlier_question, earlier_reply, newest_question, tool_round, tool_answer],
        )
    )

    assert request.system_message.content == fixed_text
    assert request.system_message.additional_kwargs == {}
    inserted = request.messages[2]
    assert isinstance(inserted, SystemMessage)
    assert inserted.content == per_turn_text
    # The per-turn text sits immediately before the newest human words, and
    # the conversation before it and the tool round after it are untouched.
    assert request.messages == [
        earlier_question, earlier_reply, inserted, newest_question, tool_round, tool_answer,
    ]


def test_a_provider_that_takes_one_system_message_gets_one(monkeypatch):
    """The Llama chat templates behind META and TOGETHER expect one system message."""
    middleware = _use_provider(monkeypatch, "TOGETHER")
    question = HumanMessage(content="Tell me about yourself.")
    request = middleware._apply(
        _middleware_request(
            _consciousness_message("FIXED\n", ROLE_SECTION_OPENING_TAG + "ROLE\n"),
            [question],
        )
    )
    assert request.system_message.content == "FIXED\n" + ROLE_SECTION_OPENING_TAG + "ROLE\n"
    assert request.messages == [question]


def test_a_prompt_without_a_recorded_offset_is_sent_whole(monkeypatch):
    from langchain_core.messages import SystemMessage

    middleware = _use_provider(monkeypatch, "OPEN_AI")
    question = HumanMessage(content="Tell me about yourself.")
    request = middleware._apply(
        _middleware_request(SystemMessage(content="WHOLE PROMPT"), [question])
    )
    assert request.system_message.content == "WHOLE PROMPT"
    assert request.messages == [question]


@pytest.mark.asyncio
async def test_the_consciousness_message_records_where_the_per_turn_text_begins():
    """The offset lands exactly where this turn's status blocks and ROLE begin."""
    from langchain_core.messages import SystemMessage

    from src.anubis.utils.prompts.system_prompts import PER_TURN_SECTION_START_KEY

    first = await _render_prompt(message="What is on my screen?", live_shares='["screen"]')
    assert "<LIVE_SHARES>" in first

    state = {
        "messages": [HumanMessage(content="What is on my screen?")],
        "user_state": {"user_id": VISITOR_ID},
        "assistant_state": {"assistant_id": ASSISTANT_ID},
    }
    assistant_ctx = {"name": "Grant Imahara", "metadata": {"user_id": CREATOR_ID}}
    update = await nodes._build_consciousness_system_message_update(
        state,
        {
            "configurable": {
                "user_id": VISITOR_ID,
                "assistant_id": ASSISTANT_ID,
                "assistant_ctx": assistant_ctx,
                "user_ctx": {},
                "thread_id": "thread-1",
                "live_shares": '["screen"]',
            }
        },
        SimpleNamespace(
            store=_EmptyStore(),
            context=SimpleNamespace(assistant_ctx=assistant_ctx, user_ctx={}),
        ),
    )
    message = update["system_message"][0]
    assert isinstance(message, SystemMessage)
    offset = message.additional_kwargs[PER_TURN_SECTION_START_KEY]
    fixed_text, per_turn_text = message.content[:offset], message.content[offset:]
    # Everything that changes with the turn is on the per-turn side...
    assert "<LIVE_SHARES>" in per_turn_text
    assert ROLE_SECTION_OPENING_TAG in per_turn_text
    assert "System Time:" in per_turn_text
    # ...and nothing that changes with the turn is on the fixed side.
    assert "<LIVE_SHARES>" not in fixed_text
    assert "System Time:" not in fixed_text
    assert "</CONTINUOUS_LEARNING>" in fixed_text


@pytest.mark.asyncio
async def test_two_turns_send_byte_identical_fixed_system_messages():
    """The fixed system message must not move at all between turns.

    On gpt-5.6-luna a one-character difference anywhere in the system message
    discards the whole system message's cache.
    """
    from src.anubis.utils.prompts.system_prompts import PER_TURN_SECTION_START_KEY

    fixed_texts = []
    for message, live_shares in (
        ("Tell me about yourself.", None),
        ("What is on my screen?", '["screen"]'),
    ):
        assistant_ctx = {"name": "Grant Imahara", "metadata": {"user_id": CREATOR_ID}}
        update = await nodes._build_consciousness_system_message_update(
            {
                "messages": [HumanMessage(content=message)],
                "user_state": {"user_id": VISITOR_ID},
                "assistant_state": {"assistant_id": ASSISTANT_ID},
            },
            {
                "configurable": {
                    "user_id": VISITOR_ID,
                    "assistant_id": ASSISTANT_ID,
                    "assistant_ctx": assistant_ctx,
                    "user_ctx": {},
                    "thread_id": "thread-1",
                    "live_shares": live_shares,
                }
            },
            SimpleNamespace(
                store=_EmptyStore(),
                context=SimpleNamespace(assistant_ctx=assistant_ctx, user_ctx={}),
            ),
        )
        system_message = update["system_message"][0]
        offset = system_message.additional_kwargs[PER_TURN_SECTION_START_KEY]
        fixed_texts.append(system_message.content[:offset])
    assert fixed_texts[0] == fixed_texts[1]
