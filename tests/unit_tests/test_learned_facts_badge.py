"""Learned-fact badges: parse kinds, attach only facts announced this turn.

The badge on a reply must name whose fact was stored (user vs avatar) and must
survive a reload. ``learned_facts`` on the reply's ``response_metadata`` comes
from ``announce_fact_learned`` (a successful new store), not from scanning
older learn ToolMessages that may still sit in the deep agent's messages.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import src.anubis.graph as graph_mod
from src.anubis.utils.learning.fact_learned import (
    TurnLearnedFactsCollector,
    announce_fact_learned,
    attach_learned_facts_metadata,
    parse_learned_fact_from_tool_content,
)


def test_parse_learned_about_the_user_returns_user_kind():
    parsed = parse_learned_fact_from_tool_content(
        "Learned about the user: I am very lucky."
    )
    assert parsed == {
        "fact": "I am very lucky.",
        "kind": "user",
        "source": "conversation",
    }


def test_parse_plain_learned_still_returns_identity():
    parsed = parse_learned_fact_from_tool_content(
        "Learned: I was born in Chicago.",
        tool_name="update_self_identity_mem_from_user_txt",
    )
    assert parsed == {
        "fact": "I was born in Chicago.",
        "kind": "identity",
        "source": "conversation",
    }


def test_parse_previously_learned_returns_none():
    assert (
        parse_learned_fact_from_tool_content(
            "Fact: I am very lucky. previously learned"
        )
        is None
    )
    assert (
        parse_learned_fact_from_tool_content(
            "Preference previously learned: short replies"
        )
        is None
    )


def test_attach_uses_announced_facts_not_stale_tool_messages():
    """Older learn ToolMessages in ``new_messages`` must not reappear on the badge."""
    TurnLearnedFactsCollector.begin_turn()
    try:
        announce_fact_learned("I am very lucky.", kind="user")
        final_message = AIMessage(content="That is wonderful.")
        new_messages = [
            ToolMessage(
                content="Learned about the user: I already knew this.",
                tool_call_id="old-1",
            ),
            ToolMessage(
                content="Learned about the user: I am very lucky.",
                tool_call_id="call-1",
            ),
            final_message,
        ]

        attach_learned_facts_metadata(final_message, new_messages)

        assert final_message.response_metadata["learned_facts"] == [
            {
                "fact": "I am very lucky.",
                "kind": "user",
                "source": "conversation",
            }
        ]
    finally:
        TurnLearnedFactsCollector.end_turn()


def test_attach_with_active_collector_skips_badge_when_nothing_announced():
    """A previously-learned ToolMessage alone must not paint the badge."""
    TurnLearnedFactsCollector.begin_turn()
    try:
        final_message = AIMessage(content="I already knew that.")
        new_messages = [
            ToolMessage(
                content="Fact: I am very lucky. previously learned",
                tool_call_id="call-1",
            ),
            final_message,
        ]

        attach_learned_facts_metadata(final_message, new_messages)

        assert "learned_facts" not in (final_message.response_metadata or {})
    finally:
        TurnLearnedFactsCollector.end_turn()


@pytest.mark.asyncio
async def test_attach_post_reply_analysis_uses_announced_facts(monkeypatch):
    """Outer ``state["messages"]`` lack the learn ToolMessage; announce supplies it."""
    monkeypatch.setattr(graph_mod, "_attach_go_emotions_metadata", lambda message: None)
    monkeypatch.setattr(
        graph_mod,
        "_attach_token_usage_metadata",
        lambda final_message, new_messages, context=None: None,
    )
    monkeypatch.setattr(
        "src.anubis.utils.ambient.playful_reactions.apply_playful_reaction_sentiment",
        lambda *args, **kwargs: None,
    )

    TurnLearnedFactsCollector.begin_turn()
    try:
        announce_fact_learned("I am very lucky.", kind="user")
        final_message = AIMessage(content="That is wonderful.")
        outer_messages = [HumanMessage(content="I am very lucky.")]
        new_messages = [
            ToolMessage(
                content="Learned about the user: I am very lucky.",
                tool_call_id="call-1",
            ),
            final_message,
        ]
        state = {
            "messages": outer_messages,
            "user_state": {"user_id": "user-1"},
            "assistant_state": {"assistant_id": "avatar-1"},
        }
        config = {"configurable": {}}
        runtime = SimpleNamespace(context=SimpleNamespace())

        await graph_mod._attach_post_reply_analysis(
            final_message,
            new_messages=new_messages,
            state=state,
            config=config,
            runtime=runtime,
        )

        assert final_message.response_metadata["learned_facts"] == [
            {
                "fact": "I am very lucky.",
                "kind": "user",
                "source": "conversation",
            }
        ]
    finally:
        TurnLearnedFactsCollector.end_turn()
