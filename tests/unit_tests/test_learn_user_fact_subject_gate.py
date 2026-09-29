"""A remark about the world must never be stored as a fact about the user.

Regression: the user commented "Terafab should complete in 2029" and the
avatar stored the comment under the user's identity, showing the chip
"Learned about you · Terafab should complete in 2029".
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils.learning.fact_learned import parse_learned_fact_from_tool_content
from src.anubis.utils.tools.identity.identity_tools import (
    fact_names_the_user,
    learn_information_about_the_user,
)


@pytest.mark.parametrize(
    "user_fact",
    [
        "My name is Evan.",
        "I have brown hair.",
        "I’m a fan of Critical Role.",
        "User is a man.",
        "The user's sister lives in Ottawa.",
        "We moved to Texas in 2019.",
        # The inference a remark supports stays learnable.
        "User follows the Terafab construction timeline.",
        "User is interested in how robotics affects universal basic income.",
    ],
)
def test_fact_with_the_user_as_subject_is_accepted(user_fact: str) -> None:
    assert fact_names_the_user(user_fact)


@pytest.mark.parametrize(
    "user_fact",
    [
        "Terafab should complete in 2029",
        "Universal basic income depends on productivity.",
        "The new model ships next quarter.",
        "",
    ],
)
def test_remark_about_the_world_is_refused(user_fact: str) -> None:
    assert not fact_names_the_user(user_fact)


class _StoreThatMustNotBeTouched:
    async def asearch(self, *arguments, **keyword_arguments):
        raise AssertionError("a refused fact must not reach the store")

    async def aput(self, *arguments, **keyword_arguments):
        raise AssertionError("a refused fact must not reach the store")


def test_tool_refuses_remark_before_touching_the_store() -> None:
    runtime = SimpleNamespace(
        tool_call_id="call-terafab",
        store=_StoreThatMustNotBeTouched(),
        state={"user_identity_documents": []},
        config={},
    )
    command = asyncio.run(
        learn_information_about_the_user.coroutine(
            user_fact="Terafab should complete in 2029",
            fact_context="The user commented on the Terafab timeline.",
            runtime=runtime,
        )
    )
    tool_message = command.update["messages"][0]
    assert tool_message.content.startswith("Not learned:")
    assert "user_identity_documents" not in command.update
    # The refusal must never become a "Learned about you" chip.
    assert (
        parse_learned_fact_from_tool_content(
            tool_message.content, tool_name="learn_information_about_the_user"
        )
        is None
    )
