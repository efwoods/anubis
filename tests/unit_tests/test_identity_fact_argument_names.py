"""The fact argument survives a model that mistypes its name.

``update_self_identity_mem_from_user_txt`` takes a 44-character argument whose
name stutters — ``fact_shared_about_the_assistant_from_the_user``. On
2026-09-07 a model dropped the second "the", Pydantic refused the call, and the
fact the user had asked the avatar to learn ("my favorite family vacation spot
was Yosemite") was lost: the avatar answered about the rest of the message as
though nothing had been asked of it. The near-miss spellings are accepted so a
fumbled argument name never costs someone a fact.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.tools.identity.identity_tools import (
    AssistantFactAndContext,
    update_self_identity_mem_from_user_txt,
)

FACT = "My favorite family vacation spot was Yosemite."
CONTEXT = "The user cited a Facebook remembrance post by the avatar's mother."


def test_the_model_is_still_shown_the_canonical_argument_name():
    """Accepting misspellings must not teach the model a different name."""
    schema = update_self_identity_mem_from_user_txt.args_schema.model_json_schema()
    assert list(schema["properties"]) == [
        "fact_shared_about_the_assistant_from_the_user",
        "fact_context",
    ]
    assert schema["required"] == [
        "fact_shared_about_the_assistant_from_the_user",
        "fact_context",
    ]


@pytest.mark.parametrize(
    "argument_name",
    [
        "fact_shared_about_the_assistant_from_the_user",
        # The exact spelling that lost the Yosemite fact.
        "fact_shared_about_the_assistant_from_user",
        "fact_shared_about_assistant_from_the_user",
        "fact_shared_about_the_assistant",
    ],
)
def test_a_near_miss_argument_name_still_carries_the_fact(argument_name):
    parsed = AssistantFactAndContext.model_validate(
        {argument_name: FACT, "fact_context": CONTEXT}
    )
    assert parsed.fact_shared_about_the_assistant_from_the_user == FACT
    assert parsed.fact_context == CONTEXT
    # The tool body reads the canonical name, so that is what must be handed on.
    assert set(parsed.model_dump()) == {
        "fact_shared_about_the_assistant_from_the_user",
        "fact_context",
    }


def test_a_missing_fact_is_still_refused():
    """Tolerating a misspelling must not tolerate an absent fact."""
    with pytest.raises(Exception):
        AssistantFactAndContext.model_validate({"fact_context": CONTEXT})
