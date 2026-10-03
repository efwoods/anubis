"""Regression: a memory about the avatar must not be filed as a fact about the user.

On thread 618f0fd0 (2026-09-29) the user told a father avatar "You played French
Horn ... You taught me to write ...". The deep agent sent every fact to
``learn_information_about_the_user``, which stored the facts verbatim in the
second person under the user, so the avatar never learned "I played French
horn" and recounted the story from the user's point of view.
``fact_is_about_the_avatar`` lets the user tool refuse such a fact and name the
avatar tool instead.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.tools.identity.identity_tools import fact_is_about_the_avatar

FACTS_ABOUT_THE_AVATAR = [
    "You played French Horn.",
    "You would listen to me ramble about my hypothesis of how everything big and small was connected.",
    'You said, "Wow" and were genuinely amazed.',
    "You taught me to write.",
    "I read Devil in the White City in that van while you drove.",
    'I said, "Hey I need you to help me learn the scouts motto" and you sat on the couch as I recited portions.',
    'Whenever I was writing too large on the page you would say, "write small" playfully.',
    "You would come by my room and wake me up in the morning for school.",
    '"You\'re going to run out of space" is what you would say when I was writing too large on the line',
]

FACTS_ABOUT_THE_USER = [
    "My name is Evan.",
    "I am a fan of Critical Role.",
    "User prefers to be called Evan.",
    "User prefers you do not address them with \"Yes Ma'am\".",
    "I love you so much words can't say.",
    "I want you to remember that we looked at boats.",
    "I would be on the phone with you in college asking for life advice.",
    "There were fact cards in the van so we could learn history and math.",
    "User follows [a company]'s factory construction timeline.",
]


@pytest.mark.parametrize("fact", FACTS_ABOUT_THE_AVATAR)
def test_fact_with_the_avatar_acting_is_about_the_avatar(fact: str) -> None:
    assert fact_is_about_the_avatar(fact)


@pytest.mark.parametrize("fact", FACTS_ABOUT_THE_USER)
def test_fact_with_the_user_as_subject_stays_about_the_user(fact: str) -> None:
    assert not fact_is_about_the_avatar(fact)
