"""The owner's accounts live on their personal avatar, and every avatar they own uses them.

Observed failure this came from: the owner asked an avatar of theirs to put a
date on the calendar. The avatar answered "I don't have access to your calendar
from here" and stopped. That was true — connected-account tools attach only to
the personal avatar — but it was a dead end, with no card and no way forward.

The rule this pins down: the owner is themselves whichever of their avatars they
are talking to. So the accounts of the PERSONAL avatar are the accounts every
avatar they own acts on, and a connection made mid-conversation binds to the
personal avatar rather than being copied onto whichever avatar happened to ask.

What must NOT change is the visitor case. Brokering keys off ownership, so a
stranger talking to a shared avatar still reaches nothing.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.personal_avatar import (
    PERSONAL_AVATAR_METADATA_FLAG,
    personal_avatar_id_for_owner,
)

OWNER_ID = "auth0|owner"
STRANGER_ID = "auth0|stranger"
PERSONAL_AVATAR_ID = "avatar-personal"
OTHER_AVATAR_ID = "avatar-shivon"


def _config(*, owner_id: str, personal: bool):
    return {
        "configurable": {
            "assistant_ctx": {
                "metadata": {
                    "user_id": owner_id,
                    PERSONAL_AVATAR_METADATA_FLAG: personal,
                }
            }
        }
    }


def _state(*, conversing_user: str, assistant_id: str):
    return {
        "user_state": {"user_id": conversing_user},
        "assistant_state": {"assistant_id": assistant_id},
    }


""" Who counts as the owner, and which avatar's accounts they reach """


def test_the_personal_avatar_uses_its_own_accounts():
    from src.anubis.graph import _user_owns_avatar, _user_personal_avatar

    config = _config(owner_id=OWNER_ID, personal=True)
    state = _state(conversing_user=OWNER_ID, assistant_id=PERSONAL_AVATAR_ID)
    assert _user_personal_avatar(config, state) is True
    assert _user_owns_avatar(config, state) is True


def test_another_avatar_of_the_owner_is_owned_but_not_personal():
    """This is the case from the transcript — owned, not personal, so brokered."""
    from src.anubis.graph import _user_owns_avatar, _user_personal_avatar

    config = _config(owner_id=OWNER_ID, personal=False)
    state = _state(conversing_user=OWNER_ID, assistant_id=OTHER_AVATAR_ID)
    assert _user_personal_avatar(config, state) is False
    assert _user_owns_avatar(config, state) is True


def test_a_visitor_owns_nothing_and_brokers_nothing():
    """The property the gate exists for: brokering must not widen it."""
    from src.anubis.graph import _user_owns_avatar, _user_personal_avatar

    config = _config(owner_id=OWNER_ID, personal=False)
    state = _state(conversing_user=STRANGER_ID, assistant_id=OTHER_AVATAR_ID)
    assert _user_owns_avatar(config, state) is False
    assert _user_personal_avatar(config, state) is False


def test_a_visitor_on_the_personal_avatar_reaches_nothing_either():
    from src.anubis.graph import _user_owns_avatar, _user_personal_avatar

    config = _config(owner_id=OWNER_ID, personal=True)
    state = _state(conversing_user=STRANGER_ID, assistant_id=PERSONAL_AVATAR_ID)
    assert _user_owns_avatar(config, state) is False
    assert _user_personal_avatar(config, state) is False


""" Finding the personal avatar when the store pointer has not been written """


class _Cursor:
    def __init__(self, row, recorder):
        self._row = row
        self._recorder = recorder

    async def execute(self, query, parameters):
        self._recorder["query"] = query
        self._recorder["parameters"] = parameters

    async def fetchone(self):
        return self._row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception):
        return False


class _Connection:
    def __init__(self, row, recorder):
        self._row = row
        self._recorder = recorder

    def cursor(self):
        return _Cursor(self._row, self._recorder)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception):
        return False


class _Pool:
    def __init__(self, row):
        self.row = row
        self.recorder: dict = {}

    def connection(self):
        return _Connection(self.row, self.recorder)


@pytest.mark.asyncio
async def test_the_personal_avatar_is_found_in_the_assistant_table():
    """The pointer is written on the personal avatar's first turn; this does not need it."""
    pool = _Pool((PERSONAL_AVATAR_ID,))
    found = await personal_avatar_id_for_owner(pool, OWNER_ID)
    assert found == PERSONAL_AVATAR_ID
    # Both halves matter: this owner's row, and the personal one among them.
    assert pool.recorder["parameters"] == (OWNER_ID,)
    assert "is_personal_avatar_of_creator" in pool.recorder["query"]
    assert "metadata->>'user_id'" in pool.recorder["query"]


@pytest.mark.asyncio
async def test_an_owner_with_no_personal_avatar_brokers_nothing():
    """A real state, not an error: provisioning happens on first API-key auth."""
    assert await personal_avatar_id_for_owner(_Pool(None), OWNER_ID) is None


@pytest.mark.asyncio
async def test_a_lookup_failure_is_not_allowed_to_break_the_turn():
    class _BrokenPool:
        def connection(self):
            raise RuntimeError("the database is unreachable")

    assert await personal_avatar_id_for_owner(_BrokenPool(), OWNER_ID) is None


@pytest.mark.asyncio
async def test_no_pool_and_no_user_are_answered_quietly():
    assert await personal_avatar_id_for_owner(None, OWNER_ID) is None
    assert await personal_avatar_id_for_owner(_Pool((PERSONAL_AVATAR_ID,)), "") is None


""" The connect tool is offered before anything is connected """


def test_the_connect_tool_exists_with_no_accounts_connected():
    """An owner with nothing connected is exactly the owner who needs the card.

    Without this the first connection would be impossible: the tool that raises
    the card would appear only once a card had already been accepted.
    """
    from src.anubis.utils.connected_accounts.connection_tools import (
        build_connection_tools,
    )

    tools = build_connection_tools(
        SimpleNamespace(),
        store=SimpleNamespace(),
        user_id=OWNER_ID,
        assistant_id=PERSONAL_AVATAR_ID,
        connected_accounts=[],
        stale_accounts=[],
    )
    assert [getattr(tool, "name", "") for tool in tools]
