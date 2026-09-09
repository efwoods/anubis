"""What the owner's decision teaches, and which action the owner's decision takes.

Two gaps in the shipped inbox are pinned down here:

- **An owner's rewrite teaches the avatar's writing.** Until now an edit wrote
  one ``edit_summary`` for that sender and kind, so the owner's correction
  changed *whether* the avatar replies and never *how* the avatar writes. The
  rewrite is now recorded through ``src/anubis/utils/learning/feedback.py`` in
  the same shape a thumbs-up in chat takes — a negative rating for the draft, a
  positive rating for what the owner actually sent, a reusable
  ``communication_style`` preference, and a realness record — so the nightly
  aggregation carries the correction into every reply the avatar writes.
- **The owner may take a different action.** ``HumanResponse`` has always
  carried ``args.action`` and the panel has always sent it, but
  ``apply_owner_decision`` read only the subject and the body, so every edit
  re-sent a reply. An owner answering "do not reply, put this on my calendar"
  now books the appointment instead.

A learning failure must never cost the owner their decision, which is asserted
here too: the preference is still recorded when the lesson call raises.
"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.anubis.utils.connected_accounts import repository as accounts_repository
from src.anubis.utils.inbox import poller, triage
from src.anubis.utils.inbox import repository as inbox_repository
from src.anubis.utils.inbox.repository import InMemoryInboxRepository
from src.anubis.utils.learning.namespaces import RATING_NEGATIVE, RATING_POSITIVE
from src.subgraphs.inbox import graph as inbox_graph_module

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-personal"
ACCOUNT_KEY = "gmail:owner@example.com"

DRAFTED = "Yes, Tuesday works."
OWNER_SENT = "Tuesday is good — see you at the usual place, and bring the contract."


class _RecordingStore:
    """The store as the learning writes see it: namespace → key → value."""

    def __init__(self) -> None:
        self.written: dict[tuple, dict[str, dict]] = {}
        self.deleted: list[tuple] = []

    async def aput(self, namespace, key, value):
        self.written.setdefault(tuple(namespace), {})[key] = value

    async def adelete(self, namespace, key):
        self.deleted.append((tuple(namespace), key))

    async def asearch(self, namespace, query=None, limit=10):
        return []

    def namespaces_containing(self, fragment: str) -> list[tuple]:
        return [
            namespace
            for namespace in self.written
            if any(fragment == str(part) for part in namespace)
        ]

    def values_in(self, fragment: str) -> list[dict]:
        values: list[dict] = []
        for namespace in self.namespaces_containing(fragment):
            values.extend(self.written[namespace].values())
        return values


def _context(**overrides):
    values = dict(inbox_auto_send_confidence=0.9, mailbox_request_timeout_seconds=5.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def _message(subject="Lunch?", body="Are you free Tuesday?"):
    return {
        "message_id": "42",
        "rfc822_message_id": f"<{subject}>",
        "thread_id": "t-1",
        "sender": "alice@example.com",
        "recipients": "owner@example.com",
        "subject": subject,
        "sent_at": "2026-09-03T10:00:00+00:00",
        "body_text": body,
        "links": [],
    }


class _FakeReasoning:
    """Stands in for the structured-output calls the graph makes."""

    def __init__(self, *, decision="respond", lesson=None, lesson_error=None):
        self.decision = decision
        self.lesson = lesson or triage.OwnerEditLesson(
            edit_summary="The owner added the practical detail and dropped the greeting.",
            writing_lesson="Answer with the practical detail first and skip the greeting.",
            changed_substantially=True,
        )
        self.lesson_error = lesson_error
        self.summarized: list[tuple[str, str]] = []
        self.appointment = triage.CalendarEventRequest(
            summary="Lunch with Alice",
            start="2026-09-08T12:00:00",
            end="",
            description="",
            location="",
        )

    def install(self, monkeypatch):
        async def classify_message(context, *, message, preferences):
            return triage.TriageClassification(
                decision=self.decision,
                needs_owner_action=False,
                message_kind="personal_note",
                reason="because",
            )

        async def draft_reply(context, *, message, voice_system_prompt):
            return triage.DraftReply(
                subject=f"Re: {message['subject']}", body=DRAFTED, summary="accepts"
            )

        async def judge_alignment(context, *, message, draft, preferences):
            return triage.PreferenceAlignment(
                aligned=False, alignment_score=0.5, reason="judged"
            )

        async def summarize_owner_edit(context, *, draft_body, final_body, message):
            self.summarized.append((draft_body, final_body))
            if self.lesson_error is not None:
                raise self.lesson_error
            return self.lesson

        async def extract_calendar_event(context, *, message):
            return self.appointment

        monkeypatch.setattr(triage, "classify_message", classify_message)
        monkeypatch.setattr(triage, "draft_reply", draft_reply)
        monkeypatch.setattr(triage, "judge_alignment", judge_alignment)
        monkeypatch.setattr(triage, "summarize_owner_edit", summarize_owner_edit)
        monkeypatch.setattr(triage, "extract_calendar_event", extract_calendar_event)
        return self


@pytest.fixture
def harness(monkeypatch):
    inbox = InMemoryInboxRepository()
    inbox_repository.set_inbox_repository(inbox)
    accounts = accounts_repository.InMemoryConnectedAccountRepository()
    accounts_repository.set_repository(accounts)
    store = _RecordingStore()
    poller.set_inbox_runtime(MemorySaver(), store)

    async def _no_voice_prompt(state, config, runtime):
        return "You are Evan."

    monkeypatch.setattr(inbox_graph_module, "_voice_system_prompt", _no_voice_prompt)

    sent: list[dict] = []

    async def _send(context, **kwargs):
        sent.append(kwargs)
        return {"status": "sent"}

    from src.anubis.utils.inbox import delivery

    monkeypatch.setattr(delivery, "send_email_reply", _send)
    yield SimpleNamespace(inbox=inbox, store=store, sent=sent)
    inbox_repository.set_inbox_repository(None)
    accounts_repository.set_repository(None)
    poller.set_inbox_runtime(None, None)


async def _run(context, message=None):
    return await poller.run_inbox_for_message(
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        account_key=ACCOUNT_KEY,
        message=message or _message(),
        assistant={
            "name": "Evan",
            "metadata": {"user_id": USER_ID, "is_personal_avatar_of_creator": True},
        },
    )


def _edit_response(body, action=None):
    inner = {"body": body, "subject": "Re: Lunch?"}
    args = {"args": inner}
    if action is not None:
        args["action"] = action
    return {"type": "edit", "args": args}


# --------------------------------------------------------------------------
# G1 — the owner's edits teach the avatar's writing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_owner_rewrite_is_learned_as_a_rating_and_a_writing_preference(
    harness, monkeypatch
):
    reasoning = _FakeReasoning().install(monkeypatch)
    item = await _run(_context())
    assert item["state"] == "pending_owner"

    resolved = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response=_edit_response(OWNER_SENT)
    )
    assert resolved["state"] == "sent"
    assert harness.sent, "the owner's edited reply is still sent"
    assert reasoning.summarized == [(DRAFTED, OWNER_SENT)]

    # The draft is rated down and what the owner actually sent is rated up.
    negative = harness.store.values_in(f"rating_{RATING_NEGATIVE}")
    positive = harness.store.values_in(f"rating_{RATING_POSITIVE}")
    assert len(negative) == 1 and DRAFTED in str(negative[0])
    assert len(positive) == 1 and OWNER_SENT in str(positive[0])

    # The two ratings must not share a message identifier: storing one deletes
    # the opposite-polarity record for the same identifier, so a shared
    # identifier would make each write erase the other.
    negative_keys = [
        key
        for namespace in harness.store.namespaces_containing(
            f"rating_{RATING_NEGATIVE}"
        )
        for key in harness.store.written[namespace]
    ]
    positive_keys = [
        key
        for namespace in harness.store.namespaces_containing(
            f"rating_{RATING_POSITIVE}"
        )
        for key in harness.store.written[namespace]
    ]
    assert set(negative_keys).isdisjoint(positive_keys)

    # The reusable lesson lands as a communication-style preference, and the
    # substantial rewrite also moves the realness signal.
    preferences = harness.store.values_in("preference")
    assert any(
        "practical detail first" in str(value) for value in preferences
    ), preferences
    assert harness.store.values_in("what_feels_real")

    # The precedent row carries the real lesson, not the canned sentence.
    recorded = await harness.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID,
        sender="alice@example.com",
        sender_domain="example.com",
        message_kind="personal_note",
    )
    assert any(
        (row.get("edit_summary") or "").startswith("The owner added") for row in recorded
    ), recorded


@pytest.mark.asyncio
async def test_an_accept_dressed_as_an_edit_teaches_nothing(harness, monkeypatch):
    reasoning = _FakeReasoning().install(monkeypatch)
    item = await _run(_context())

    resolved = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response=_edit_response(DRAFTED)
    )
    assert resolved["state"] == "sent"
    assert reasoning.summarized == [], "an unchanged body is not a rewrite"
    assert harness.store.values_in(f"rating_{RATING_POSITIVE}") == []
    assert harness.store.values_in(f"rating_{RATING_NEGATIVE}") == []


@pytest.mark.asyncio
async def test_a_failure_to_learn_never_costs_the_owner_their_decision(
    harness, monkeypatch
):
    _FakeReasoning(lesson_error=RuntimeError("the model is down")).install(monkeypatch)
    item = await _run(_context())

    resolved = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response=_edit_response(OWNER_SENT)
    )
    assert resolved["state"] == "sent"
    assert harness.sent, "the reply is sent even though learning failed"
    recorded = await harness.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID,
        sender="alice@example.com",
        sender_domain="example.com",
        message_kind="personal_note",
    )
    assert recorded, "the decision is still recorded as precedent"


# --------------------------------------------------------------------------
# G3 — precedent generalises past the individual sender
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_decision_also_writes_a_row_keyed_on_the_kind_alone(
    harness, monkeypatch
):
    _FakeReasoning().install(monkeypatch)
    item = await _run(_context())
    await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response=_edit_response(OWNER_SENT)
    )

    # A sender the owner has never ruled on still reaches the coarse row.
    for_a_stranger = await harness.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID,
        sender="someone-new@elsewhere.example",
        sender_domain="elsewhere.example",
        message_kind="personal_note",
    )
    assert for_a_stranger, "the kind-only row must be recalled for an unknown sender"
    coarse_rows = [row for row in for_a_stranger if not (row.get("sender") or "")]
    assert coarse_rows, "the decision must also be recorded against the kind alone"

    # Both key columns are empty strings, never None: the unique key spans them
    # and Postgres ON CONFLICT never matches a NULL, so a None would duplicate
    # the row on every decision instead of counting up.
    coarse = coarse_rows[0]
    assert coarse["sender"] == ""
    assert coarse["sender_domain"] == ""

    # Recall also returns the row about Alice, because the kind matches. For a
    # stranger that row is evidence about the kind, not about the stranger, and
    # must be weighted as such — otherwise a decision about somebody else would
    # count as the owner's history with the person now writing.
    about_alice = next(row for row in for_a_stranger if row.get("sender"))
    assert triage.preference_specificity_weight(
        about_alice,
        sender="someone-new@elsewhere.example",
        sender_domain="elsewhere.example",
    ) == triage.SPECIFICITY_WEIGHT_MESSAGE_KIND
    assert triage.preference_specificity_weight(
        about_alice, sender="alice@example.com", sender_domain="example.com"
    ) == triage.SPECIFICITY_WEIGHT_SENDER


# --------------------------------------------------------------------------
# G2 — the owner may change which action is taken
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_owner_can_book_the_appointment_instead_of_replying(
    harness, monkeypatch
):
    _FakeReasoning().install(monkeypatch)
    booked: list[dict] = []

    class _CalendarTool:
        name = "create_calendar_event"

        async def ainvoke(self, arguments):
            booked.append(arguments)
            return {"status": "created", "event": {"id": "evt-1"}}

    async def _read_connected_accounts(store, user_id):
        return [{"kind": "google_calendar", "provider": "google_calendar"}]

    async def _build_tools_for_accounts(context, accounts, **kwargs):
        return [_CalendarTool()]

    from src.anubis.utils.connected_accounts import store as accounts_store
    from src.anubis.utils.connected_accounts import tool_factories

    monkeypatch.setattr(
        accounts_store, "read_connected_accounts", _read_connected_accounts
    )
    monkeypatch.setattr(
        tool_factories, "build_tools_for_accounts", _build_tools_for_accounts
    )

    item = await _run(_context())
    resolved = await poller.resume_inbox_item(
        _context(),
        item_id=item["item_id"],
        human_response=_edit_response(DRAFTED, action="create_calendar_event"),
    )

    assert harness.sent == [], "the owner asked for a calendar entry, not a reply"
    assert booked and booked[0]["summary"] == "Lunch with Alice"
    assert booked[0]["start"] == "2026-09-08T12:00:00"
    assert resolved["state"] == "resolved"


@pytest.mark.asyncio
async def test_an_action_the_owner_did_not_name_still_sends_the_reply(
    harness, monkeypatch
):
    """The default stays exactly what it was: an accepted or edited reply sends."""
    _FakeReasoning().install(monkeypatch)
    item = await _run(_context())
    resolved = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response=_edit_response(OWNER_SENT)
    )
    assert resolved["state"] == "sent"
    assert len(harness.sent) == 1
