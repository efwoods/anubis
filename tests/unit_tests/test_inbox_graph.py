"""Unit tests for the agent inbox: the triage graph, its interrupts, and preferences.

Pinned down:

- **Ignore never touches the owner.** An ``ignore`` classification resolves the
  item immediately with no interrupt and no send.
- **Notify pauses for the owner** with the Agent Inbox ``notify_owner`` shape,
  mirrors the pending item, and the owner's decision records a preference.
- **A stranger never gets an automatic reply.** With no recorded decisions the
  confidence is capped below the threshold, so the draft waits for the owner;
  accepting or editing it sends, and the decision is learned.
- **Preferences drive the score.** With enough supportive decisions and an
  aligned draft, the reply is sent automatically.
- **A real-world action item always reaches the owner**, however confident.
- **Nothing sends on ignore**, and a send failure lands on the item, not the run.
"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.anubis.utils.connected_accounts import repository as accounts_repository
from src.anubis.utils.inbox import repository as inbox_repository
from src.anubis.utils.inbox import poller, triage
from src.anubis.utils.inbox.repository import InMemoryInboxRepository
from src.subgraphs.inbox import graph as inbox_graph_module

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-personal"
ACCOUNT_KEY = "gmail:owner@example.com"


def _context(**overrides):
    values = dict(inbox_auto_send_confidence=0.9, mailbox_request_timeout_seconds=5.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def _message(sender="alice@example.com", subject="Lunch?", body="Are you free Tuesday?"):
    return {
        "message_id": "42",
        "rfc822_message_id": f"<{subject}-{sender}>",
        "thread_id": "t-1",
        "sender": sender,
        "recipients": "owner@example.com",
        "subject": subject,
        "sent_at": "2026-09-03T10:00:00+00:00",
        "body_text": body,
        "links": [],
    }


class _FakeReasoning:
    """Stands in for the three structured-output calls."""

    def __init__(self, *, decision="respond", needs_owner_action=False, kind="personal_note", alignment=0.5):
        self.decision = decision
        self.needs_owner_action = needs_owner_action
        self.kind = kind
        self.alignment = alignment
        self.judged = []

    def install(self, monkeypatch):
        async def classify_message(context, *, message, preferences):
            return triage.TriageClassification(
                decision=self.decision,
                needs_owner_action=self.needs_owner_action,
                message_kind=self.kind,
                reason="because",
            )

        async def draft_reply(context, *, message, voice_system_prompt):
            return triage.DraftReply(subject=f"Re: {message['subject']}", body="Yes, Tuesday works.", summary="accepts")

        async def judge_alignment(context, *, message, draft, preferences):
            self.judged.append(len(preferences))
            return triage.PreferenceAlignment(aligned=self.alignment >= 0.7, alignment_score=self.alignment, reason="judged")

        monkeypatch.setattr(triage, "classify_message", classify_message)
        monkeypatch.setattr(triage, "draft_reply", draft_reply)
        monkeypatch.setattr(triage, "judge_alignment", judge_alignment)
        return self


@pytest.fixture
def repositories(monkeypatch):
    inbox = InMemoryInboxRepository()
    inbox_repository.set_inbox_repository(inbox)
    accounts = accounts_repository.InMemoryConnectedAccountRepository()
    accounts_repository.set_repository(accounts)
    poller.set_inbox_runtime(MemorySaver(), None)

    async def _no_voice_prompt(state, config, runtime):
        return "You are Evan."

    monkeypatch.setattr(inbox_graph_module, "_voice_system_prompt", _no_voice_prompt)
    sent = []

    async def _send(context, **kwargs):
        sent.append(kwargs)
        return {"status": "sent"}

    from src.anubis.utils.inbox import delivery

    monkeypatch.setattr(delivery, "send_email_reply", _send)
    yield SimpleNamespace(inbox=inbox, accounts=accounts, sent=sent)
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
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID, "is_personal_avatar_of_creator": True}},
    )


@pytest.mark.asyncio
async def test_ignore_resolves_without_asking_or_sending(repositories, monkeypatch):
    _FakeReasoning(decision="ignore", kind="newsletter").install(monkeypatch)
    item = await _run(_context())
    assert item["state"] == "ignored"
    assert item["decision"] == "ignore"
    assert repositories.sent == []
    assert await repositories.inbox.count_open(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_notify_pauses_for_the_owner_and_learns_the_decision(repositories, monkeypatch):
    _FakeReasoning(decision="notify", needs_owner_action=True, kind="invoice").install(monkeypatch)
    item = await _run(_context())
    assert item["state"] == "pending_owner"
    assert await repositories.inbox.count_open(ASSISTANT_ID) == 1

    # The paused thread carries the Agent Inbox HumanInterrupt shape.
    state = await poller._graph().aget_state(poller._run_config(item, None))
    pending = [i for task in state.tasks for i in task.interrupts]
    assert pending, "the run must be paused on an interrupt"
    payload = pending[0].value
    assert payload["action_request"]["action"] == "notify_owner"
    assert payload["config"]["allow_edit"] is False

    resolved = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response={"type": "accept", "args": None}
    )
    assert resolved["state"] == "resolved"
    assert repositories.sent == [], "a notification never sends anything"
    preferences = await repositories.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID, sender="alice@example.com", sender_domain="example.com", message_kind="invoice"
    )
    assert preferences and preferences[0]["decision"] == "accept"


@pytest.mark.asyncio
async def test_a_stranger_waits_for_the_owner_even_when_the_draft_aligns(repositories, monkeypatch):
    _FakeReasoning(decision="respond", alignment=0.99).install(monkeypatch)
    item = await _run(_context())
    assert item["state"] == "pending_owner", "no recorded decisions caps the confidence"
    assert item["draft"] == "Yes, Tuesday works."
    assert item["confidence"] < 0.9
    assert repositories.sent == []

    edited = await poller.resume_inbox_item(
        _context(),
        item_id=item["item_id"],
        human_response={"type": "edit", "args": {"action": "send_reply", "args": {"body": "Tuesday at noon works."}}},
    )
    assert edited["state"] == "sent"
    assert repositories.sent[0]["body_text"] == "Tuesday at noon works."
    assert repositories.sent[0]["to_address"] == "alice@example.com"
    assert repositories.sent[0]["account_key"] == ACCOUNT_KEY
    preferences = await repositories.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID, sender="alice@example.com", sender_domain="example.com", message_kind="personal_note"
    )
    assert preferences[0]["decision"] == "edit"
    assert preferences[0]["edit_summary"]


@pytest.mark.asyncio
async def test_supportive_precedent_plus_an_aligned_draft_sends_automatically(repositories, monkeypatch):
    reasoning = _FakeReasoning(decision="respond", alignment=0.98).install(monkeypatch)
    for _ in range(3):
        await repositories.inbox.record_preference(
            user_id=USER_ID, assistant_id=ASSISTANT_ID, sender="alice@example.com",
            sender_domain="example.com", message_kind="personal_note", decision="accept",
        )
    item = await _run(_context(), _message(subject="Dinner?"))
    assert item["state"] == "auto_sent", item["confidence_detail"]
    assert item["confidence"] >= 0.9
    assert repositories.sent[0]["subject"] == "Re: Dinner?"
    assert reasoning.judged and reasoning.judged[0] >= 1, "the judge saw the precedent"


@pytest.mark.asyncio
async def test_a_real_world_action_item_always_reaches_the_owner(repositories, monkeypatch):
    _FakeReasoning(decision="respond", needs_owner_action=True, alignment=0.99).install(monkeypatch)
    await repositories.inbox.record_preference(
        user_id=USER_ID, assistant_id=ASSISTANT_ID, sender="alice@example.com",
        sender_domain="example.com", message_kind="personal_note", decision="accept",
    )
    for _ in range(4):
        await repositories.inbox.record_preference(
            user_id=USER_ID, assistant_id=ASSISTANT_ID, sender="alice@example.com",
            sender_domain="example.com", message_kind="personal_note", decision="accept",
        )
    item = await _run(_context())
    assert item["state"] == "pending_owner"
    assert repositories.sent == []


@pytest.mark.asyncio
async def test_ignoring_a_proposed_reply_sends_nothing_and_is_learned(repositories, monkeypatch):
    _FakeReasoning(decision="respond", alignment=0.5).install(monkeypatch)
    item = await _run(_context())
    ignored = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response={"type": "ignore", "args": None}
    )
    assert ignored["state"] == "ignored"
    assert repositories.sent == []
    preferences = await repositories.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID, sender="alice@example.com", sender_domain="example.com", message_kind="personal_note"
    )
    assert preferences[0]["decision"] == "ignore"


@pytest.mark.asyncio
async def test_the_same_message_is_never_triaged_twice(repositories, monkeypatch):
    _FakeReasoning(decision="ignore").install(monkeypatch)
    first = await _run(_context())
    second = await _run(_context())
    assert first is not None
    assert second is None
    assert len(repositories.inbox.items) == 1


@pytest.mark.asyncio
async def test_a_send_failure_lands_on_the_item_not_the_run(repositories, monkeypatch):
    _FakeReasoning(decision="respond", alignment=0.5).install(monkeypatch)
    from src.anubis.utils.inbox import delivery

    async def _fail(context, **kwargs):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(delivery, "send_email_reply", _fail)
    item = await _run(_context())
    failed = await poller.resume_inbox_item(
        _context(), item_id=item["item_id"], human_response={"type": "accept", "args": None}
    )
    assert failed["state"] == "failed"
    assert "smtp down" in failed["reason"]


def test_the_preference_prior_rewards_consistent_accepts_and_caps_strangers():
    capped, _ = triage.preference_prior([], auto_send_threshold=0.9)
    assert capped < 0.9
    strong, _ = triage.preference_prior(
        [{"decision": "accept", "count": 4}], auto_send_threshold=0.9
    )
    assert strong == 1.0
    mixed, _ = triage.preference_prior(
        [{"decision": "accept", "count": 2}, {"decision": "ignore", "count": 2}], auto_send_threshold=0.9
    )
    assert 0.3 < mixed < 0.7
    assert triage.combine_confidence(0.95, strong) == pytest.approx(0.95)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_count_and_items_routes_report_the_owner_inbox(repositories, monkeypatch):
    from src.api import webapp as webapp_module

    _FakeReasoning(decision="notify").install(monkeypatch)
    await _run(_context())

    monkeypatch.setattr(webapp_module, "get_client", lambda **kwargs: SimpleNamespace())

    async def _resolve(client, request, user, api_key):
        return {"assistant_id": ASSISTANT_ID}

    monkeypatch.setattr(webapp_module, "_resolve_personal_avatar_for_connection", _resolve)
    current_user = {"API_KEY": "k", "identities": [{"user_id": USER_ID}]}

    count = await webapp_module.inbox_count(request=SimpleNamespace(), current_user=current_user)
    assert b'"pending_count":1' in count.body.replace(b" ", b"")

    listing = await webapp_module.list_inbox_items(
        request=SimpleNamespace(), state="open", limit=10, current_user=current_user
    )
    body = listing.body.decode("utf-8")
    assert "Lunch?" in body
    assert "notify_owner" not in body, "the panel view is the item, not the raw interrupt"
