"""Unit tests for the mailbox tools the avatar calls during a turn.

The behaviours pinned here are the ones that decide whether this capability
scales and whether it can misbehave:

- **The tool count stays flat as accounts are added.** Four tools whether the
  owner has one mailbox or five, resolved by an ``account_label`` argument. One
  tool set per account would grow the prompt linearly and is the thing that has
  to not happen for the social accounts the registry already declares.
- **Nothing raises.** A dead server, a rotated password, or a missing key all
  become results the model can read and relay. A raised exception inside a tool
  costs the whole turn and tells the owner nothing useful.
- **Nothing sends.** There is no send tool, and the draft path writes to the
  drafts folder. This is the guarantee the whole landing rests on.
- **Message parsing survives real mail** — encoded headers, HTML-only bodies,
  attachments, and runaway link counts.
"""

import asyncio
from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.tools.email import imap_client
from src.anubis.utils.tools.email.mailbox_tools import build_mailbox_tools


def _context(key=None):
    return SimpleNamespace(
        connected_account_encryption_key=key or secret_store.generate_encryption_key(),
        mailbox_fetch_max_messages=25,
        mailbox_request_timeout_seconds=5.0,
    )


def _record(context, address, label):
    return {
        "account_key": f"gmail:{address}",
        "provider": "gmail",
        "kind": "mailbox",
        "account_address": address,
        "display_label": label,
        "encrypted_secret": secret_store.encrypt_secret("app-password", context),
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "drafts_mailbox": "[Gmail]/Drafts",
        "status": "connected",
    }


def _tools(context, records):
    return {tool.name: tool for tool in build_mailbox_tools(context, records)}


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------


def test_the_tool_set_is_flat_regardless_of_account_count():
    context = _context()
    one = _tools(context, [_record(context, "a@x.com", "a")])
    five = _tools(
        context,
        [_record(context, f"user{index}@x.com", f"user{index}") for index in range(5)],
    )

    assert set(one) == {
        "search_mailbox_messages",
        "read_mailbox_message",
        "read_mailbox_thread",
        "draft_mailbox_reply",
    }
    assert set(five) == set(one), "adding mailboxes must not add tools"


def test_no_tool_can_send_email():
    """The guarantee the landing rests on: drafting exists, sending does not."""
    context = _context()
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    assert not any("send" in name for name in tools)
    assert "draft_mailbox_reply" in tools


def test_an_avatar_with_no_mailbox_gets_no_tools():
    context = _context()
    assert build_mailbox_tools(context, []) == []
    # A connected social account is not a mailbox and must not produce mail tools.
    assert (
        build_mailbox_tools(context, [{"kind": "social", "display_label": "yt"}]) == []
    )


# --------------------------------------------------------------------------
# Account resolution
# --------------------------------------------------------------------------


def test_a_single_mailbox_needs_no_label(monkeypatch):
    context = _context()
    monkeypatch.setattr(
        imap_client,
        "search_messages",
        lambda credentials, query, limit: [{"message_id": "1", "subject": "hi"}],
    )
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    result = asyncio.run(tools["search_mailbox_messages"].coroutine(query="x"))

    assert result["account_label"] == "a"
    assert result["message_count"] == 1


def test_several_mailboxes_require_a_label_rather_than_guessing(monkeypatch):
    """Reading the wrong mailbox is worse than asking which one."""
    context = _context()
    records = [_record(context, "a@x.com", "a"), _record(context, "b@y.com", "b")]
    tools = _tools(context, records)

    result = asyncio.run(tools["search_mailbox_messages"].coroutine(query="x"))

    assert "error" in result
    assert "account_label" in result["error"]
    assert "'a'" in result["error"] and "'b'" in result["error"]


def test_an_unknown_label_names_the_valid_ones(monkeypatch):
    """Listing valid names lets the model self-correct in one step."""
    context = _context()
    records = [_record(context, "a@x.com", "a"), _record(context, "b@y.com", "b")]
    tools = _tools(context, records)

    result = asyncio.run(
        tools["search_mailbox_messages"].coroutine(query="x", account_label="Toaster")
    )

    assert "Toaster" in result["error"]
    assert "'a'" in result["error"] and "'b'" in result["error"]


def test_a_mailbox_can_be_named_by_address_or_label(monkeypatch):
    context = _context()
    records = [_record(context, "a@x.com", "a"), _record(context, "b@y.com", "b")]
    monkeypatch.setattr(
        imap_client, "search_messages", lambda credentials, query, limit: []
    )
    tools = _tools(context, records)

    by_label = asyncio.run(
        tools["search_mailbox_messages"].coroutine(account_label="b")
    )
    by_address = asyncio.run(
        tools["search_mailbox_messages"].coroutine(account_label="b@y.com")
    )

    assert by_label["account_label"] == "b"
    assert by_address["account_label"] == "b"


def test_the_result_limit_is_capped_by_configuration(monkeypatch):
    """A request for 'all my email' must not spend the whole context window."""
    context = _context()
    context.mailbox_fetch_max_messages = 25
    seen = {}

    def _search(credentials, query, limit):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(imap_client, "search_messages", _search)
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    asyncio.run(tools["search_mailbox_messages"].coroutine(limit=9999))

    assert seen["limit"] == 25


# --------------------------------------------------------------------------
# Failures degrade, never raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised,expected_status,expected_phrase",
    [
        (
            imap_client.MailboxAuthenticationError("bad"),
            "needs_reconnect",
            "app password",
        ),
        (imap_client.MailboxUnreachableError("timeout"), "unreachable", "reached"),
        (RuntimeError("something unexpected"), "error", "went wrong"),
    ],
)
def test_a_failing_mailbox_returns_a_result_instead_of_raising(
    monkeypatch, raised, expected_status, expected_phrase
):
    context = _context()

    def _boom(credentials, query, limit):
        raise raised

    monkeypatch.setattr(imap_client, "search_messages", _boom)
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    result = asyncio.run(tools["search_mailbox_messages"].coroutine(query="x"))

    assert result["status"] == expected_status
    assert expected_phrase in result["error"].lower()
    assert result["account_label"] == "a"


def test_a_rotated_encryption_key_asks_for_a_reconnect(monkeypatch):
    """A credential written under an old key must not crash the turn."""
    written_context = _context()
    record = _record(written_context, "a@x.com", "a")
    tools = _tools(_context(), [record])  # different key

    result = asyncio.run(tools["search_mailbox_messages"].coroutine(query="x"))

    assert result["status"] == "needs_reconnect"


def test_a_missing_encryption_key_is_reported_as_configuration(monkeypatch):
    written_context = _context()
    record = _record(written_context, "a@x.com", "a")
    unconfigured = SimpleNamespace(
        connected_account_encryption_key=None,
        mailbox_fetch_max_messages=25,
        mailbox_request_timeout_seconds=5.0,
    )
    tools = _tools(unconfigured, [record])

    result = asyncio.run(tools["search_mailbox_messages"].coroutine(query="x"))

    assert result["status"] == "not_configured"


def test_a_message_that_vanished_is_reported_plainly(monkeypatch):
    """The owner may delete a message between the search and the read."""
    context = _context()
    monkeypatch.setattr(
        imap_client, "fetch_message", lambda credentials, message_id: None
    )
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    result = asyncio.run(tools["read_mailbox_message"].coroutine(message_id="42"))

    assert "error" in result and "42" in result["error"]


# --------------------------------------------------------------------------
# Drafting
# --------------------------------------------------------------------------


def test_drafting_writes_to_the_drafts_folder_and_threads_the_reply(monkeypatch):
    context = _context()
    captured = {}

    def _append(credentials, *, to_address, subject, body_text, in_reply_to=None):
        captured.update(
            drafts=credentials.drafts_mailbox,
            to=to_address,
            subject=subject,
            body=body_text,
            in_reply_to=in_reply_to,
            password=credentials.password,
        )
        return {"status": "draft_saved", "drafts_mailbox": credentials.drafts_mailbox}

    monkeypatch.setattr(imap_client, "append_draft", _append)
    tools = _tools(context, [_record(context, "a@x.com", "a")])

    result = asyncio.run(
        tools["draft_mailbox_reply"].coroutine(
            to="alice@example.com",
            subject="Re: Lunch",
            body="Tuesday works.",
            in_reply_to="<abc@example.com>",
        )
    )

    assert result["status"] == "draft_saved"
    assert result["account_label"] == "a"
    assert captured["drafts"] == "[Gmail]/Drafts"
    assert captured["in_reply_to"] == "<abc@example.com>"
    # The stored credential must decrypt back to the original for the session.
    assert captured["password"] == "app-password"


# --------------------------------------------------------------------------
# Message parsing
# --------------------------------------------------------------------------


def test_the_plain_text_part_wins_over_the_html_alternative():
    message = EmailMessage()
    message["From"] = "=?utf-8?B?w4l2YSBTdMOlbA==?= <eva@example.com>"
    message["Subject"] = "=?utf-8?B?TcO2dGU/?="
    message["Date"] = "Tue, 26 Aug 2026 10:15:00 +0000"
    message["Message-ID"] = "<abc@example.com>"
    message.set_content("Plain body https://calendly.com/x/y pick a time.")
    message.add_alternative("<p>HTML alternative</p>", subtype="html")

    parsed = imap_client._normalize_message(message.as_bytes(), "42", "999")

    assert parsed["subject"] == "Möte?", "encoded headers must be decoded"
    assert "Éva Stål" in parsed["sender"]
    assert "Plain body" in parsed["body_text"]
    assert "HTML alternative" not in parsed["body_text"]
    assert parsed["links"] == ["https://calendly.com/x/y"]
    assert parsed["thread_id"] == "999"
    assert parsed["rfc822_message_id"] == "<abc@example.com>"


def test_an_html_only_message_is_reduced_to_readable_text():
    message = EmailMessage()
    message["Subject"] = "html only"
    message.set_content(
        "<html><body><h1>Hi</h1><p>Meeting at 3.</p></body></html>", subtype="html"
    )

    parsed = imap_client._normalize_message(message.as_bytes(), "7", None)

    assert "Meeting at 3." in parsed["body_text"]
    assert "<p>" not in parsed["body_text"]
    # Without a Gmail thread id the message stands as its own conversation.
    assert parsed["thread_id"] == "7"


def test_attachments_never_reach_the_body():
    """A base64 PDF in the body would be pure noise in a conversation."""
    message = EmailMessage()
    message["Subject"] = "with attachment"
    message.set_content("See attached.")
    message.add_attachment(
        b"\x00\x01BINARYPDFCONTENT",
        maintype="application",
        subtype="pdf",
        filename="report.pdf",
    )

    parsed = imap_client._normalize_message(message.as_bytes(), "9", None)

    assert "See attached." in parsed["body_text"]
    assert "BINARYPDFCONTENT" not in parsed["body_text"]


def test_a_long_body_is_truncated_and_flagged():
    message = EmailMessage()
    message["Subject"] = "long"
    message.set_content("x" * (imap_client.BODY_TEXT_MAX_CHARACTERS + 500))

    parsed = imap_client._normalize_message(message.as_bytes(), "8", None)

    assert len(parsed["body_text"]) == imap_client.BODY_TEXT_MAX_CHARACTERS
    assert parsed["body_truncated"] is True


def test_the_link_count_is_capped():
    """Marketing mail carries hundreds of tracking URLs."""
    message = EmailMessage()
    message["Subject"] = "many links"
    message.set_content("\n".join(f"https://e.example/{index}" for index in range(50)))

    parsed = imap_client._normalize_message(message.as_bytes(), "10", None)

    assert len(parsed["links"]) == imap_client.EXTRACTED_LINK_MAX_COUNT


# --------------------------------------------------------------------------
# The connect card
# --------------------------------------------------------------------------
#
# The card is raised by a LangGraph ``interrupt``, which only works inside a
# running graph. These tests replace ``interrupt`` with a stand-in that captures
# the card payload and returns a chosen resume value, which is what lets the
# payload and the post-resume behaviour be pinned without standing up a graph.


class _FakeStore:
    """The two-method slice of ``BaseStore`` that ``bound_accounts_for`` uses."""

    def __init__(self, records=()):
        self.records = list(records)
        self.searched = []

    async def asearch(self, namespace, limit=100):
        self.searched.append(namespace)
        return [SimpleNamespace(value=record) for record in self.records]


def _connect_tool(monkeypatch, resume_value, store=None, accounts=()):
    """Build the connect tool with ``interrupt`` stubbed out.

    Returns the tool and a one-element list that receives the card payload, so a
    test can assert on what the client would have been asked to render.
    """
    from src.anubis.utils.connected_accounts import connection_tools

    raised_cards = []

    def _fake_interrupt(payload):
        raised_cards.append(payload)
        return resume_value

    monkeypatch.setattr(connection_tools, "interrupt", _fake_interrupt)
    tools = connection_tools.build_connection_tools(
        _context(),
        store=store if store is not None else _FakeStore(),
        user_id="auth0|owner",
        assistant_id="assistant-1",
        connected_accounts=list(accounts),
    )
    return {tool.name: tool for tool in tools}, raised_cards


def test_the_built_mailbox_tools_match_the_declared_tool_names():
    # MAILBOX_TOOL_NAMES is what the connect card counts, so a tool added or
    # removed above without updating the constant would have the card advertise
    # the wrong number of tools.
    from src.anubis.utils.tools.email.mailbox_tools import MAILBOX_TOOL_NAMES

    context = _context()
    built = _tools(context, [_record(context, "a@x.com", "a")])
    assert tuple(built) == MAILBOX_TOOL_NAMES


def test_the_connect_tool_is_offered_when_no_mailbox_is_connected(monkeypatch):
    # The owner with nothing connected is exactly the owner who needs to
    # connect, so this tool must not be gated on having a connection.
    tools, _ = _connect_tool(monkeypatch, {"type": "cancel"})
    assert "connect_mailbox_account" in tools


def test_the_connect_card_describes_the_provider_form(monkeypatch):
    from src.anubis.utils.tools.email.mailbox_tools import MAILBOX_TOOL_NAMES

    tools, cards = _connect_tool(monkeypatch, {"type": "cancel"})
    asyncio.run(tools["connect_mailbox_account"].coroutine())

    card = cards[0]
    assert card["kind"] == "connect_account"
    assert card["provider"] == "gmail"
    assert card["display_name"] == "Gmail"
    assert card["tool_count"] == len(MAILBOX_TOOL_NAMES)
    assert card["connect_endpoint"] == "/connect_mailbox"
    assert card["credential_help_url"] == "https://myaccount.google.com/apppasswords"

    fields = {field["name"]: field for field in card["fields"]}
    assert set(fields) == {"email_address", "app_password"}
    # The secret is declared as a password input so the client masks it without
    # having to know which of the fields is the secret one.
    assert fields["app_password"]["input_type"] == "password"
    # An owner who is not told this types their account password, is rejected,
    # and types the same password again.
    assert "app password" in fields["app_password"]["help_text"].lower()


def test_an_unknown_provider_is_refused_without_raising(monkeypatch):
    tools, cards = _connect_tool(monkeypatch, {"type": "apply"})
    result = asyncio.run(
        tools["connect_mailbox_account"].coroutine(provider="carrier-pigeon")
    )
    assert result["status"] == "unsupported_provider"
    assert "gmail" in result["error"]
    assert cards == [], "an unknown provider must not raise a card"


def test_a_social_provider_cannot_be_connected_as_a_mailbox(monkeypatch):
    # Gmail is kind="mailbox" and the social rows are kind="social"; this tool
    # connects mailboxes only, so a social name must be refused rather than
    # rendering a mailbox sign-in form for an account that has none.
    tools, cards = _connect_tool(monkeypatch, {"type": "apply"})
    result = asyncio.run(tools["connect_mailbox_account"].coroutine(provider="twitch"))
    assert result["status"] == "unsupported_provider"
    assert cards == []


def test_connecting_reports_the_accounts_found_in_the_store(monkeypatch):
    context = _context()
    stored = _record(context, "evan@gmail.com", "evan")
    stored["assistant_id"] = "assistant-1"
    store = _FakeStore([stored])

    tools, _ = _connect_tool(monkeypatch, {"type": "apply"}, store=store)
    result = asyncio.run(tools["connect_mailbox_account"].coroutine())

    assert result["status"] == "connected"
    assert [account["account_address"] for account in result["accounts"]] == [
        "evan@gmail.com"
    ]
    # The reply is grounded in the store, not in anything the client claimed.
    assert store.searched, "the store must be re-read after the card resumes"


def test_a_connected_report_never_carries_the_stored_secret(monkeypatch):
    context = _context()
    stored = _record(context, "evan@gmail.com", "evan")
    stored["assistant_id"] = "assistant-1"
    ciphertext = stored["encrypted_secret"]

    tools, _ = _connect_tool(monkeypatch, {"type": "apply"}, store=_FakeStore([stored]))
    result = asyncio.run(tools["connect_mailbox_account"].coroutine())

    rendered = repr(result)
    assert ciphertext not in rendered
    assert "encrypted_secret" not in rendered


def test_cancelling_the_card_connects_nothing(monkeypatch):
    store = _FakeStore()
    tools, _ = _connect_tool(monkeypatch, {"type": "cancel"}, store=store)
    result = asyncio.run(tools["connect_mailbox_account"].coroutine())

    assert result["status"] == "cancelled"
    assert store.searched == [], "a cancelled card must not even read the store"


def test_finishing_with_nothing_stored_is_reported_as_not_connected(monkeypatch):
    # The card can be dismissed after a failed sign-in. Claiming success here
    # would have the avatar assert a mailbox it cannot read.
    tools, _ = _connect_tool(monkeypatch, {"type": "apply"}, store=_FakeStore())
    result = asyncio.run(tools["connect_mailbox_account"].coroutine())
    assert result["status"] == "not_connected"


def test_a_credential_in_the_resume_value_is_ignored(monkeypatch):
    # THE load-bearing test. A resume value is persisted by the checkpointer, so
    # the card must post the credential to /connect_mailbox and resume with only
    # a decision. Should a client ever send one anyway, this tool must not read
    # it, act on it, or echo it back into the conversation.
    secret = "abcd efgh ijkl mnop"
    store = _FakeStore()
    tools, _ = _connect_tool(
        monkeypatch,
        {
            "type": "apply",
            "email_address": "evan@gmail.com",
            "app_password": secret,
        },
        store=store,
    )
    result = asyncio.run(tools["connect_mailbox_account"].coroutine())

    assert result["status"] == "not_connected", (
        "a credential in the resume value must not be treated as a connection"
    )
    assert secret not in repr(result)
    assert "evan@gmail.com" not in repr(result)
