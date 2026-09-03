"""Mailbox access over IMAP, using the standard library's ``imaplib``.

Why the standard library and not ``langchain_google_community.GmailToolkit``
    The toolkit builds on a ``googleapiclient`` resource constructed from OAuth
    credentials. Reading mail through that API needs a scope Google classifies
    as *restricted*, which obliges a published application to pass OAuth
    verification and an annual CASA security assessment. Connecting with an app
    password over IMAP needs no OAuth client at all, so neither requirement ever
    applies and the credential does not expire. The function names below
    deliberately track the toolkit's tools (search / get message / get thread /
    create draft) so a future OAuth-backed implementation can be dropped in
    behind the same call sites.

    Since 2025-03-14 a regular Google account password no longer authenticates
    against IMAP or SMTP; only OAuth 2.0 and app passwords do, and creating an
    app password requires 2-Step Verification on the account.
    :func:`verify_credentials` exists to turn that into an actionable message at
    connect time instead of a mysterious failure on the first mail call.

Every call here is BLOCKING
    ``imaplib`` is a synchronous socket client. This module is
    written as plain synchronous functions and the tool layer is responsible for
    running each one in a worker thread, so a slow or unreachable mail server
    cannot stall the event loop that is concurrently streaming tokens to other
    conversations. Keeping the blocking calls unwrapped here also makes the
    module trivially testable without an event loop.
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import logging
import re
import time
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

# Ceiling on how much of one message body is carried back to the model. A mail
# thread can run to tens of thousands of characters, and the conversation only
# needs enough to decide what the message is and what to do about it; the full
# text remains fetchable by naming the message explicitly.
BODY_TEXT_MAX_CHARACTERS = 4000

# Ceiling on how many links are reported from one message. Marketing mail can
# carry hundreds of tracking URLs, which would swamp the reply without adding
# anything a reader would act on.
EXTRACTED_LINK_MAX_COUNT = 20

_URL_PATTERN = re.compile(r"https?://[^\s<>\"')]+")


class MailboxAuthenticationError(RuntimeError):
    """The mail server rejected the supplied address and password."""


class MailboxUnreachableError(RuntimeError):
    """The mail server could not be reached or failed mid-operation."""


@dataclass(frozen=True)
class MailboxCredentials:
    """Everything needed to open one mailbox session.

    Built by the tool layer from a stored record plus the decrypted password, so
    the plaintext exists only for the duration of a call and is never part of a
    record, a log line, or a tool result.
    """

    account_address: str
    password: str
    imap_host: str
    imap_port: int = 993
    smtp_host: str | None = None
    smtp_port: int = 587
    drafts_mailbox: str = "Drafts"
    timeout_seconds: float = 30.0


def _decode_header_value(raw_value: Any) -> str:
    """Decode a MIME-encoded header into readable text.

    Subjects and display names arrive RFC 2047 encoded (``=?utf-8?B?...?=``).
    Handing that to the model verbatim would make every non-ASCII subject
    unreadable, so it is decoded here rather than at each call site.
    """
    if raw_value is None:
        return ""
    try:
        return str(make_header(decode_header(str(raw_value))))
    except Exception:
        return str(raw_value)


def _html_to_text(html_markup: str) -> str:
    """Reduce an HTML body to visible text.

    Uses the same BeautifulSoup ``html.parser`` approach as the API module's
    ``_extract_text_from_html_bytes``, reimplemented here rather than imported:
    that helper lives in ``src/api/webapp.py``, and importing the FastAPI
    application into a tool module would drag the whole web app — auth, billing,
    every route — into any process that only wanted to read mail.
    """
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(html_markup, "html.parser").get_text(" ")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", html_markup)


def _extract_body_text(message: email.message.Message) -> str:
    """Return the readable text of a message, preferring a plain-text part.

    Multipart mail usually carries both ``text/plain`` and ``text/html``
    alternatives. The plain part is preferred because it is what the sender
    wrote; the HTML part is converted only when no plain part exists. Attachments
    are skipped outright — a base64 PDF would be pure noise in a conversation.
    """
    plain_segments: list[str] = []
    html_segments: list[str] = []

    for part in message.walk() if message.is_multipart() else [message]:
        if part.get_content_maintype() == "multipart":
            continue
        if "attachment" in str(part.get("Content-Disposition") or "").lower():
            continue
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            continue
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            decoded = payload.decode(charset, errors="replace")
        except LookupError:
            decoded = payload.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            plain_segments.append(decoded)
        else:
            html_segments.append(decoded)

    if plain_segments:
        body_text = "\n".join(plain_segments)
    else:
        body_text = _html_to_text("\n".join(html_segments))
    # Collapse the runs of blank lines mail clients leave behind, so the
    # character budget is spent on content rather than on whitespace.
    return re.sub(r"\n{3,}", "\n\n", body_text).strip()


def _extract_links(body_text: str, raw_html: str = "") -> list[str]:
    """Collect the distinct URLs a message contains, in first-seen order.

    Links matter because acting on a message often means following one. Order is
    preserved rather than sorted so the first link in the body — usually the
    call to action — stays first.
    """
    seen: list[str] = []
    for candidate in _URL_PATTERN.findall(f"{body_text}\n{raw_html}"):
        cleaned = candidate.rstrip(".,;:)?]}>\"'")
        if cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= EXTRACTED_LINK_MAX_COUNT:
            break
    return seen


def _normalize_message(
    raw_bytes: bytes, message_uid: str, thread_id: str | None
) -> dict[str, Any]:
    """Turn one fetched RFC 822 message into the shape the tools return."""
    parsed = email.message_from_bytes(raw_bytes)
    body_text = _extract_body_text(parsed)
    sent_at = None
    if parsed.get("Date"):
        try:
            sent_at = parsedate_to_datetime(parsed["Date"]).isoformat()
        except Exception:
            sent_at = str(parsed.get("Date"))
    return {
        "message_id": message_uid,
        "thread_id": thread_id or message_uid,
        "sender": _decode_header_value(parsed.get("From")),
        "recipients": _decode_header_value(parsed.get("To")),
        "subject": _decode_header_value(parsed.get("Subject")),
        "sent_at": sent_at,
        "body_text": body_text[:BODY_TEXT_MAX_CHARACTERS],
        "body_truncated": len(body_text) > BODY_TEXT_MAX_CHARACTERS,
        "links": _extract_links(body_text),
        "rfc822_message_id": str(parsed.get("Message-ID") or ""),
    }


def _connect(credentials: MailboxCredentials) -> imaplib.IMAP4_SSL:
    """Open an authenticated IMAP session, mapping failures to typed errors.

    An authentication failure and an unreachable server are separated because
    the caller's remedies differ: the first means the owner must supply a new app
    password, the second means try again later. Collapsing them would make the
    avatar tell a user to re-authenticate every time their network blipped.
    """
    try:
        connection = imaplib.IMAP4_SSL(
            credentials.imap_host,
            credentials.imap_port,
            timeout=credentials.timeout_seconds,
        )
    except Exception as connect_error:
        raise MailboxUnreachableError(
            f"Could not reach {credentials.imap_host}: {connect_error}"
        ) from connect_error

    try:
        connection.login(credentials.account_address, credentials.password)
    except imaplib.IMAP4.error as login_error:
        try:
            connection.logout()
        except Exception:
            pass
        raise MailboxAuthenticationError(str(login_error)) from login_error
    except Exception as unexpected_error:
        try:
            connection.logout()
        except Exception:
            pass
        raise MailboxUnreachableError(str(unexpected_error)) from unexpected_error
    return connection


def _close(connection: imaplib.IMAP4_SSL) -> None:
    """Close a session without letting teardown mask the real result."""
    try:
        connection.logout()
    except Exception:
        logger.debug("Ignoring error while closing IMAP session", exc_info=True)


def verify_credentials(credentials: MailboxCredentials) -> None:
    """Prove an address and password work before anything is stored.

    Called by the connect endpoint so a bad credential is rejected at the moment
    the owner supplies it, while they still have the connect form in front of
    them — rather than being written to the store and failing on the first mail
    call days later.

    Raises:
        MailboxAuthenticationError: The server rejected the credential. For
            Gmail the overwhelmingly likely cause is that the owner supplied
            their account password rather than an app password.
        MailboxUnreachableError: The server could not be reached.
    """
    connection = _connect(credentials)
    _close(connection)


def _select_mailbox(
    connection: imaplib.IMAP4_SSL, mailbox: str, readonly: bool = True
) -> None:
    """Select a folder, raising a typed error when it does not exist."""
    status, _ = connection.select(f'"{mailbox}"', readonly=readonly)
    if status != "OK":
        raise MailboxUnreachableError(
            f"Mailbox folder {mailbox!r} could not be opened."
        )


def _supports_gmail_extensions(connection: imaplib.IMAP4_SSL) -> bool:
    """Whether the server advertises Gmail's IMAP extensions.

    Gmail exposes ``X-GM-RAW`` (full Gmail search syntax, so a query like
    ``from:alice has:attachment`` works verbatim) and ``X-GM-THRID`` (a stable
    conversation identifier). Both are far better than the IMAP fallbacks, but
    neither is standard, so their use is capability-gated rather than assumed —
    which also keeps this client working against a non-Gmail provider added to
    the registry later.
    """
    capabilities = getattr(connection, "capabilities", ()) or ()
    return any("X-GM-EXT-1" in str(capability).upper() for capability in capabilities)


def search_messages(
    credentials: MailboxCredentials,
    query: str | None = None,
    limit: int = 10,
    mailbox: str = "INBOX",
) -> list[dict[str, Any]]:
    """Return the most recent messages matching a query, newest first.

    Args:
        credentials: The mailbox to search.
        query: Gmail search syntax on a Gmail server (``from:alice newer_than:7d``),
            otherwise a plain substring matched against the whole message. When
            omitted, the most recent messages are returned unfiltered.
        limit: Maximum messages to return.
        mailbox: IMAP folder to search.

    Newest-first ordering matters: IMAP returns matching sequence numbers in
    ascending order, so taking the LAST ``limit`` identifiers and reversing them
    is what makes "find the recent email about X" answer with the recent one
    rather than the oldest one in the folder.
    """
    connection = _connect(credentials)
    try:
        _select_mailbox(connection, mailbox)
        if query and query.strip():
            if _supports_gmail_extensions(connection):
                status, data = connection.uid(
                    "SEARCH", "X-GM-RAW", f'"{query.strip()}"'
                )
            else:
                status, data = connection.uid("SEARCH", "TEXT", query.strip())
        else:
            status, data = connection.uid("SEARCH", "ALL")
        if status != "OK":
            raise MailboxUnreachableError("The mail server rejected the search.")

        identifiers = data[0].split() if data and data[0] else []
        selected = list(reversed(identifiers[-max(1, int(limit)) :]))

        results: list[dict[str, Any]] = []
        for identifier in selected:
            message_uid = identifier.decode("ascii", errors="ignore")
            fetched = _fetch_one(connection, message_uid)
            if fetched is not None:
                results.append(fetched)
        return results
    finally:
        _close(connection)


def _fetch_one(
    connection: imaplib.IMAP4_SSL, message_uid: str
) -> dict[str, Any] | None:
    """Fetch and normalize a single message by UID, or None when it is gone.

    A message can disappear between the search and the fetch — the owner may be
    reading the same mailbox in another client. That is an ordinary race, not an
    error, so it yields None and the caller simply reports one fewer result.
    """
    thread_id = None
    if _supports_gmail_extensions(connection):
        try:
            status, thread_data = connection.uid("FETCH", message_uid, "(X-GM-THRID)")
            if status == "OK" and thread_data and thread_data[0]:
                match = re.search(rb"X-GM-THRID\s+(\d+)", thread_data[0])
                if match:
                    thread_id = match.group(1).decode("ascii")
        except Exception:
            logger.debug("Could not read Gmail thread id", exc_info=True)

    try:
        status, data = connection.uid("FETCH", message_uid, "(RFC822)")
    except Exception as fetch_error:
        raise MailboxUnreachableError(str(fetch_error)) from fetch_error
    if status != "OK" or not data or not data[0]:
        return None
    raw_bytes = data[0][1] if isinstance(data[0], tuple) else None
    if not raw_bytes:
        return None
    return _normalize_message(raw_bytes, message_uid, thread_id)


def fetch_message(
    credentials: MailboxCredentials, message_id: str, mailbox: str = "INBOX"
) -> dict[str, Any] | None:
    """Return one message in full by the identifier a search reported."""
    connection = _connect(credentials)
    try:
        _select_mailbox(connection, mailbox)
        return _fetch_one(connection, str(message_id).strip())
    finally:
        _close(connection)


def fetch_thread(
    credentials: MailboxCredentials, thread_id: str, mailbox: str = "INBOX"
) -> list[dict[str, Any]]:
    """Return every message in one conversation, oldest first.

    Oldest-first here, unlike search: a conversation is read in the order it
    happened. On a Gmail server the thread is resolved with ``X-GM-THRID``; on
    any other server the identifier is treated as a message identifier and the
    single message is returned, since there is no portable IMAP notion of a
    conversation.
    """
    connection = _connect(credentials)
    try:
        _select_mailbox(connection, mailbox)
        if not _supports_gmail_extensions(connection):
            single = _fetch_one(connection, str(thread_id).strip())
            return [single] if single else []

        status, data = connection.uid("SEARCH", "X-GM-THRID", str(thread_id).strip())
        if status != "OK":
            return []
        identifiers = data[0].split() if data and data[0] else []
        messages: list[dict[str, Any]] = []
        for identifier in identifiers:
            fetched = _fetch_one(
                connection, identifier.decode("ascii", errors="ignore")
            )
            if fetched is not None:
                messages.append(fetched)
        return messages
    finally:
        _close(connection)


def append_draft(
    credentials: MailboxCredentials,
    *,
    to_address: str,
    subject: str,
    body_text: str,
    in_reply_to: str | None = None,
) -> dict[str, Any]:
    """Save a draft into the mailbox's drafts folder.

    A draft is APPENDed over IMAP rather than sent over SMTP. That is the whole
    point of this landing: the avatar composes in the owner's voice and the
    result waits in the owner's own mail client for them to read and send. No
    code path in this module transmits a message, so "the avatar sent something
    I did not approve" is not a failure this feature can have.

    ``in_reply_to`` carries the original's ``Message-ID`` so the draft threads
    correctly in the mail client instead of starting a new conversation.
    """
    draft = EmailMessage()
    draft["From"] = credentials.account_address
    draft["To"] = to_address
    draft["Subject"] = subject
    if in_reply_to:
        draft["In-Reply-To"] = in_reply_to
        draft["References"] = in_reply_to
    draft.set_content(body_text)

    connection = _connect(credentials)
    try:
        status, _ = connection.append(
            f'"{credentials.drafts_mailbox}"',
            "\\Draft",
            imaplib.Time2Internaldate(time.time()),
            draft.as_bytes(),
        )
        if status != "OK":
            raise MailboxUnreachableError(
                f"The mail server refused to save the draft in "
                f"{credentials.drafts_mailbox!r}."
            )
    finally:
        _close(connection)

    return {
        "status": "draft_saved",
        "drafts_mailbox": credentials.drafts_mailbox,
        "to": to_address,
        "subject": subject,
    }


class MailboxSendError(RuntimeError):
    """The submission server accepted the session but refused the message."""


def send_message(
    credentials: MailboxCredentials,
    *,
    to_address: str,
    subject: str,
    body_text: str,
    in_reply_to: str | None = None,
    cc_addresses: list[str] | None = None,
) -> dict[str, Any]:
    """Transmit a message through the provider's SMTP submission server.

    The owner asked for sending — "send it" — and drafting alone was the earlier
    landing's deliberate limit, so this function is the single code path that
    transmits. The tool layer gates it on the provider row's ``send_supported``
    and the ``MAILBOX_SEND_ENABLED`` switch, and the prompt instructs the avatar
    to send only when the owner explicitly asks in the conversation; the future
    inbox triage graph sends only through its confidence gate or an accepted
    approval.

    Same app password as IMAP: Google accepts an app password on the submission
    port (587, STARTTLS) exactly as on IMAP, so no second credential is needed.

    Raises:
        MailboxAuthenticationError: The submission server rejected the login.
        MailboxUnreachableError: The server could not be reached.
        MailboxSendError: The server refused the recipients or the message.
    """
    import smtplib

    if not credentials.smtp_host:
        raise MailboxSendError(
            "This mailbox provider declares no submission server, so nothing can be sent."
        )

    message = EmailMessage()
    message["From"] = credentials.account_address
    message["To"] = to_address
    if cc_addresses:
        message["Cc"] = ", ".join(cc_addresses)
    message["Subject"] = subject
    message["Date"] = email.utils.formatdate(localtime=True)
    message["Message-ID"] = email.utils.make_msgid()
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
        message["References"] = in_reply_to
    message.set_content(body_text)

    recipients = [to_address, *(cc_addresses or [])]
    try:
        with smtplib.SMTP(
            credentials.smtp_host,
            credentials.smtp_port,
            timeout=credentials.timeout_seconds,
        ) as session:
            session.ehlo()
            session.starttls()
            session.ehlo()
            try:
                session.login(credentials.account_address, credentials.password)
            except smtplib.SMTPAuthenticationError as authentication_error:
                raise MailboxAuthenticationError(
                    "The submission server rejected the address and password."
                ) from authentication_error
            refused = session.send_message(message, to_addrs=recipients)
    except (MailboxAuthenticationError, MailboxSendError):
        raise
    except smtplib.SMTPRecipientsRefused as refused_error:
        raise MailboxSendError(
            f"The server refused every recipient: {refused_error.recipients}"
        ) from refused_error
    except smtplib.SMTPException as smtp_error:
        raise MailboxUnreachableError(
            f"The submission server failed while sending: {smtp_error}"
        ) from smtp_error
    except OSError as socket_error:
        raise MailboxUnreachableError(
            f"The submission server could not be reached: {socket_error}"
        ) from socket_error

    if refused:
        raise MailboxSendError(f"The server refused some recipients: {refused}")

    return {
        "status": "sent",
        "to": to_address,
        "cc": list(cc_addresses or []),
        "subject": subject,
        "rfc822_message_id": message["Message-ID"],
    }
