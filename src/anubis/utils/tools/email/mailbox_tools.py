"""The mailbox tools the avatar calls during a conversation turn.

Built as a per-turn factory closing over the authenticated
``(user_id, assistant_id)`` pair and the accounts bound to the answering avatar,
matching ``build_data_analysis_tools``. Nothing here reads identity from a tool
argument, so a model cannot reach another user's mail by inventing an argument
value — the account list is fixed before the model is given the tools.

Two shapes are carried over from the multi-device work on purpose:

**One tool set regardless of how many accounts are connected.** Each tool takes
an optional ``account_label`` resolved against the bound accounts, exactly as
the data-analysis tools take a ``device_label``. Emitting one set of tools per
account would grow the tool list — and the prompt describing it — linearly with
the number of mailboxes, which is the thing that has to stay flat for this to
scale to the social accounts the registry already declares.

**Errors are returned, not raised.** An unreachable server or a stale credential
becomes a result the model can read and relay ("your work mailbox needs to be
reconnected"), because a raised exception inside a tool costs the whole turn and
tells the owner nothing actionable.

Every mail call is blocking, so each one is dispatched with ``asyncio.to_thread``
— see the note in ``imap_client``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

# The tools a connected mailbox contributes, named once so the connect card can
# state how many tools connecting an account adds without hardcoding a number
# that silently goes stale the moment a tool is added or removed here.
# ``test_mailbox_tools`` asserts the built tool set matches this tuple exactly.
MAILBOX_TOOL_NAMES: tuple[str, ...] = (
    "search_mailbox_messages",
    "read_mailbox_message",
    "read_mailbox_thread",
    "draft_mailbox_reply",
)


def _credentials_for(record: dict[str, Any], context: Any) -> Any:
    """Build live mailbox credentials from a stored record.

    The plaintext password exists only inside this call and the mail session it
    feeds; it is never returned, logged, or written back to a record.
    """
    from src.anubis.utils.secret_store import decrypt_secret
    from src.anubis.utils.tools.email.imap_client import MailboxCredentials

    return MailboxCredentials(
        account_address=record["account_address"],
        password=decrypt_secret(record["encrypted_secret"], context),
        imap_host=record["imap_host"],
        imap_port=int(record.get("imap_port") or 993),
        smtp_host=record.get("smtp_host"),
        smtp_port=int(record.get("smtp_port") or 587),
        drafts_mailbox=record.get("drafts_mailbox") or "Drafts",
        timeout_seconds=float(
            getattr(context, "mailbox_request_timeout_seconds", None) or 30.0
        ),
    )


def build_mailbox_tools(context: Any, accounts: list[dict[str, Any]]) -> list[Any]:
    """Build the per-turn mailbox tool set across every connected account.

    Args:
        context: The ``GlobalContext`` carrying the encryption key and timeouts.
        accounts: The connected-account records bound to the answering avatar,
            already filtered by ``bound_accounts_for``. An empty list yields no
            tools, so the caller need not branch.

    Returns:
        The tools to append to the deep agent's tool list, or an empty list.
    """
    if not accounts:
        return []

    mailbox_accounts = [
        record for record in accounts if record.get("kind") == "mailbox"
    ]
    if not mailbox_accounts:
        return []

    fetch_limit = int(getattr(context, "mailbox_fetch_max_messages", None) or 25)

    def _labels() -> list[str]:
        return [str(record.get("display_label") or "") for record in mailbox_accounts]

    def _select_account(
        account_label: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve the account a call names, or explain why it could not.

        Defaulting to the sole account when only one is connected is what keeps
        the common case free of an argument the owner would find pointless. When
        several are connected and none is named, the model is asked to choose
        rather than one being picked silently — reading the wrong mailbox is not
        a mistake worth guessing at.

        An unknown label reports the valid labels so the model can correct itself
        in one step instead of guessing repeatedly.
        """
        if account_label is None or not str(account_label).strip():
            if len(mailbox_accounts) == 1:
                return mailbox_accounts[0], None
            return None, {
                "error": (
                    "Several mailboxes are connected, so the mailbox must be "
                    "named with the account_label argument. Connected "
                    f"mailboxes: {_labels()}."
                )
            }
        wanted = str(account_label).strip().lower()
        for record in mailbox_accounts:
            if str(record.get("display_label") or "").strip().lower() == wanted:
                return record, None
            if str(record.get("account_address") or "").strip().lower() == wanted:
                return record, None
        return None, {
            "error": (
                f"No connected mailbox is named {account_label!r}. "
                f"Connected mailboxes: {_labels()}."
            )
        }

    async def _run(record: dict[str, Any], operation: Any, *args: Any) -> Any:
        """Run one blocking mailbox operation off the event loop.

        Maps the client's typed failures onto result dictionaries. An
        authentication failure additionally flags the stored record, so the next
        turn's status block can tell the owner which account to reconnect rather
        than the avatar rediscovering the failure every time.
        """
        from src.anubis.utils.secret_store import (
            SecretDecryptionError,
            SecretEncryptionNotConfiguredError,
        )
        from src.anubis.utils.tools.email.imap_client import (
            MailboxAuthenticationError,
            MailboxUnreachableError,
        )

        label = record.get("display_label")
        try:
            credentials = _credentials_for(record, context)
        except SecretEncryptionNotConfiguredError:
            return {
                "status": "not_configured",
                "account_label": label,
                "error": (
                    "Mailbox credentials cannot be read because the server's "
                    "credential encryption key is not configured."
                ),
            }
        except SecretDecryptionError:
            return {
                "status": "needs_reconnect",
                "account_label": label,
                "error": (
                    f"The stored credential for {label} could not be read. "
                    "Ask the owner to connect the mailbox again."
                ),
            }

        try:
            return await asyncio.to_thread(operation, credentials, *args)
        except MailboxAuthenticationError:
            logger.info("Mailbox %s rejected its stored credential", label)
            return {
                "status": "needs_reconnect",
                "account_label": label,
                "error": (
                    f"{label} rejected its saved password. Ask the owner to "
                    "generate a new app password and connect the mailbox again."
                ),
            }
        except MailboxUnreachableError as unreachable_error:
            logger.info("Mailbox %s unreachable: %s", label, unreachable_error)
            return {
                "status": "unreachable",
                "account_label": label,
                "error": (
                    f"{label} could not be reached just now. Say so plainly "
                    "rather than reporting that there is no mail."
                ),
            }
        except Exception:
            logger.exception("Unexpected failure reading mailbox %s", label)
            return {
                "status": "error",
                "account_label": label,
                "error": f"Something went wrong reading {label}.",
            }

    @tool
    async def search_mailbox_messages(
        query: str | None = None,
        limit: int = 10,
        account_label: str | None = None,
    ) -> dict[str, Any]:
        """Search the owner's connected mailbox and return matching messages.

        Use this tool first to find a message before reading it in full. Results
        come back newest first, each with a message_id you can pass to
        read_mailbox_message. Message bodies in these results are shortened —
        read a specific message when the summary is not enough to answer.

        On Gmail the query accepts Gmail's own search syntax, so
        "from:alice newer_than:7d" and "has:attachment invoice" both work. Omit
        the query to see the most recent messages.

        A mailbox that is unreachable or whose password has stopped working
        comes back with a status field explaining which — report that plainly
        instead of saying there is no mail.

        Args:
            query: What to search for. Omit for the most recent messages.
            limit: Maximum number of messages to return.
            account_label: Which mailbox to search. Only needed when the owner
                has connected more than one.
        """
        from src.anubis.utils.tools.email.imap_client import search_messages

        record, error = _select_account(account_label)
        if error is not None:
            return error
        capped = max(1, min(int(limit or 10), fetch_limit))
        result = await _run(record, search_messages, query, capped)
        if isinstance(result, dict):
            return result
        return {
            "account_label": record.get("display_label"),
            "message_count": len(result),
            "messages": result,
        }

    @tool
    async def read_mailbox_message(
        message_id: str,
        account_label: str | None = None,
    ) -> dict[str, Any]:
        """Read one message in full, using an id from search_mailbox_messages.

        Returns the sender, recipients, subject, date, body text, and every link
        the message contains.

        Args:
            message_id: The message_id reported by search_mailbox_messages.
            account_label: Which mailbox the message is in. Only needed when the
                owner has connected more than one.
        """
        from src.anubis.utils.tools.email.imap_client import fetch_message

        record, error = _select_account(account_label)
        if error is not None:
            return error
        result = await _run(record, fetch_message, str(message_id))
        if isinstance(result, dict) and result.get("status"):
            return result
        if result is None:
            return {
                "error": (
                    f"No message with id {message_id!r} is in "
                    f"{record.get('display_label')} any more."
                )
            }
        return {"account_label": record.get("display_label"), "message": result}

    @tool
    async def read_mailbox_thread(
        thread_id: str,
        account_label: str | None = None,
    ) -> dict[str, Any]:
        """Read a whole conversation, oldest message first.

        Use this when a single message refers to an earlier exchange and you
        need the context before answering.

        Args:
            thread_id: The thread_id reported by search_mailbox_messages.
            account_label: Which mailbox the conversation is in. Only needed
                when the owner has connected more than one.
        """
        from src.anubis.utils.tools.email.imap_client import fetch_thread

        record, error = _select_account(account_label)
        if error is not None:
            return error
        result = await _run(record, fetch_thread, str(thread_id))
        if isinstance(result, dict):
            return result
        return {
            "account_label": record.get("display_label"),
            "message_count": len(result),
            "messages": result,
        }

    @tool
    async def draft_mailbox_reply(
        to: str,
        subject: str,
        body: str,
        in_reply_to: str | None = None,
        account_label: str | None = None,
    ) -> dict[str, Any]:
        """Save a draft reply in the owner's mailbox. This never sends anything.

        The draft is written to the mailbox's drafts folder for the owner to
        review and send themselves. You cannot send email — say so plainly if
        asked to, and tell the owner the draft is waiting in their drafts folder.

        Write the draft in the owner's own voice. Pass the original message's
        rfc822_message_id as in_reply_to so the draft threads under the
        conversation it answers.

        Args:
            to: Recipient address.
            subject: Subject line.
            body: The message text, in the owner's voice.
            in_reply_to: The rfc822_message_id of the message being answered.
            account_label: Which mailbox to draft from. Only needed when the
                owner has connected more than one.
        """
        from src.anubis.utils.tools.email.imap_client import append_draft

        record, error = _select_account(account_label)
        if error is not None:
            return error

        def _append(credentials: Any) -> dict[str, Any]:
            return append_draft(
                credentials,
                to_address=to,
                subject=subject,
                body_text=body,
                in_reply_to=in_reply_to,
            )

        result = await _run(record, _append)
        if isinstance(result, dict) and result.get("status") == "draft_saved":
            result["account_label"] = record.get("display_label")
        return result

    return [
        search_mailbox_messages,
        read_mailbox_message,
        read_mailbox_thread,
        draft_mailbox_reply,
    ]
