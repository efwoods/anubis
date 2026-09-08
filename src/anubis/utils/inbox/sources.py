"""New items from ANY connected account, and replies back through the same account.

The inbox graph triages one message at a time and does not care where the
message came from. What differs per account is only how new items are found
and how a reply is delivered — and that difference is exactly what the
account's own tools already encode: a mailbox has search and send tools, a
signed-in site has open, read, find, click, and type tools, a vendor with an
API has list and post tools. So discovery and delivery are one small
tool-calling loop each, run with the tools of the ONE account concerned and a
fixed instruction, ending in a structured answer. Connecting a new kind of
account therefore adds triage automatically; no per-account code exists here.

Mailboxes keep their direct IMAP path (cheaper and exact); every other
connected account goes through the loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Accounts whose records hold no credential to act with (a device, a website
# by address, a bank) are not message sources.
NON_MESSAGE_MECHANISMS = frozenset({"device_pairing", "url_only", "plaid_link"})
NON_MESSAGE_KINDS = frozenset({"bank", "website", "data_source", "mcp_server"})


class DiscoveredItem(BaseModel):
    """One new item addressed to the owner, found on a connected account."""

    external_id: str = Field(description="A stable identifier of the item on the site or service (a message id, a post id, a URL). Never invent one; use the URL when nothing else exists.")
    sender: str = Field(description="Who the item is from (a name or handle).")
    subject: str = Field(default="", description="A short title or the first line.")
    body_text: str = Field(default="", description="The item's text, as read.")
    received_at: str | None = Field(default=None, description="When the item was received or posted, ISO 8601, when known.")
    url: str | None = Field(default=None, description="Where the item can be opened or replied to.")
    kind: str = Field(default="message", description="message, mention, comment, issue, review, invitation, or other.")
    reply_hint: str | None = Field(default=None, description="How a reply would be posted (which control, which tool).")


class DiscoveredItems(BaseModel):
    """Everything new found in one discovery pass."""

    items: list[DiscoveredItem] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list, description="Pages, folders, or endpoints that were checked.")
    note: str = Field(default="", description="What could not be read, in one sentence, or empty.")


class DeliveryResult(BaseModel):
    """Whether a reply was posted through the account."""

    sent: bool
    detail: str = Field(default="", description="What was done, or why the reply could not be posted.")
    url: str | None = Field(default=None, description="Where the reply can be seen, when known.")


def is_message_source(record: dict[str, Any]) -> bool:
    """Whether a connected account can carry messages the inbox should triage."""
    mechanism = str(record.get("credential_mechanism") or "")
    kind = str(record.get("kind") or "")
    if record.get("status") != "connected":
        return False
    if mechanism in NON_MESSAGE_MECHANISMS or kind in NON_MESSAGE_KINDS:
        return False
    return True


def _stable_external_id(item: DiscoveredItem) -> str:
    candidate = str(item.external_id or item.url or "").strip()
    if candidate:
        return candidate[:300]
    digest = hashlib.sha256(
        f"{item.sender}|{item.subject}|{item.body_text[:400]}".encode()
    ).hexdigest()
    return f"content:{digest[:32]}"


def item_to_message(item: DiscoveredItem, record: dict[str, Any]) -> dict[str, Any]:
    """Shape a discovered item like the message the inbox graph triages."""
    return {
        "message_id": _stable_external_id(item),
        "sender": item.sender or record.get("display_label") or "unknown",
        "recipients": [str(record.get("account_address") or "")],
        "subject": item.subject or (item.body_text[:80] if item.body_text else item.kind),
        "body_text": item.body_text,
        "sent_at": item.received_at or datetime.now(UTC).isoformat(),
        "thread_id": item.url,
        "url": item.url,
        "kind": item.kind,
        "reply_hint": item.reply_hint,
        "provider": record.get("provider"),
        "provider_label": record.get("display_label"),
    }


async def _account_tools(context: Any, store: Any, record: dict[str, Any]) -> list[Any]:
    from src.anubis.utils.connected_accounts.tool_factories import (
        build_tools_for_accounts,
    )

    return await build_tools_for_accounts(context, [record], store=store)


async def _run_tool_loop(
    context: Any,
    *,
    tools: list[Any],
    system_text: str,
    task_text: str,
    result_schema: type[BaseModel],
    max_steps: int,
    model: Any = None,
) -> BaseModel:
    """Run a bounded tool-calling loop and finish with one structured answer.

    The model may call the account's tools up to ``max_steps`` times; the
    transcript is then handed to a structured-output model that must answer
    in ``result_schema``. Tool results are truncated so one large page cannot
    blow the budget.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    from src.anubis.utils.model import (
        STRUCTURED_OUTPUT_STREAM_TAG,
        init_chat_model_unbound,
        init_model,
    )

    base_model = model or init_chat_model_unbound(context)
    bound = base_model.bind_tools(tools) if tools else base_model
    by_name = {getattr(tool, "name", ""): tool for tool in tools}
    messages: list[Any] = [SystemMessage(content=system_text), HumanMessage(content=task_text)]
    for _ in range(max(1, int(max_steps))):
        response = await bound.ainvoke(messages)
        messages.append(response)
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            break
        for call in tool_calls:
            name = str(call.get("name") or "")
            arguments = dict(call.get("args") or {})
            tool = by_name.get(name)
            if tool is None:
                output: Any = {"status": "error", "error": f"No tool named {name!r}."}
            else:
                try:
                    output = await tool.ainvoke(arguments)
                except Exception as tool_error:  # noqa: BLE001 - reported to the model
                    output = {"status": "error", "error": str(tool_error)[:500]}
            text = output if isinstance(output, str) else json.dumps(output, default=str)
            messages.append(ToolMessage(content=text[:16000], tool_call_id=str(call.get("id") or name), name=name))
    if isinstance(messages[-1], AIMessage) and not getattr(messages[-1], "tool_calls", None):
        pass
    structured = (
        model.with_structured_output(result_schema) if model is not None else init_model(context, response_format=result_schema)
    )
    if hasattr(structured, "with_config"):
        structured = structured.with_config(tags=[STRUCTURED_OUTPUT_STREAM_TAG])
    transcript = [
        *messages,
        HumanMessage(content=f"Now answer in the {result_schema.__name__} shape, using only what the tools returned."),
    ]
    result = await structured.ainvoke(transcript)
    if isinstance(result, result_schema):
        return result
    if isinstance(result, dict):
        return result_schema.model_validate(result)
    raise RuntimeError("The structured answer could not be read.")


def _discovery_system_text(record: dict[str, Any], owner_label: str) -> str:
    return (
        "You find NEW items addressed to the owner on one connected account, using only the "
        "tools given. The account: "
        f"{record.get('display_label')} ({record.get('provider')}). The owner: {owner_label}. "
        "Open the account's notifications, inbox, mentions, messages, comments, or issues "
        "pages (or call the listing tools) and read what arrived since the given time. "
        "Report only items addressed to or about the owner that arrived since that time, "
        "with a stable identifier and the text as read. Never post, click send, delete, or "
        "change anything; reading only. Never reveal a token, cookie, or password."
    )


async def discover_new_items(
    context: Any,
    store: Any,
    record: dict[str, Any],
    *,
    since: datetime | None,
    limit: int = 10,
    max_steps: int = 8,
    model: Any = None,
) -> tuple[list[dict[str, Any]], DiscoveredItems]:
    """Find new items on one non-mailbox account; return graph messages and the raw answer."""
    tools = await _account_tools(context, store, record)
    if not tools:
        return [], DiscoveredItems(note="The account offers no tools in this process.")
    since_text = since.isoformat() if since else "the last day"
    owner_label = str(record.get("account_address") or record.get("display_label") or "the owner")
    task = (
        f"Find new items since {since_text}. Return at most {int(limit)} items, newest first. "
        "List which pages or endpoints you checked."
    )
    answer = await _run_tool_loop(
        context,
        tools=tools,
        system_text=_discovery_system_text(record, owner_label),
        task_text=task,
        result_schema=DiscoveredItems,
        max_steps=max_steps,
        model=model,
    )
    if not isinstance(answer, DiscoveredItems):
        answer = DiscoveredItems()
    messages = [item_to_message(item, record) for item in answer.items[: int(limit)]]
    return messages, answer


async def deliver_reply_via_account(
    context: Any,
    store: Any,
    record: dict[str, Any],
    *,
    message: dict[str, Any],
    draft_text: str,
    max_steps: int = 8,
    model: Any = None,
) -> DeliveryResult:
    """Post a reply to one item through the account it arrived on."""
    tools = await _account_tools(context, store, record)
    if not tools:
        return DeliveryResult(sent=False, detail="The account offers no tools to reply with.")
    system_text = (
        "You post ONE reply on behalf of the owner through one connected account, using only "
        f"the tools given. The account: {record.get('display_label')} ({record.get('provider')}). "
        "Post exactly the reply text given, once, to the item described; do not post anything "
        "else and do not change other settings. If the item cannot be found or the site offers "
        "no way to reply, say so instead of posting elsewhere."
    )
    task = (
        f"Item: from {message.get('sender')!r}, subject {message.get('subject')!r}, "
        f"kind {message.get('kind') or 'message'}, at {message.get('url') or 'unknown address'}. "
        f"Reply hint: {message.get('reply_hint') or 'none'}.\n\nReply text to post, verbatim:\n"
        f"{draft_text}"
    )
    answer = await _run_tool_loop(
        context,
        tools=tools,
        system_text=system_text,
        task_text=task,
        result_schema=DeliveryResult,
        max_steps=max_steps,
        model=model,
    )
    return answer if isinstance(answer, DeliveryResult) else DeliveryResult(sent=False, detail="No answer.")
