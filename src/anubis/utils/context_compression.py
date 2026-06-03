"""Token budgeting, rolling conversation summarization, and map-reduce for oversized inputs."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.tokenizer import count_tokens, split_text_by_token_budget

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """<ROLE>
You compress prior dialogue for another assistant that will continue the conversation.
</ROLE>
<INSTRUCTIONS>
Produce a dense factual summary. Preserve:
- explicit decisions, commitments, deadlines, names, numbers, URLs, and technical identifiers
- open questions, blockers, and stated user goals
- definitions and constraints the user gave
Do not invent facts. Omit pleasantries and repetition.
Use clear sections with short headings if helpful. No preamble or meta-commentary about summarizing.
</INSTRUCTIONS>"""

_MERGE_SYSTEM = """<ROLE>
You merge partial summaries of one long text into one coherent summary.
</ROLE>
<INSTRUCTIONS>
Preserve factual detail; deduplicate overlap. No preamble.
</INSTRUCTIONS>"""


def effective_prompt_budget(context: GlobalContext) -> int:
    """Maximum tokens allowed for system + messages + internal_thoughts combined."""
    raw = context.model_token_limit - context.context_completion_reserve_tokens
    return max(8_192, raw)


def message_content_as_string(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type")
                if t in ("text", "input_text"):
                    tx = block.get("text")
                    if isinstance(tx, str):
                        parts.append(tx)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def message_to_budget_line(msg: BaseMessage) -> str:
    label = type(msg).__name__
    body = message_content_as_string(msg.content)
    return f"{label}:{body}\n"


def estimate_messages_token_count(messages: list[BaseMessage]) -> int:
    return sum(count_tokens(message_to_budget_line(m)) for m in messages)


def estimate_total_prompt_tokens(
    system_messages: list[SystemMessage],
    messages: list[BaseMessage],
    internal_thoughts: list[BaseMessage],
) -> int:
    return (
        estimate_messages_token_count(list(system_messages))
        + estimate_messages_token_count(list(messages))
        + estimate_messages_token_count(list(internal_thoughts))
    )


def truncate_string_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate ``text`` from the end so total tokens stay under ``max_tokens``."""
    if max_tokens <= 0:
        return text
    if count_tokens(text) <= max_tokens:
        return text
    note = (
        "\n\n[Context truncated for length; episodic and identity stores may still "
        "hold material not shown here.]\n"
    )
    note_tok = count_tokens(note)
    lo, hi = 0, len(text)
    target = max_tokens - note_tok
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return text[:best] + note


def apply_removals(messages: list[BaseMessage], removals: list[RemoveMessage]) -> list[BaseMessage]:
    remove_ids = {r.id for r in removals if getattr(r, "id", None)}
    return [m for m in messages if getattr(m, "id", None) not in remove_ids]


async def _invoke_summarizer(instruction: str, body: str, context: GlobalContext) -> str:
    _ = context
    model = init_model(tools=[], tool_choice="auto", response_format=None)
    resp = await model.ainvoke(
        [
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=f"{instruction}\n\n{body}"),
        ]
    )
    return message_content_as_string(getattr(resp, "content", ""))


async def _invoke_merge(chunks: list[str], context: GlobalContext) -> str:
    _ = context
    model = init_model(tools=[], tool_choice="auto", response_format=None)
    body = "\n\n---\n\n".join(chunks)
    resp = await model.ainvoke(
        [
            SystemMessage(content=_MERGE_SYSTEM),
            HumanMessage(content=body),
        ]
    )
    return message_content_as_string(getattr(resp, "content", ""))


async def map_reduce_large_human_text(text: str, context: GlobalContext) -> str:
    """Summarize a single oversized string via map-reduce."""
    if not text.strip():
        return text
    chunk_size = max(4096, context.map_reduce_chunk_max_tokens)
    while True:
        chunks = split_text_by_token_budget(text, chunk_size)
        if len(chunks) <= context.context_summarization_max_chunks:
            break
        chunk_size = int(chunk_size * 1.35)
        if chunk_size > context.model_token_limit:
            chunks = split_text_by_token_budget(text, chunk_size)
            chunks = chunks[: context.context_summarization_max_chunks]
            logger.warning(
                "map_reduce: still %s chunks after raising chunk_size; hard-capping",
                len(chunks),
            )
            break

    partials: list[str] = []
    for i, ch in enumerate(chunks):
        partials.append(
            await _invoke_summarizer(
                f"Summarize excerpt {i + 1}/{len(chunks)}.",
                ch,
                context,
            )
        )

    while len(partials) > 1:
        nxt: list[str] = []
        for j in range(0, len(partials), 8):
            group = partials[j : j + 8]
            nxt.append(await _invoke_merge(group, context))
        partials = nxt
    return partials[0]


async def summarize_message_prefix_for_budget(
    messages: list[BaseMessage],
    context: GlobalContext,
) -> tuple[list[RemoveMessage], list[HumanMessage]]:
    """Summarize all but the last ``conversation_verbatim_tail_messages`` messages."""
    tail = max(1, context.conversation_verbatim_tail_messages)
    if len(messages) <= tail:
        return [], []

    prefix = messages[:-tail]
    removals: list[RemoveMessage] = []
    for m in prefix:
        mid = getattr(m, "id", None)
        if mid:
            removals.append(RemoveMessage(id=mid))
    if not removals and prefix:
        logger.warning(
            "context_compression: prefix has %s messages but none have ids; skipping removal",
            len(prefix),
        )
        return [], []

    combined = "\n\n".join(message_to_budget_line(m) for m in prefix)
    summary_body = await _invoke_summarizer(
        "Summarize the following prior conversation turns for continuation.",
        combined,
        context,
    )
    labeled = (
        "[Prior conversation summary — verbatim tail follows; full history may exist in storage]\n\n"
        + summary_body
    )
    summary_msg = HumanMessage(id=str(uuid.uuid4()), content=labeled)
    return removals, [summary_msg]


def _find_last_human_index(messages: list[BaseMessage]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return None


async def maybe_compress_oversized_last_human(
    messages: list[BaseMessage],
    context: GlobalContext,
    budget: int,
) -> tuple[list[RemoveMessage], list[HumanMessage]]:
    """If the last HumanMessage is huge, replace it with a summarized version."""
    idx = _find_last_human_index(messages)
    if idx is None:
        return [], []
    hm = messages[idx]
    if not isinstance(hm, HumanMessage):
        return [], []
    body = message_content_as_string(hm.content)
    line_tok = count_tokens(message_to_budget_line(hm))
    if line_tok <= max(budget // 4, 50_000):
        return [], []

    new_body = await map_reduce_large_human_text(body, context)
    old_id = getattr(hm, "id", None)
    if not old_id:
        logger.warning("context_compression: last HumanMessage has no id; cannot replace in place")
        return [], []
    return (
        [RemoveMessage(id=old_id)],
        [HumanMessage(id=old_id, content=new_body)],
    )


async def ensure_context_fits_budget_impl(
    system_messages: list[SystemMessage],
    messages: list[BaseMessage],
    internal_thoughts: list[BaseMessage],
    context: GlobalContext,
) -> dict[str, Any]:
    """
    Return graph update dict with ``messages`` and/or ``internal_thoughts`` deltas.
    """
    budget = effective_prompt_budget(context)
    cur_messages = list(messages)
    cur_internal = list(internal_thoughts)
    msg_delta: list[Any] = []
    int_delta: list[Any] = []

    def current_total() -> int:
        return estimate_total_prompt_tokens(system_messages, cur_messages, cur_internal)

    if current_total() <= budget:
        return {}

    r1, a1 = await summarize_message_prefix_for_budget(cur_messages, context)
    if r1 or a1:
        cur_messages = apply_removals(cur_messages, r1) + a1
        msg_delta.extend(r1 + a1)

    if current_total() <= budget:
        return _pack_deltas(msg_delta, int_delta)

    r2, a2 = await maybe_compress_oversized_last_human(cur_messages, context, budget)
    if r2 or a2:
        cur_messages = apply_removals(cur_messages, r2) + a2
        msg_delta.extend(r2 + a2)

    if current_total() <= budget:
        return _pack_deltas(msg_delta, int_delta)

    if cur_internal and len(cur_internal) > 6:
        keep = cur_internal[-6:]
        drop = cur_internal[:-6]
        for m in drop:
            mid = getattr(m, "id", None)
            if mid:
                int_delta.append(RemoveMessage(id=mid))
        cur_internal = keep

    if current_total() <= budget:
        return _pack_deltas(msg_delta, int_delta)

    idx = _find_last_human_index(cur_messages)
    if idx is not None:
        hm = cur_messages[idx]
        if isinstance(hm, HumanMessage) and getattr(hm, "id", None):
            body = message_content_as_string(hm.content)
            cap = max(4_096, int(budget * 0.55))
            new_body = truncate_string_to_token_limit(body, cap)
            msg_delta.extend(
                [RemoveMessage(id=hm.id), HumanMessage(id=hm.id, content=new_body)]
            )
            cur_messages = apply_removals(cur_messages, [RemoveMessage(id=hm.id)]) + [
                HumanMessage(id=hm.id, content=new_body)
            ]

    return _pack_deltas(msg_delta, int_delta)


def _pack_deltas(msg_delta: list[Any], int_delta: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if msg_delta:
        out["messages"] = msg_delta
    if int_delta:
        out["internal_thoughts"] = int_delta
    return out
