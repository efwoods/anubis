"""The person's own words, read from the conversations already stored.

What somebody types or says to Neural Nexus is a direct quote of that person. It
is not a different kind of data from a quote lifted out of an uploaded interview,
and this module deliberately does not treat it as one: it yields the person's own
turns, and the same consumers that read an uploaded corpus read these.

**This module writes nothing, ever.** Every turn is already durable — the
LangGraph ``AsyncPostgresSaver`` checkpoints each thread, and every thread is
already indexed by who spoke in it (``thread_metadata`` carries ``user_id`` and
``assistant_id``, which ``threads.search`` filters on). Copying those turns into
the store would duplicate the whole conversational history of the platform, and
because ``langgraph.json`` auto-embeds ``document.kwargs.page_content``, each copy
would cost an embedding row as well. So the words are read where they already sit
and the derived artifacts — the style profile, the key phrases, the psychological
profile, a training file — are the only things anybody keeps.

Read-through is safe because compaction does not truncate a thread:
``AvatarSummarizationMiddleware`` keeps the deepagents behaviour of "non-mutating
compaction applied at model-call time" (``src/anubis/utils/middleware/avatar_summarization.py``).
Compaction shortens what the model is shown, never what the thread holds. If that
ever changes, this module silently starts seeing less history, which is why
``tests/unit_tests/test_speaker_turns.py`` pins the assumption.

What is deliberately NOT a quote of the person:

- ambient webcam and screen observations, which the person never typed;
- browser harvest turns — the ``[neural-nexus:conversation-suggestions]`` prompts
  the browser injects to ask for follow-up suggestions. Those are the product
  talking to itself, and treating them as the person's words would teach an
  avatar to answer in JSON;
- hidden turns this system authored on the person's behalf;
- turns from a group platform (Slack, Discord, Twitch), because the platforms'
  own terms forbid training a model on the people in a room;
- anything the avatar itself wrote or said, in any position that would train it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

GROUP_PLATFORM_NAMES = frozenset({"slack", "discord", "twitch"})

THREAD_SEARCH_PAGE_SIZE = 100
"""Explicit page size. ``threads.search`` defaults to ten and says nothing about it."""


@dataclass(frozen=True)
class SpeakerQuote:
    """One stretch of a person's own words, with what prompted the words.

    ``counterpart_text`` is what the other side said immediately before. For a
    typed conversation that is the avatar's preceding reply; for a spoken turn it
    is whatever the other person in the room said. Either way it is the question
    these words answered, which is what makes the pair usable as training data.
    """

    text: str
    message_id: str
    counterpart_text: str = ""
    spoken: bool = False
    speaker_label: str = ""


def _additional_kwargs_of(message: Any) -> dict[str, Any]:
    """Return a message's ``additional_kwargs`` whether object or serialized dict."""
    if isinstance(message, dict):
        return message.get("additional_kwargs") or {}
    return getattr(message, "additional_kwargs", None) or {}


def _is_human(message: Any) -> bool:
    """Whether ``message`` is a human turn, object or serialized dict."""
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "") in ("human", "user")
    return message.__class__.__name__ == "HumanMessage"


def _is_ai(message: Any) -> bool:
    """Whether ``message`` is an assistant turn, object or serialized dict."""
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "") in ("ai", "assistant")
    return message.__class__.__name__ == "AIMessage"


def _text_of(message: Any) -> str:
    """Return a message's plain text, flattening content blocks."""
    content = (
        message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _message_id_of(message: Any) -> str:
    """Return a message's identifier, or an empty string."""
    if isinstance(message, dict):
        return str(message.get("id") or "")
    return str(getattr(message, "id", "") or "")


def is_group_platform_turn(message: Any) -> bool:
    """Whether this turn was spoken in a Slack, Discord, or Twitch room.

    Those conversations reach the avatar through a different graph and a
    synthetic viewer identity, so they are already outside what this module
    enumerates. The check stands anyway, and is tested, because
    ``platform_policies.py`` commits the product to not training a model on the
    people in somebody else's room, and an accident of routing is not a
    safeguard.
    """
    additional_kwargs = _additional_kwargs_of(message)
    platform = str(additional_kwargs.get("platform") or "").strip().lower()
    if platform in GROUP_PLATFORM_NAMES:
        return True
    source_kind = str(additional_kwargs.get("source_kind") or "").strip().lower()
    return source_kind in GROUP_PLATFORM_NAMES


def is_quotable_human_turn(message: Any) -> bool:
    """Whether this turn is genuinely the person's own words.

    Everything excluded here is excluded because it is somebody or something
    else's text wearing a human turn's shape. See the module docstring.
    """
    from src.anubis.utils.ambient.observations import is_ambient_observation
    from src.anubis.utils.client_harvest_turns import is_client_harvest_turn

    if not _is_human(message):
        return False
    if is_ambient_observation(message):
        return False
    if is_client_harvest_turn(message):
        return False
    if is_group_platform_turn(message):
        return False
    additional_kwargs = _additional_kwargs_of(message)
    # A hidden turn this system authored on the person's behalf — a connection
    # acknowledgement, for instance. A spoken turn is hidden in some modes and IS
    # the person's words, so the presence of a speaker record wins.
    if additional_kwargs.get("hidden") and not additional_kwargs.get("speakers"):
        return False
    return True


def _quotes_from_spoken_turn(
    message: Any, *, counterpart_text: str, minimum_characters: int
) -> list[SpeakerQuote]:
    """Split one diarized utterance into the owner's words and everyone else's.

    Owner segments are the person's own words. Segments labelled as the avatar
    are the avatar's cloned voice picked up through a speaker in the room and are
    discarded outright — an avatar trained on its own playback is the collapse
    case this whole module exists to avoid. Other speakers are not quoted as the
    owner; their words serve only as the counterpart side of the dialogue.
    """
    from src.anubis.utils.voice.speakers import spoken_turn_of

    record = spoken_turn_of(message) or {}
    segments = [
        segment for segment in (record.get("segments") or []) if isinstance(segment, dict)
    ]
    if not segments:
        return []

    quotes: list[SpeakerQuote] = []
    preceding_other_text = counterpart_text
    message_id = _message_id_of(message)
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if segment.get("is_avatar"):
            # Never the person, never a counterpart: the avatar's own playback.
            continue
        if not segment.get("is_owner"):
            preceding_other_text = text
            continue
        if len(text) < minimum_characters:
            continue
        quotes.append(
            SpeakerQuote(
                text=text,
                message_id=f"{message_id}:{index}" if message_id else str(index),
                counterpart_text=preceding_other_text,
                spoken=True,
                speaker_label=str(segment.get("speaker") or ""),
            )
        )
    return quotes


def speaker_quotes(
    messages: list[Any], *, minimum_characters: int = 40
) -> list[SpeakerQuote]:
    """Return the person's own words from one thread, in the order spoken.

    ``minimum_characters`` is the floor below which a turn is not worth keeping:
    "ok" and "yes" are the person's words in the strictest sense and say nothing
    whatever about how the person speaks.
    """
    quotes: list[SpeakerQuote] = []
    counterpart_text = ""
    for message in messages or []:
        if _is_ai(message):
            counterpart_text = _text_of(message).strip()
            continue
        if not is_quotable_human_turn(message):
            continue
        if _additional_kwargs_of(message).get("speakers"):
            quotes.extend(
                _quotes_from_spoken_turn(
                    message,
                    counterpart_text=counterpart_text,
                    minimum_characters=minimum_characters,
                )
            )
            continue
        text = _text_of(message).strip()
        if len(text) < minimum_characters:
            continue
        quotes.append(
            SpeakerQuote(
                text=text,
                message_id=_message_id_of(message),
                counterpart_text=counterpart_text,
            )
        )
    return quotes


def role_rows_for_speaker(messages: list[Any]) -> list[dict[str, str]]:
    """Return the conversation with the roles a training row needs, not the ones a chat has.

    In a chat the person is the ``user`` and the avatar is the ``assistant``.
    Training the PERSON's avatar inverts that: the person's turns are what the
    avatar must learn to produce, so they become the ``assistant`` side, and the
    avatar's preceding reply becomes the ``user`` prompt that they answered.

    That does put model-written text on the prompt side, and the trade is
    deliberate: the prompt side is context a model conditions on, never text it
    learns to imitate. The completion side is the one that matters, and only
    human words may ever appear there — see
    :func:`assert_no_model_text_on_completion_side`.

    Consecutive same-role turns are merged, matching
    ``_build_adapter_dialogue_document`` in the media pipeline so a conversation
    row and an uploaded-dialogue row have the same shape.
    """
    rows: list[dict[str, str]] = []
    for message in messages or []:
        if _is_human(message):
            if not is_quotable_human_turn(message):
                continue
            role, text = "assistant", _text_of(message).strip()
        elif _is_ai(message):
            role, text = "user", _text_of(message).strip()
        else:
            continue
        if not text:
            continue
        if rows and rows[-1]["role"] == role:
            rows[-1]["content"] = f"{rows[-1]['content']}\n\n{text}"
            continue
        rows.append({"role": role, "content": text})
    return rows


def assert_no_model_text_on_completion_side(
    rows: list[dict[str, str]], model_texts: set[str]
) -> list[dict[str, str]]:
    """Drop any completion-side row whose text the model wrote, and say so.

    An avatar trained on another model's output drifts away from the person it
    reconstructs and towards the model — the failure has a name, and it is not
    recoverable by training harder. The guard is structural rather than a
    convention in a comment so that a later refactor of role assignment breaks
    loudly here instead of quietly poisoning a corpus.
    """
    normalized = {text.strip() for text in model_texts if text and text.strip()}
    kept: list[dict[str, str]] = []
    for row in rows:
        if row.get("role") == "assistant" and row.get("content", "").strip() in normalized:
            logger.warning(
                "Refusing a training row whose completion side holds model-written text."
            )
            continue
        kept.append(row)
    return kept


def model_texts_of(messages: list[Any]) -> set[str]:
    """Return every piece of text the avatar wrote in this thread."""
    return {
        _text_of(message).strip() for message in messages or [] if _is_ai(message)
    } - {""}


async def speaker_thread_ids(
    langgraph_client: Any,
    user_id: str,
    *,
    page_size: int = THREAD_SEARCH_PAGE_SIZE,
    maximum_threads: int = 1000,
) -> list[str]:
    """Return every thread this person has spoken in, across every avatar.

    Threads are already indexed by speaker, so this is a filter on data that
    exists rather than an index of our own. Note the scope: a person talking to
    somebody ELSE's public avatar still spoke those words, and those threads come
    back here too, because the words belong to the speaker and not to the avatar
    addressed.

    Paging is explicit on purpose. ``threads.search`` defaults to ``limit=10``
    and gives no sign that it truncated, which once silently reduced an account's
    forty-nine conversations to ten (see the note on ``GET /conversations`` in
    ``src/api/webapp.py``).
    """
    if langgraph_client is None or not user_id:
        return []
    thread_ids: list[str] = []
    offset = 0
    while len(thread_ids) < maximum_threads:
        try:
            page = await langgraph_client.threads.search(
                metadata={"thread_metadata": {"user_id": user_id}},
                limit=min(page_size, maximum_threads - len(thread_ids)),
                offset=offset,
                sort_by="updated_at",
                sort_order="desc",
            )
        except Exception as search_error:  # noqa: BLE001 - a reader never fails a caller
            logger.warning(
                "Could not list the threads of %s: %s", user_id, search_error
            )
            break
        if not page:
            break
        for thread in page:
            thread_id = str((thread or {}).get("thread_id") or "")
            if thread_id:
                thread_ids.append(thread_id)
        if len(page) < page_size:
            break
        offset += len(page)
    return thread_ids


CONVERSATION_CALIBRATION_MAX_THREADS = 200
"""Ceiling on threads read into one corpus pass, so a heavy account stays bounded."""

CONVERSATION_CALIBRATION_MAX_QUOTES = 2000
"""Ceiling on the person's turns fed into one corpus pass."""


async def speaker_thread_ids_from_engagement(store: Any, user_id: str) -> list[str]:
    """Return this person's threads using the index the platform already keeps.

    The engagement record written on every human turn already tracks
    ``conversation_thread_ids`` per avatar, so "which conversations has this
    person held" is answerable from the store alone — no software development kit
    client, no API key, and nothing new to maintain. That matters because the
    readers that need this answer (a background sweep, a scheduled analysis) hold
    neither.

    Threads are unioned across every avatar the person has spoken to, including
    other people's avatars: the words belong to whoever said them, not to the
    avatar addressed.
    """
    if store is None or not user_id:
        return []
    from src.anubis.utils.learning.namespaces import LEARNING_KIND_ENGAGEMENT

    try:
        namespaces = await store.alist_namespaces(prefix=(user_id,), limit=1000)
    except Exception as list_error:  # noqa: BLE001 - a reader never fails a caller
        logger.warning("Could not list the namespaces of %s: %s", user_id, list_error)
        return []

    thread_ids: list[str] = []
    seen: set[str] = set()
    for namespace in namespaces or []:
        namespace = tuple(namespace)
        if len(namespace) != 3 or namespace[2] != LEARNING_KIND_ENGAGEMENT:
            continue
        try:
            item = await store.aget(namespace, "engagement")
        except Exception:  # noqa: BLE001 - one unreadable record skips, never raises
            continue
        if item is None:
            continue
        value = getattr(item, "value", None)
        if value is None and isinstance(item, dict):
            value = item.get("value")
        record = (value or {}).get("value") if isinstance(value, dict) else None
        if not isinstance(record, dict):
            record = value if isinstance(value, dict) else {}
        for thread_id in record.get("conversation_thread_ids") or []:
            thread_id = str(thread_id or "")
            if thread_id and thread_id not in seen:
                seen.add(thread_id)
                thread_ids.append(thread_id)
        if len(thread_ids) >= CONVERSATION_CALIBRATION_MAX_THREADS:
            break
    return thread_ids[:CONVERSATION_CALIBRATION_MAX_THREADS]


def conversation_quote_documents(
    quotes: list[SpeakerQuote], *, thread_id: str, target_name: str = ""
) -> list[Any]:
    """Turn a person's own turns into ordinary quote Documents, in memory only.

    These carry exactly the metadata the media pipeline puts on a quote lifted
    from an uploaded interview, because a conversational quote is not a different
    kind of thing and nothing downstream should have to know which one it is
    holding. ``adapter_prompt`` is the counterpart turn, the same way
    ``_build_target_quote_documents_from_dialogue`` records the question a
    target's answer answered.

    **Nothing here is stored.** The Documents are built for one pass and dropped;
    the text they carry already lives in the thread they were read from. The
    identifier is a uuid5 of the thread and message so a later pass produces the
    same identifier for the same turn, which keeps the derived per-document
    feature dictionary from growing a second entry for a turn it already knows.
    """
    from uuid import NAMESPACE_URL, uuid5

    from langchain_core.documents import Document

    documents: list[Any] = []
    for quote in quotes:
        document_id = str(
            uuid5(NAMESPACE_URL, f"conversation:{thread_id}:{quote.message_id}")
        )
        documents.append(
            Document(
                page_content=quote.text,
                metadata={
                    "namespace": "quote",
                    "type": "text",
                    "source": "conversation",
                    "classified_situation": "tweets_or_quotes",
                    "is_target": True,
                    "target_name": target_name,
                    "adapter_prompt": quote.counterpart_text,
                    "spoken": quote.spoken,
                    "thread_id": thread_id,
                    "document_id": document_id,
                    "vectorstore_acceptable": False,
                    "adapter_acceptable": True,
                    "analysis_acceptable": True,
                    "synthetic": False,
                },
            )
        )
    return documents


async def read_speaker_corpus(
    store: Any,
    graph: Any,
    user_id: str,
    *,
    minimum_characters: int = 40,
    maximum_quotes: int = CONVERSATION_CALIBRATION_MAX_QUOTES,
) -> list[Any]:
    """Return this person's own words from every conversation, as Documents.

    Reads the threads the engagement index already names, pulls each one out of
    the checkpoint it is already stored in, keeps the person's own turns, and
    returns them as quote Documents held in memory. Nothing is written at any
    point, and the caller is expected to use the result and drop it.
    """
    thread_ids = await speaker_thread_ids_from_engagement(store, user_id)
    if not thread_ids:
        return []
    from src.anubis.utils.learning.bulk_learning import load_thread_messages

    documents: list[Any] = []
    for thread_id in thread_ids:
        if len(documents) >= maximum_quotes:
            break
        try:
            messages = await load_thread_messages(graph, thread_id)
        except Exception:  # noqa: BLE001 - one unreadable thread skips, never raises
            logger.debug("Could not read thread %s", thread_id, exc_info=True)
            continue
        if not messages:
            continue
        quotes = speaker_quotes(messages, minimum_characters=minimum_characters)
        if not quotes:
            continue
        documents.extend(
            conversation_quote_documents(quotes, thread_id=thread_id)[
                : maximum_quotes - len(documents)
            ]
        )
    return documents


__all__ = [
    "GROUP_PLATFORM_NAMES",
    "SpeakerQuote",
    "assert_no_model_text_on_completion_side",
    "conversation_quote_documents",
    "is_group_platform_turn",
    "is_quotable_human_turn",
    "model_texts_of",
    "read_speaker_corpus",
    "role_rows_for_speaker",
    "speaker_quotes",
    "speaker_thread_ids",
    "speaker_thread_ids_from_engagement",
]
