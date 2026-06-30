# src/anubis/utils/helper_functions

# Vectore store helper functions
import asyncio
import base64
import io
import logging
import math
import random
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns
from typing import Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.base import BaseStore
from moviepy import AudioFileClip, VideoFileClip
from openai import OpenAI
from openai.types.audio.transcription_diarized import TranscriptionDiarized
from PIL import Image

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.system_prompts import (
    TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION,
)

logger = logging.getLogger(__name__)

from pydantic import BaseModel


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries for the vectorstore.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """

    query_list = list(existing) + list(new)
    query_list = list(set(query_list))
    if len(query_list) > 10:
        return query_list[-10:]
    else:
        return query_list


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    # metadata = doc.metadata or {}
    # meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    # if meta:
    # meta = f" {meta}"
    # f"<document{meta}>\n{doc.page_content}\n</document>"
    return f"<document>\n{doc.page_content}\n</document>"


def format_docs(docs: list[Document] | None) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


############################  Doc Indexing State  #############################
import uuid
from typing import Any, Literal, Union

from langgraph.store.base import Item, SearchItem


# region agent log
def _agent_debug_log(
    message: str, data: dict | None = None, hypothesis_id: str = ""
) -> None:
    """Append one NDJSON line to the debug session log.

    Picks the in-container bind-mount path when present, falls back to the
    host workspace path otherwise. Best-effort: any failure is silently
    swallowed so instrumentation never affects production behavior.
    """
    try:
        import json as _json
        import os as _os
        import time as _time

        _container_path = "/deps/anubis/.cursor/debug-aaf3d3.log"
        _host_path = (
            "/home/user/gh/anubis-project/wt/f-psycho-analysis/.cursor/debug-aaf3d3.log"
        )
        _path = (
            _container_path if _os.path.isdir("/deps/anubis/.cursor") else _host_path
        )
        _payload = {
            "sessionId": "aaf3d3",
            "timestamp": int(_time.time() * 1000),
            "location": "src/anubis/utils/utility.py",
            "message": message,
            "data": data or {},
            "hypothesisId": hypothesis_id,
        }
        with open(_path, "a") as _f:
            _f.write(_json.dumps(_payload, default=str) + "\n")
    except Exception:
        pass


# endregion


def _doc_dedup_key(doc: Document) -> str:
    """Return a stable identity key for de-duplicating a Document in the buffer.

    Prefers the explicit ``document_id`` / top-level ``id`` so re-emitting the
    same cached doc (e.g. ``load_consciousness`` returning its identity lists on
    every tool-loop iteration) is idempotent; falls back to ``page_content`` for
    docs that carry no stable id.
    """
    meta = getattr(doc, "metadata", None) or {}
    return str(
        getattr(doc, "id", None)
        or meta.get("document_id")
        or meta.get("id")
        or doc.page_content
    )


def remove_docs_update(docs: Sequence[Document]) -> dict[str, Any]:
    """Build a :func:`reduce_docs` instruction that removes exactly ``docs``.

    A consumer (e.g. ``index_docs``) returns this instead of the blunt
    ``"delete"`` so it clears only the documents it actually processed, leaving
    un-processed docs — including any appended concurrently in the same
    superstep — on the buffer for the next pass.
    """
    return {"op": "remove", "keys": [_doc_dedup_key(d) for d in docs]}


def reduce_docs(
    existing: Sequence[Document] | None,
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        Sequence[SearchItem],
        dict[str, Any],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Append new documents onto the existing working buffer.

    This is an *append* reducer, not a replace: a node contributes only the new
    documents it produced and they are appended to whatever is already on the
    channel. A node never re-reads the existing list — which avoids re-queuing
    already-processed documents.

    Supported ``new`` values:
        * a sequence of Documents / dicts / strings / ``SearchItem``s — coerced
          to Documents and **appended** (de-duped by identity);
        * the literal ``"delete"`` — clears the **entire** buffer;
        * a removal instruction ``{"op": "remove", "keys": [...]}`` (see
          :func:`remove_docs_update`) — removes only the listed docs, leaving
          everything else. Prefer this over ``"delete"`` when other nodes may
          write the same channel in the same superstep.

    De-duplication by stable document identity (:func:`_doc_dedup_key`) keeps
    the append idempotent: re-emitting a doc already on the buffer does not grow
    it, while genuinely new docs accumulate.

    Args:
        existing: The docs already on the channel, if any.
        new: The instruction to apply (see above).
    """
    # region agent log
    _existing_len = len(existing) if existing is not None else 0
    _new_kind = type(new).__name__
    _new_summary: dict[str, Any] = {"kind": _new_kind}
    if isinstance(new, list):
        _new_summary["len"] = len(new)
        _new_summary["item_kinds"] = list({type(i).__name__ for i in new})[:5]
    elif isinstance(new, dict):
        _new_summary["op"] = new.get("op")
        _new_summary["keys_len"] = len(new.get("keys") or [])
    elif isinstance(new, str):
        _new_summary["str_value_short"] = new[:24]
    _agent_debug_log(
        "reduce_docs:enter",
        {"existing_len": _existing_len, "new": _new_summary},
        hypothesis_id="H1+H2+H3",
    )
    # endregion

    if new == "delete":
        # region agent log
        _agent_debug_log(
            "reduce_docs:branch=delete",
            {"existing_len": _existing_len, "result_len": 0},
            hypothesis_id="H2",
        )
        # endregion
        return []

    # Targeted removal: drop only the processed docs, keep the rest.
    if isinstance(new, dict) and new.get("op") == "remove":
        to_remove = set(new.get("keys") or [])
        _filtered = [d for d in (existing or []) if _doc_dedup_key(d) not in to_remove]
        # region agent log
        _agent_debug_log(
            "reduce_docs:branch=remove",
            {
                "existing_len": _existing_len,
                "remove_keys_len": len(to_remove),
                "result_len": len(_filtered),
            },
            hypothesis_id="H2",
        )
        # endregion
        return _filtered

    coerced: list[Document] = []
    if isinstance(new, str):
        coerced.append(
            Document(page_content=new, metadata={"document_id": str(uuid.uuid4())})
        )
    elif isinstance(new, list):
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(
                        page_content=item, metadata={"document_id": str(uuid.uuid4())}
                    )
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            elif isinstance(item, SearchItem) or isinstance(item, Item):
                page_content = (
                    getattr(item, "value", {})
                    .get("document", {})
                    .get("kwargs", {})
                    .get("page_content", "")
                )
                document_metadata = (
                    getattr(item, "value", {})
                    .get("document", {})
                    .get("kwargs", {})
                    .get("metadata", {})
                )
                coerced.append(
                    Document(page_content=page_content, metadata=document_metadata)
                )
            else:
                coerced.append(item)
    else:
        # region agent log
        _agent_debug_log(
            "reduce_docs:branch=unknown_new_type",
            {"existing_len": _existing_len, "new_kind": _new_kind},
            hypothesis_id="H2",
        )
        # endregion
        # Unknown ``new`` type: leave the buffer unchanged.
        return existing or []

    # Append new docs onto existing, dropping any whose identity is already
    # present so re-emitted docs do not accumulate.
    result: list[Document] = list(existing or [])
    seen = {_doc_dedup_key(doc) for doc in result}
    # region agent log
    _skip_count = 0
    _sample_skip_key: str | None = None
    _sample_added_key: str | None = None
    # endregion
    for doc in coerced:
        key = _doc_dedup_key(doc)
        if key in seen:
            # region agent log
            _skip_count += 1
            if _sample_skip_key is None:
                _sample_skip_key = key[:64]
            # endregion
            continue
        seen.add(key)
        # region agent log
        if _sample_added_key is None:
            _sample_added_key = key[:64]
        # endregion
        result.append(doc)
    # region agent log
    _agent_debug_log(
        "reduce_docs:branch=append_dedup",
        {
            "existing_len": _existing_len,
            "coerced_len": len(coerced),
            "result_len": len(result),
            "skip_count": _skip_count,
            "sample_skip_key": _sample_skip_key,
            "sample_added_key": _sample_added_key,
        },
        hypothesis_id="H3+H4+H5",
    )
    # endregion
    return result


from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from src.anubis.utils.context import GlobalContext


async def extract_user_id_assistant_id(config: RunnableConfig):
    user_state = {}
    assistant_state = {}

    user_id = config.get("configurable", {}).get("user_id", "")

    if user_id != "":
        user_state.update({"user_id": user_id})
    else:
        """anonymous_user_id is 'str(uuid5(NAMESPACE_URL, 'anonymous_user_id"""
        user_state.update({"user_id": "9977df19-9ceb-5f87-a130-55f6a6282069"})

    assistant_id = config.get("configurable", {}).get("assistant_id", "")

    if assistant_id != "":
        assistant_state.update({"assistant_id": assistant_id})
    else:
        raise Exception(
            "Assistant does not have an id from the context. Provide an assistant_id in config['configurable']['assistant_id']."
        )

    return user_state, assistant_state


async def configure_assistant_context(config: RunnableConfig, store: BaseStore):
    user_id, assistant_id = await extract_user_id_assistant_id(config)

    namespace = (user_id, assistant_id, "assistant_ctx")
    ai_context_item = await store.aget(namespace, key=assistant_id)
    logger.info(f"ai_context_item: {ai_context_item}")

    # Load/UPDATE AI SELF IDENTITY
    logger.info("item object breakpoint")

    # get the current assistant context as a dict

    configurable_assistant_ctx = config.get("configurable", {}).get(
        "assistant_ctx", None
    )

    if configurable_assistant_ctx is not None:
        if ai_context_item is not None:
            for key, value in configurable_assistant_ctx:
                if (value != "" and value != None) and key != "metadata":
                    ai_context_item.value["assistant_ctx"].update({key: value})
            if configurable_assistant_ctx.get("metadata", None) is not None:
                update_metadata = configurable_assistant_ctx.get("metadata")
                ai_context_item.value["assistant_ctx"]["metadata"].update(
                    update_metadata
                )

            await store.aput(
                namespace,
                key=assistant_id,
                value={"assistant_ctx": ai_context_item.value["assistant_ctx"]},
            )
        else:
            init_assistant_ctx = {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "name": configurable_assistant_ctx.get("name", ""),
                "description": configurable_assistant_ctx.get("description", ""),
                "metadata": configurable_assistant_ctx.get("metadata", {}),
            }
            await store.aput(
                namespace, key=assistant_id, value={"assistant_ctx": init_assistant_ctx}
            )
    else:
        if ai_context_item is None:
            init_assistant_ctx = {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "name": "",
                "description": "",
                "metadata": {},
            }
            await store.aput(
                namespace, key=assistant_id, value={"assistant_ctx": init_assistant_ctx}
            )

    ai_context_item = await store.aget(namespace, key=assistant_id)

    return ai_context_item


async def image_to_text(
    target_image_url: str,
    reference_image_url: Optional[str] = None,
):
    """
    Convert an image of a target to text.
    Describe the target individual to the best of your ability.
    args:
        target_image_url: base64 encoded string or a url to an image to describe.
        reference_image_url (Optional[str]): base64 encoded string or a url to an image.
            Expected to only have a single individual. Used to identify the target to describe in the target image.

    Returns:
        description (str): This is the description of the target with respect to the individual. The description is of the target from the FIRST PERSON PERSPECTIVE.
    """

    if reference_image_url is not None:
        if "." in reference_image_url:
            # url
            reference_message = {
                "type": "image_url",
                "image_url": {"url": reference_image_url},
            }
        else:
            # base 64 encoding
            reference_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{reference_image_url}"},
            }

    if "." in target_image_url:
        target_message = {"type": "image_url", "image_url": {"url": target_image_url}}
    else:
        target_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{target_image_url}"},
        }

    # Compile the message
    content = (
        [reference_message, target_message]
        if reference_image_url is not None
        else [target_message]
    )
    system_message = [
        SystemMessage(
            content=TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
        )
    ]
    human_message = [{"role": "user", "content": content}]

    messages = system_message + human_message

    # TODO: response_metrics_aggregation
    model = init_model()

    response = await model.ainvoke(input=messages)

    description = response.content

    return description


from typing import Optional

""" YOUTUBE HELPER FUNCTIONS """
import os
import re

import yt_dlp


async def download_transcript(
    url: str, lang: str = "en", auto_subs: bool = True, output_dir: Optional[str] = None
) -> str:
    """
    Download transcript/subtitles from a YouTube video.

    Args:
        url: YouTube video URL
        lang: Language code (e.g., 'en', 'es', 'fr')
        auto_subs: Fall back to auto-generated subtitles if manual not found
        output_dir: Directory to save files. Defaults to a fresh temp dir per call
            so concurrent downloads never pick up each other's ``.vtt`` files.

    Returns:
        Path to the downloaded subtitle file
    """
    # Each call gets an isolated directory: the batch pipeline downloads many
    # videos in parallel, and the previous "write to '.' then grab the first
    # *.vtt" approach raced (one video's transcript shadowing another's).
    out_dir = output_dir or tempfile.mkdtemp(prefix="yt_subs_")

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": auto_subs,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
    }

    # yt_dlp.YoutubeDL is a *synchronous* context manager and extract_info blocks,
    # so run it off the event loop in a worker thread (the same pattern used by
    # the audio-download and playlist-extraction paths).
    def _extract() -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "video")

    title = await asyncio.to_thread(_extract)

    # Find the downloaded .vtt file in this call's own directory.
    for f in os.listdir(out_dir):
        if f.endswith(".vtt"):
            return os.path.join(out_dir, f)

    raise FileNotFoundError(
        f"No subtitle file found for '{title}'. Try listing available languages first."
    )


def parse_vtt(vtt_path: str) -> str:
    """
    Parse a .vtt subtitle file into clean plain text.
    Removes timestamps, cue settings, and duplicate lines.
    """
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove WEBVTT header and NOTE blocks
    content = re.sub(r"WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    content = re.sub(r"NOTE.*?\n\n", "", content, flags=re.DOTALL)

    lines = content.splitlines()
    clean_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        # Skip timestamps, cue IDs, and empty lines
        if not line or "-->" in line or re.match(r"^\d+$", line):
            continue
        # Remove inline tags like <00:00:01.000>, <c>, </c>
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        # Deduplicate consecutive repeated lines (common in auto-subs)
        if line and line not in seen:
            clean_lines.append(line)
            seen.add(line)
        elif line in seen:
            seen = {line}  # reset window to allow repeated words in new context

    return " ".join(clean_lines)


def list_available_subtitles(url: str) -> dict:
    """List all available subtitle languages for a video."""
    ydl_opts = {"quiet": True, "skip_download": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    manual = info.get("subtitles", {})
    auto = info.get("automatic_captions", {})

    print(f"\nVideo: {info.get('title')}")
    print(f"Manual subtitles:    {list(manual.keys()) or 'None'}")
    print(f"Auto-generated subs: {list(auto.keys()) or 'None'}")

    return {"manual": manual, "automatic": auto}


# This needs to be in-memory rather than using file storage
async def get_transcript(url: str, lang: str = "en", save_txt: bool = True) -> str:
    """
    High-level function: download + parse transcript, return plain text.

    Args:
        url: YouTube video URL
        lang: Subtitle language code
        save_txt: If True, saves plain text transcript to disk

    Returns:
        Transcript as a plain text string
    """
    print(f"Downloading subtitles for: {url}")
    vtt_path = await download_transcript(url, lang=lang)

    print(f"Parsing: {vtt_path}")
    transcript = parse_vtt(vtt_path)

    if save_txt:
        txt_path = vtt_path.rsplit(".", 2)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved to: {txt_path}")

    return transcript


# --- Example usage ---
# if __name__ == "__main__":
#     VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

#     # 1. Check what's available
#     list_available_subtitles(VIDEO_URL)

#     # 2. Download and parse transcript
#     text = get_transcript(VIDEO_URL, lang="en", save_txt=True)

#     print("\n--- Transcript Preview ---")
#     print(text[:500])

""" AUDIO TRANSCRIPTION """


def _openai_client_for_speech(context: GlobalContext) -> OpenAI:
    key = context.openai_api_key or context.llm_provider_api_key
    if not key:
        msg = "Set `openai_api_key` / OPENAI_API_KEY or `llm_provider_api_key` for speech APIs."
        raise ValueError(msg)
    # max_retries=0: retries are handled by _speech_call_with_retry so we can
    # back off transient failures (rate_limit_exceeded / timeouts / 5xx) while
    # failing FAST on permanent ones (insufficient_quota / auth) instead of
    # burning the SDK's blind exponential backoff on an error that can't recover.
    return OpenAI(api_key=key, max_retries=0)


# Permanent OpenAI error codes — retrying cannot succeed, so surface immediately.
_NON_RETRYABLE_OPENAI_CODES = frozenset(
    {"insufficient_quota", "billing_hard_limit_reached", "access_terminated"}
)


def _speech_call_with_retry(make_call, context: GlobalContext, *, description: str):
    """Run a synchronous OpenAI speech call with durable backoff on transient errors.

    Retries ``rate_limit_exceeded`` (429), request timeouts, connection errors and
    5xx with exponential backoff + jitter, up to ``openai_speech_max_retries``.
    Permanent failures (``insufficient_quota`` and other billing/auth errors) are
    re-raised on the first occurrence so they surface as a real item error rather
    than being slow-retried or silently swallowed.
    """
    import openai

    max_retries = max(0, int(context.openai_speech_max_retries or 0))
    base = float(context.openai_speech_retry_base_seconds or 0.0)
    attempts = max_retries + 1
    last_exc: Exception | None = None

    for attempt in range(attempts):
        try:
            return make_call()
        except openai.RateLimitError as exc:
            code = getattr(exc, "code", None)
            if code in _NON_RETRYABLE_OPENAI_CODES:
                logger.error(
                    "%s: permanent OpenAI error (%s); not retrying: %s",
                    description,
                    code,
                    exc,
                )
                raise
            last_exc = exc
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            last_exc = exc
        except openai.APIStatusError as exc:
            # Retry only transient 5xx; other status errors (4xx) are caller bugs.
            if getattr(exc, "status_code", 0) < 500:
                raise
            last_exc = exc

        if attempt < attempts - 1:
            delay = base * (2**attempt) + random.uniform(0, base or 0.0)
            logger.warning(
                "%s: transient OpenAI failure (attempt %d/%d), retrying in %.2fs: %s",
                description,
                attempt + 1,
                attempts,
                delay,
                last_exc,
            )
            time.sleep(delay)

    # Exhausted all retries on a transient error.
    assert last_exc is not None
    raise last_exc


def _audio_mime_from_suffix(suffix: str) -> str:
    s = (suffix or "").lower()
    if s == ".mp3":
        return "audio/mpeg"
    if s in (".m4a", ".mp4", ".aac"):
        return "audio/mp4"
    if s == ".wav":
        return "audio/wav"
    if s == ".webm":
        return "audio/webm"
    return "audio/mp4"


def _truncate_reference_audio_path_if_long(path: str, context: GlobalContext) -> str:
    """If audio is longer than _REFERENCE_AUDIO_CLIP_MAX_S, keep the first segment and return a new .mp3 path."""
    clip = AudioFileClip(path)
    try:
        duration = float(clip.duration or 0.0)
        if duration <= context.reference_audio_clip_max_seconds:
            return path
        sub = clip.subclipped(0.0, context.reference_audio_clip_max_seconds)
        fd, out_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            sub.write_audiofile(out_path, logger=None)
        except Exception:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise
        finally:
            sub.close()
    finally:
        clip.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    return out_path


def _decode_base64_media_payload(payload: str) -> bytes:
    """Decode raw base64 or an RFC 2397 ``data:*;base64,...`` string to bytes."""
    s = (payload or "").strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    return base64.b64decode(s)


# 1. Save the bytes to a temp file
async def get_audio_duration_seconds(
    audio_base64: str, filename: Optional[str] = None
) -> float:
    raw = _decode_base64_media_payload(audio_base64)
    suffix = Path(filename or "audio.m4a").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
        temp_audio.write(raw)
        temp_audio.flush()

        # 2. Load with MoviePy
        with AudioFileClip(temp_audio.name) as clip:
            duration_seconds = float(clip.duration or 0.0)
        return duration_seconds


def _transcribe_one_segment_path(
    path: str,
    upload_filename: str,
    context: GlobalContext,
    client: OpenAI,
) -> dict:
    start = time_ns()
    with AudioFileClip(path) as clip:
        duration_seconds = float(clip.duration or 0.0)
    transcription_cost = duration_seconds * context.audio_transcription_price_per_minute
    model = context.audio_transcription_model or "whisper-1"
    with open(path, "rb") as audio_f:
        content = _speech_call_with_retry(
            lambda: client.audio.transcriptions.create(
                file=(upload_filename, audio_f),
                model=model,
                response_format="text",
            ),
            context,
            description=f"transcription({upload_filename})",
        )
    return {
        "text": content,
        "file_duration_s": duration_seconds,
        "total_cost": transcription_cost,
        "latency_ms": (time_ns() - start) / 1e6,
        "model": model,
        "inference_type": "transcription",
    }


def _transcribe_saved_path(
    path: str, upload_filename: str, context: GlobalContext
) -> dict:
    """Transcribe audio on disk; split by time when the file exceeds the Whisper 25 MiB limit.

    Fully synchronous (moviepy chunking + the blocking OpenAI speech client +
    ``time.sleep`` backoff). Callers MUST run it via ``asyncio.to_thread`` so it
    never blocks the event loop.
    """
    start_time = time_ns()
    size_bytes = os.path.getsize(path)
    client = _openai_client_for_speech(context)

    if size_bytes <= context.whisper_max_bytes:
        seg = _transcribe_one_segment_path(path, upload_filename, context, client)
        seg["latency_ms"] = (time_ns() - start_time) / 1e6
        return seg

    clip = AudioFileClip(path)
    try:
        duration = float(clip.duration or 0.0)
        n = max(2, math.ceil(size_bytes / context.chunk_source_bytes_target))
        chunk_dur = duration / n
        parts: list[str] = []
        total_cost = 0.0
        inner_latency = 0.0
        for i in range(n):
            t0 = i * chunk_dur
            t1 = duration if i == n - 1 else (i + 1) * chunk_dur
            sub = clip.subclipped(t0, t1)
            fd, chunk_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                sub.write_audiofile(chunk_path, logger=None)
                seg = _transcribe_one_segment_path(
                    chunk_path, f"chunk_{i}.mp3", context, client
                )
                parts.append(seg["text"])
                total_cost += seg["total_cost"]
                inner_latency += seg["latency_ms"]
            finally:
                sub.close()
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
        return {
            "text": " ".join(parts),
            "file_duration_s": duration,
            "total_cost": total_cost,
            "latency_ms": inner_latency,
            "whisper_chunk_count": n,
            "model": context.audio_transcription_model,
            "inference_type": "transcription",
        }
    finally:
        clip.close()


async def transcribe_audio(
    audio_base64: str,
    context: GlobalContext,
    filename: Optional[str] = None,
    reference_audio: bool = False,
    max_duration_seconds: Optional[float] = 9.0,
) -> dict:

    # Remove noise and isolate the vocals; if reference audio, truncate to 9 seconds:
    preprocessed_audio = await preprocess_audio(
        audio_base64,
        truncate_only=False,
        reference_audio=reference_audio,
        max_duration_seconds=max_duration_seconds,
    )
    audio_base64 = preprocessed_audio["audio_base64"]

    raw = _decode_base64_media_payload(audio_base64)

    # Update preprocessed filename to mp3 codec
    suffix = ".mp3"  # mp3 after preprocessing to mp3 codec

    filename = Path(filename).stem + ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        name = Path(filename or f"audio{suffix}").name
        result = await asyncio.to_thread(_transcribe_saved_path, path, name, context)
        result["audio_base64_preprocessed"] = audio_base64
        return result

    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def get_file_size_MB(audio_base64: str) -> float:
    raw = _decode_base64_media_payload(audio_base64)
    return len(raw) / 1048576


async def preprocess_audio(
    audio_base64: str,
    truncate_only: bool,
    reference_audio: bool,
    filename: Optional[str] = None,
    max_duration_seconds: Optional[float] = 9.0,
    enhance_vocals_and_remove_noise: Optional[bool] = False,
) -> dict:
    """Preprocess audio and return an MP3 ``data:`` URI.
    Enhance vocals and remove noise:
    - noisereduce: spectral-gating noise reduction (suppresses background
      noise while preserving the voice).

    Modes:
      * ``truncate_only=True``: skip enhancement and energy detection; keep the
        first ``max_duration_seconds`` of the input as-is.

      * ``truncate_only=False``: run ``noisereduce`` spectral-gating noise
        reduction to suppress background noise and isolate the voice. If
        ``reference_audio=True`` and the cleaned audio is longer than
        ``max_duration_seconds``, select the contiguous window of that length
        with the greatest short-time energy. If ``reference_audio=False`` the
        cleaned audio is returned without further trimming.

    Input is expected to be MP3 (preprocessed audio) and the output is always
    re-encoded to MP3.

    When ``context.dev == "TRUE"`` the final MP3 is also written to
    ``tempfile.gettempdir()`` for inspection; the path is returned under
    ``saved_dev_path``.

    The ``truncate_only`` path uses only ``moviepy`` (already loaded at module
    import). ``numpy``, ``noisereduce``, ``librosa``, and ``soundfile`` are
    imported lazily inside the enhancement branch and are not paid for on the
    fast path.
    """
    # TODO: This function needs to separate the noise removal and the energy estimation of the audio file. The largest energy in the audio file may contain noise or multiple speakers. This requires VAD afterwards. This function will only perform the following: ensure the format is mp3, the noise is optionally removed and voice enhanced, and the entire duration is clipped to max seconds.
    #
    #
    #  Either implement custom VAD in a separate function after this or transcribe all the audio, truncate using the VAD from the diarization in combination with energy analysis (transcription must match the segment that is truncated for reference)

    raw = _decode_base64_media_payload(audio_base64)
    in_suffix = Path(filename or "audio.mp3").suffix or ".mp3"
    max_seconds = float(max_duration_seconds or 9.0)

    src_path: Optional[str] = None
    wav_path: Optional[str] = None
    out_mp3_path: Optional[str] = None
    saved_dev_path: Optional[str] = None

    try:
        # Convert to mp3 codec and truncate reference audio to max length
        with tempfile.NamedTemporaryFile(suffix=in_suffix, delete=False) as tmp_in:
            tmp_in.write(raw)
            tmp_in.flush()
            src_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            out_mp3_path = tmp_mp3.name

        sample_rate: Optional[int] = None

        # moviepy + ffmpeg are fully synchronous and CPU/IO-bound. Run them in a
        # worker thread so the event loop stays free (otherwise a long encode
        # blocks every other request — including auth — and starves cancellation).
        def _convert() -> tuple[float, Optional[int]]:
            clip = AudioFileClip(src_path)
            try:
                full_duration = float(clip.duration or 0.0)

                if reference_audio and truncate_only and full_duration > max_seconds:
                    sub = clip.subclipped(0, max_seconds)
                    local_duration = max_seconds
                else:
                    sub = clip
                    local_duration = full_duration
                try:
                    sub.write_audiofile(out_mp3_path, codec="mp3", logger=None)
                finally:
                    if sub is not clip:
                        sub.close()
                fps = getattr(clip, "fps", None)
                local_sample_rate = int(fps) if fps else None
            finally:
                clip.close()
            return local_duration, local_sample_rate

        duration_seconds, sample_rate = await asyncio.to_thread(_convert)
        # else: # Noise reduction and ENERGY clipping
        #     import librosa
        #     import noisereduce as nr
        #     import numpy as np
        #     import soundfile as sf

        #     # noisereduce: spectral-gating noise reduction (suppresses background
        #     # noise while preserving the voice). Decode the input MP3 to a mono
        #     # float waveform with ffmpeg (moviepy) — libsndfile/librosa can't
        #     # reliably decode compressed containers — then re-encode the cleaned
        #     # audio back to MP3. moviepy is already loaded; numpy/noisereduce/
        #     # librosa/soundfile are imported lazily so the truncate-only path
        #     # doesn't pay for them.
        #     in_clip = AudioFileClip(src_path)
        #     try:
        #         sample_rate = int(in_clip.fps or 44100)
        #         samples = in_clip.to_soundarray(fps=sample_rate)
        #     finally:
        #         in_clip.close()
        #     if samples.ndim == 2:
        #         samples = samples.mean(axis=1)
        #     mono = np.ascontiguousarray(samples, dtype=np.float32)

        #     # Enhance vocals and remove noise
        #     if enhance_vocals_and_remove_noise:
        #         logger.info(f"preprocess_audio noisereduce sr: {sample_rate}")
        #         waveform = nr.reduce_noise(y=mono, sr=sample_rate).astype(np.float32)
        #     else:
        #         waveform = mono

        #     # Identify the longest window of audio that contains potential speech
        #     duration_seconds = waveform.shape[-1] / float(sample_rate)

        #     if reference_audio and duration_seconds > max_seconds:
        #         frame_length = 2048
        #         hop_length = 512
        #         rms = librosa.feature.rms(
        #             y=waveform, frame_length=frame_length, hop_length=hop_length
        #         )[0]
        #         window_frames = max(1, int(round((max_seconds * sample_rate) / hop_length)))
        #         if window_frames < len(rms):
        #             # Sliding-window energy via cumulative sum on rms^2.
        #             energy = rms.astype(np.float64) ** 2
        #             csum = np.concatenate(([0.0], np.cumsum(energy)))
        #             window_sums = csum[window_frames:] - csum[:-window_frames]
        #             start_frame = int(np.argmax(window_sums))
        #             start_sample = start_frame * hop_length
        #             end_sample = min(
        #                 start_sample + int(round(max_seconds * sample_rate)),
        #                 waveform.shape[-1],
        #             )
        #             waveform = waveform[start_sample:end_sample]

        #     duration_seconds = waveform.shape[-1] / float(sample_rate)

        #     # Create MP3 Codec
        #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        #         wav_path = tmp_wav.name
        #     sf.write(wav_path, waveform, int(sample_rate))
        #     clip = AudioFileClip(wav_path)
        #     try:
        #         clip.write_audiofile(out_mp3_path, codec="mp3", logger=None)
        #     finally:
        #         clip.close()

        # Create base64 encoded MP3
        with open(out_mp3_path, "rb") as mp3_f:
            mp3_bytes = mp3_f.read()
        out_b64 = f"data:audio/mp3;base64,{base64.b64encode(mp3_bytes).decode('utf-8')}"

        # Dev Testing: uncomment to save the preprocessed audio to a file

        # stem = Path(filename or "audio").stem or "audio"
        # tag = "truncated" if truncate_only else ("ref" if reference_audio else "vocals")
        # dev_name = f"preprocess_{tag}_{stem}_{time_ns()}.mp3"
        # saved_file_path = str(Path(tempfile.gettempdir()) / dev_name)
        # with open(saved_file_path, "wb") as f:
        #     f.write(mp3_bytes)
        # logger.info(f"preprocess_audio dev artifact saved: {saved_file_path}")

        return {
            "audio_base64": out_b64,
            "duration_seconds": duration_seconds,
            "sample_rate": int(sample_rate) if sample_rate else None,
            "reference_audio": reference_audio,
            "truncate_only": truncate_only,
        }
    finally:
        for p in (src_path, wav_path, out_mp3_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


""" AUDIO DIARIZATION """

_AUDIO_UPLOAD_SUFFIXES = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".aiff",
        ".aif",
        ".wma",
    }
)
_VIDEO_UPLOAD_SUFFIXES = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
    }
)


def _upload_is_audio_for_diarize(
    filename: Optional[str], content_type: Optional[str]
) -> bool:
    suffix = Path(filename or "").suffix.lower()
    if suffix in _AUDIO_UPLOAD_SUFFIXES:
        return True
    if suffix in _VIDEO_UPLOAD_SUFFIXES:
        return False
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("audio/"):
        return True
    if ct.startswith("video/"):
        return False
    return False


def _diarize_known_speaker_extra_body(
    context: GlobalContext,
    encoded_reference_audio: Optional[str],
) -> Optional[dict]:
    if not encoded_reference_audio or not str(encoded_reference_audio).strip():
        return None
    return {
        "known_speaker_names": [context.audio_diarization_known_speaker_name],
        "known_speaker_references": [encoded_reference_audio],
    }


def _diarize_usage_tokens_dict(u: object) -> dict:
    if u is None:
        return {}
    d = u if isinstance(u, dict) else u.model_dump()
    if d.get("type") != "tokens":
        return {}
    return d


def _diarize_token_cost(usage_dict: dict, context: GlobalContext) -> float:
    u = usage_dict or {}
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    return (
        inp * context.audio_diarization_price_per_million_tokens_input
        + out * context.audio_diarization_price_per_million_tokens_output
    )


def _diarize_one_mp3_path(
    mp3_path: str,
    upload_name: str,
    context: GlobalContext,
    client: OpenAI,
    encoded_reference_audio: Optional[str],
) -> TranscriptionDiarized:
    model = context.audio_diarization_model or "gpt-4o-transcribe-diarize"
    extra = _diarize_known_speaker_extra_body(context, encoded_reference_audio)
    with open(mp3_path, "rb") as audio_f:
        if extra:
            result = _speech_call_with_retry(
                lambda: client.audio.transcriptions.create(
                    model=model,
                    file=(upload_name, audio_f),
                    response_format="diarized_json",
                    chunking_strategy="auto",
                    extra_body=extra,
                ),
                context,
                description=f"diarization({upload_name})",
            )
        else:
            result = _speech_call_with_retry(
                lambda: client.audio.transcriptions.create(
                    model=model,
                    file=(upload_name, audio_f),
                    response_format="diarized_json",
                    chunking_strategy="auto",
                ),
                context,
                description=f"diarization({upload_name})",
            )
    if not isinstance(result, TranscriptionDiarized):
        msg = "Expected diarized_json transcription response from OpenAI."
        raise TypeError(msg)
    return result


def _merge_diarized_segments_from_chunks(
    chunk_responses: list[TranscriptionDiarized],
    time_offsets: list[float],
) -> list[dict]:
    """Flatten chunked diarization responses onto the original timeline.

    Segment timestamps are shifted by each chunk's start offset so they
    reference the source audio. Speaker labels are kept verbatim — the
    diarizer assigns labels per call, so the same raw label (e.g.
    ``speaker_0``) across chunks usually refers to the most-prominent voice
    in each chunk and is therefore the same person for interview-style
    content. Callers that need strict cross-chunk speaker identity should
    pass an ``encoded_reference_audio`` to ``transcribe_audio_diarize`` so
    the diarizer labels the target with the known-speaker name (e.g.
    ``"avatar"``) in every chunk. The ``chunk_idx`` field is kept on each
    segment for diagnostic / debugging use.
    """
    merged: list[dict] = []
    for idx, (resp, t0) in enumerate(zip(chunk_responses, time_offsets, strict=True)):
        for seg in resp.segments:
            sd = seg.model_dump()
            merged.append(
                {
                    **sd,
                    "start": float(sd["start"]) + t0,
                    "end": float(sd["end"]) + t0,
                    "speaker": str(sd.get("speaker") or "unknown"),
                    "chunk_idx": idx,
                }
            )
    return merged


def extract_video_audio_b64(
    video_base64: str, filename: Optional[str] = None
) -> tuple[str, str]:
    """Extract a video's audio track as an mp3 data URI.

    Returns ``(data_uri, mp3_filename)`` where ``mp3_filename`` is the
    original filename's stem with an ``.mp3`` suffix (defaults to
    ``upload.mp3``). Caller is responsible for the returned URI; the
    intermediate files are cleaned up here.
    """
    raw = _decode_base64_media_payload(video_base64)
    orig_name = filename or "upload.mp4"
    src_suffix = Path(orig_name).suffix or ".mp4"
    source_path: Optional[str] = None
    audio_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=src_suffix, delete=False
        ) as temp_upload:
            temp_upload.write(raw)
            temp_upload.flush()
            source_path = temp_upload.name

        video = VideoFileClip(source_path)
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name
            video.audio.write_audiofile(audio_path, codec="mp3", logger=None)
        finally:
            video.close()

        with open(audio_path, "rb") as audio_f:
            data_uri = "data:audio/mp3;base64," + base64.b64encode(
                audio_f.read()
            ).decode("utf-8")
        return data_uri, Path(orig_name).stem + ".mp3"
    finally:
        for pth in (source_path, audio_path):
            if pth:
                try:
                    os.unlink(pth)
                except OSError:
                    pass


async def transcribe_video(
    video_base64: str,
    context: GlobalContext,
    filename: Optional[str] = None,
    reference_audio: bool = False,
    max_duration_seconds: Optional[float] = 9.0,
) -> dict:
    audio_uri, audio_name = await asyncio.to_thread(
        extract_video_audio_b64, video_base64, filename
    )
    return await transcribe_audio(
        audio_uri,
        context,
        audio_name,
        reference_audio=reference_audio,
        max_duration_seconds=max_duration_seconds,
    )


# TODO: when chunking, the total tokens, and input and output tokens should be calculated and returned in the response (aggregated from all chunks)

# TODO: when diarizing The model cost needs to be calculated and returned in the response (using total aggregated tokens from all chunks for input and output tokens)


def _select_dominant_speaker_segments(
    diarized_segments: list,
    *,
    short_fallback_s: float = 1.0,
    allow_single_speaker: bool = False,
) -> Optional[tuple[str, list[dict], dict[str, float], float]]:
    """Pure selection logic: filter text-bearing segments, pick the speaker
    with the largest total speech time, return that speaker's segments.

    Returns ``(target_speaker, target_segments, totals, target_total_seconds)``
    or ``None`` when the input has no text-bearing segments, only one speaker,
    or the dominant speaker's combined speech is shorter than
    ``short_fallback_s``. Tie-break for the dominant pick is first-seen label.

    ``allow_single_speaker`` keeps a single-speaker input instead of returning
    ``None``. The default (False) is for isolating one voice out of a
    conversation, where a single speaker means "nothing to separate, pass the
    clip through". Reference-audio extraction passes ``True``: a reference clip
    is the target talking alone, so a single speaker is the desired case and
    must still yield that speaker's segments (and therefore a transcript) rather
    than falling through to the empty-text passthrough.
    """
    speech_segs: list[dict] = []
    for seg in diarized_segments:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        speaker = str(seg.get("speaker") or "unknown")
        speech_segs.append(
            {"speaker": speaker, "start": start, "end": end, "text": text}
        )

    if not speech_segs:
        return None

    totals: dict[str, float] = {}
    first_seen: list[str] = []
    for seg in speech_segs:
        spk = seg["speaker"]
        if spk not in totals:
            totals[spk] = 0.0
            first_seen.append(spk)
        totals[spk] += seg["end"] - seg["start"]

    # Only one distinct speaker: for dominant-speaker isolation there is nothing
    # to separate, so signal passthrough (use the whole clip). For reference
    # audio a single speaker is the expected, desired case, so keep that speaker
    # rather than bailing — otherwise the reference document is stored with an
    # empty transcript.
    if len(totals) <= 1 and not allow_single_speaker:
        return None

    target_speaker = sorted(
        first_seen, key=lambda s: (-totals[s], first_seen.index(s))
    )[0]
    target_segs = [s for s in speech_segs if s["speaker"] == target_speaker]
    target_total = sum(s["end"] - s["start"] for s in target_segs)
    if (
        target_total < short_fallback_s
    ):  # This needs to indicate that the reference audio is too short and must be at least 1 second
        return None
    return target_speaker, target_segs, totals, target_total


async def isolate_dominant_speaker_audio_b64(
    audio_base64: str,
    context: GlobalContext,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    reference_audio: Optional[bool] = False,
) -> dict:
    """Diarize input audio, pick the dominant speech-bearing speaker, and
    return a coherent triple describing the produced clip.

    Returns:
        ``{"audio_base64_preprocessed": <data URI>,
           "duration": <seconds of the clip>,
           "text": <transcript text of the clip>}``

    The ``audio_base64_preprocessed``, ``duration``, and ``text`` always
    describe the same audio. ``text`` matches the OpenAI transcription API's
    key and is the transcript of what is in the encoded clip.

    Selection cascade (after filtering to text-bearing segments and picking
    the speaker with the largest total speech time, tie-break by first-seen
    label):

    * ``reference_audio=True``: take the single longest contiguous target
      segment, capped at ``context.reference_audio_clip_max_seconds`` (~9 s).
      ``content`` is that segment's transcript; ``duration`` is its length.
    * ``reference_audio=False``: concatenate all target segments in order
      with short fades. ``content`` is the joined transcripts; ``duration``
      is the sum of segment durations.

    Falls back to ``{"audio_base64_preprocessed": <input>, "duration": None,
    "text": ""}`` on decode failure, diarization failure, single-speaker
    input, no text-bearing segments, or combined target speech < 1 s.
    """
    # OpenAI's diarizer rejects ``known_speaker_references`` whose duration is
    # not strictly between 1.2 s and 10.0 s. Keep a margin inside both bounds:
    # mp3 frame padding can nudge a clip a few ms past the requested length, so
    # capping at exactly 10.0 s can still produce a >10.0 s file and a 400.
    _OPENAI_REF_MIN_S = 1.3
    _OPENAI_REF_MAX_S = 9.5
    target_clip_max_s = min(
        float(getattr(context, "reference_audio_clip_max_seconds", None) or 9.0),
        _OPENAI_REF_MAX_S,
    )
    short_fallback_s = 1.0
    fade_s = 0.025

    def _fallback() -> dict:
        return {
            "audio_base64_preprocessed": audio_base64,
            "duration": None,
            "text": "",
        }

    work_path: Optional[str] = None
    output_path: Optional[str] = None
    try:
        # Materialize the input so the cropping step has a real AudioFileClip
        # source. Diarizer timestamps refer to the preprocessed audio, which
        # preserves wall-clock timing (preprocess_audio does not time-shift),
        # so the same timestamps are valid against the raw input.
        try:
            raw = _decode_base64_media_payload(audio_base64)
        except Exception as exc:
            logger.warning(
                "isolate_dominant_speaker_audio_b64: cannot decode input (%s); returning fallback",
                exc,
            )
            return _fallback()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(raw)
            work_path = tmp.name

        try:
            diar = await transcribe_audio_diarize(
                media_base64=audio_base64,
                context=context,
                encoded_reference_audio=None,
                filename=filename,
                content_type=content_type,
                reference_audio=reference_audio,
            )
        except Exception as exc:
            logger.warning(
                "isolate_dominant_speaker_audio_b64: diarization failed (%s); returning fallback",
                exc,
            )
            return _fallback()

        selection = _select_dominant_speaker_segments(
            (diar or {}).get("segments") or [],
            short_fallback_s=short_fallback_s,
            allow_single_speaker=bool(reference_audio),
        )
        if selection is None:
            logger.info(
                "isolate_dominant_speaker_audio_b64: no usable diarization (single speaker / no text / target < 1 s); returning fallback"
            )
            return _fallback()
        target_speaker, target_segs, totals, target_total = selection
        logger.info(
            "isolate_dominant_speaker_audio_b64: dominant speaker=%r totals=%s",
            target_speaker,
            {k: round(v, 2) for k, v in totals.items()},
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
            output_path = tmp_out.name

        # The cropping/concatenation/encode below is all synchronous moviepy +
        # ffmpeg work. Run it in a worker thread so the event loop stays free
        # (a long encode here previously blocked auth and other requests, and
        # prevented cancellation from being serviced).
        def _produce_clip() -> dict:
            clip_for_crop = AudioFileClip(work_path)
            try:
                if reference_audio:
                    # Reference clip: OpenAI's known_speaker_references must be a
                    # single-speaker clip between 1.2 s and 10 s, so we keep one
                    # contiguous window of the target capped at
                    # reference_audio_clip_max_seconds. Anchor on the target's
                    # longest contiguous segment (the cleanest sustained speech),
                    # then extend through the immediately-adjacent following
                    # target segments while the window stays under the cap.
                    # ``text`` is the concatenation of exactly the segments inside
                    # the kept window, so the stored transcript matches the stored
                    # audio (the previous code paired a single segment's *full*
                    # text with an audio clip that could be hard-truncated to the
                    # cap, leaving text and audio out of sync). Extension stops at
                    # any real gap so the subclip can't swallow an intervening
                    # other-speaker turn — the reference must stay single-speaker.
                    _ref_max_gap_s = 0.5
                    ordered = sorted(target_segs, key=lambda s: float(s["start"]))
                    anchor = max(ordered, key=lambda s: s["end"] - s["start"])
                    anchor_idx = ordered.index(anchor)
                    seg_start = float(anchor["start"])
                    seg_end = float(anchor["end"])
                    kept = [anchor]
                    for nxt in ordered[anchor_idx + 1 :]:
                        if (float(nxt["start"]) - seg_end) > _ref_max_gap_s:
                            break
                        if (float(nxt["end"]) - seg_start) > target_clip_max_s:
                            break
                        seg_end = float(nxt["end"])
                        kept.append(nxt)
                    # Hard cap covers a single anchor segment longer than the cap
                    # (one continuous utterance): the audio is truncated to the
                    # cap; ``text`` then over-covers slightly, which is the only
                    # case the segment granularity can't keep perfectly in sync.
                    if (seg_end - seg_start) > target_clip_max_s:
                        seg_end = seg_start + target_clip_max_s
                    # Floor the clip at OpenAI's 1.2 s minimum: if the kept window
                    # is too short, widen it into the surrounding audio (it still
                    # centers on the target) rather than emit a sub-1.2 s
                    # reference the diarizer will reject.
                    clip_total = float(clip_for_crop.duration or 0.0)
                    if clip_total and (seg_end - seg_start) < _OPENAI_REF_MIN_S:
                        seg_end = min(clip_total, seg_start + _OPENAI_REF_MIN_S)
                        if (seg_end - seg_start) < _OPENAI_REF_MIN_S:
                            seg_start = max(0.0, seg_end - _OPENAI_REF_MIN_S)
                    sub = clip_for_crop.subclipped(seg_start, seg_end)
                    try:
                        sub.write_audiofile(output_path, codec="mp3", logger=None)
                    finally:
                        sub.close()
                    clip_duration = float(seg_end - seg_start)
                    clip_content = " ".join(
                        (s.get("text") or "").strip() for s in kept
                    ).strip()
                    logger.info(
                        "isolate_dominant_speaker_audio_b64[ref]: kept %d target segment(s) %.2fs (cap %.2fs)",
                        len(kept),
                        clip_duration,
                        target_clip_max_s,
                    )
                else:
                    # Non-reference: concatenate all target segments with short
                    # fades, capturing the full target transcript.
                    from moviepy.audio.AudioClip import concatenate_audioclips
                    from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

                    subs = []
                    try:
                        for seg in target_segs:
                            seg_dur = seg["end"] - seg["start"]
                            sub = clip_for_crop.subclipped(seg["start"], seg["end"])
                            f = min(fade_s, seg_dur / 4)
                            if f > 0:
                                sub = sub.with_effects(
                                    [AudioFadeIn(f), AudioFadeOut(f)]
                                )
                            subs.append(sub)
                        glued = concatenate_audioclips(subs)
                        try:
                            glued.write_audiofile(output_path, codec="mp3", logger=None)
                        finally:
                            glued.close()
                    finally:
                        for s in subs:
                            try:
                                s.close()
                            except Exception:
                                pass
                    clip_duration = float(target_total)
                    clip_content = " ".join(
                        (s.get("text") or "").strip() for s in target_segs
                    ).strip()
                    logger.info(
                        "isolate_dominant_speaker_audio_b64[nonref]: concatenated %d target segments (%.2fs)",
                        len(target_segs),
                        target_total,
                    )
            finally:
                clip_for_crop.close()

            with open(output_path, "rb") as fh:
                final_audio_b64 = "data:audio/mp3;base64," + base64.b64encode(
                    fh.read()
                ).decode("utf-8")
            return {
                "audio_base64_preprocessed": final_audio_b64,
                "duration": clip_duration,
                "text": clip_content,
            }

        return await asyncio.to_thread(_produce_clip)

    finally:
        for pth in (work_path, output_path):
            if pth:
                try:
                    os.unlink(pth)
                except OSError:
                    pass


async def transcribe_audio_diarize(
    media_base64: str,
    context: GlobalContext,
    encoded_reference_audio: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    reference_audio: Optional[bool] = False,
) -> dict:
    """Diarize video or audio from base64 (chunked when audio exceeds whisper_max_bytes)."""
    start_time = time_ns()
    client = _openai_client_for_speech(context)

    orig_name = filename or "upload.mp4"
    suffix = Path(orig_name).suffix or ".mp4"
    is_audio = _upload_is_audio_for_diarize(filename, content_type)

    # asdf

    source_path = None
    audio_path = None
    if is_audio:
        # Handle Audio
        # Identify the longest up to 9 second audio clip for reference audio.
        # Preprocess as necessary to improve quality otherwise.

        preprocessed_audio = await preprocess_audio(  # Convert to MP3 Codec
            media_base64,
            truncate_only=False,
            reference_audio=False,
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_upload:
            raw = _decode_base64_media_payload(media_base64)
            temp_upload.write(raw)
            temp_upload.flush()
            source_path = temp_upload.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_path = temp_audio.name

        # moviepy/ffmpeg audio extraction is synchronous and CPU/IO-bound; run
        # it off the event loop so concurrent requests stay responsive.
        def _extract_audio() -> None:
            video = VideoFileClip(source_path)
            try:
                video.audio.write_audiofile(audio_path, codec="mp3", logger=None)
            finally:
                video.close()
                os.unlink(source_path)

        await asyncio.to_thread(_extract_audio)

        with open(audio_path, "rb") as audio_f:
            b64_encoded_reference_audio = f"data:audio/mp3;base64,{base64.b64encode(audio_f.read()).decode('utf-8')}"
            # Create
            preprocessed_audio = await preprocess_audio(
                b64_encoded_reference_audio,
                truncate_only=False,
                reference_audio=False,
            )

    raw = _decode_base64_media_payload(preprocessed_audio["audio_base64"])
    if not is_audio:
        os.unlink(audio_path)  # Non preprocessed audio from video

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(raw)
        temp_audio.flush()
        audio_path = temp_audio.name  # Preprocessed audio

    ##### Diarize content using reference audio
    try:
        size_bytes = len(raw)
        diarize_upload_name = (
            Path(orig_name).stem + ".mp3"  # Preprocessed audio codec
        )

        if size_bytes <= context.whisper_max_bytes:
            # The diarizer SDK call is synchronous; hop it to a thread so it
            # doesn't block the event loop (which also serves the media-job
            # progress SSE endpoint). Mirrors the chunked path below.
            response = await asyncio.to_thread(
                _diarize_one_mp3_path,
                audio_path,
                diarize_upload_name,
                context,
                client,
                encoded_reference_audio,
            )
            response_dict = response.model_dump()
        else:
            # Probe duration with a one-shot AudioFileClip, then close it.
            # Slicing uses ffmpeg ``-c copy`` (stream copy, no re-encode) so
            # each chunk file is produced in well under a second regardless
            # of input size — no MoviePy reader is held across chunks, which
            # also sidesteps the MoviePy 2.x shared-reader bug that closes
            # the parent's ffmpeg process when a subclip is closed.
            with AudioFileClip(audio_path) as probe:
                duration = float(probe.duration or 0.0)
            n = max(2, math.ceil(size_bytes / context.chunk_source_bytes_target))
            chunk_dur = duration / n

            chunk_paths: list[str] = []
            offsets: list[float] = []
            for i in range(n):
                t_off = i * chunk_dur
                t_end = duration if i == n - 1 else (i + 1) * chunk_dur
                fd, chunk_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                # ``-ss`` before ``-i`` does fast (input) seek; with ``-c
                # copy`` ffmpeg snaps to the nearest mp3 frame boundary,
                # which is acceptable for diarization-chunk granularity.
                # ``-loglevel error`` suppresses progress chatter.
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{t_off}",
                        "-to",
                        f"{t_end}",
                        "-i",
                        audio_path,
                        "-c",
                        "copy",
                        chunk_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                chunk_paths.append(chunk_path)
                offsets.append(t_off)

            try:
                # POST all chunk diarizations to OpenAI concurrently. The
                # SDK call is synchronous so we hop each one to a thread.
                async def _diarize_chunk_async(
                    idx: int, path: str
                ) -> TranscriptionDiarized:
                    return await asyncio.to_thread(
                        _diarize_one_mp3_path,
                        path,
                        f"chunk_{idx}.mp3",
                        context,
                        client,
                        encoded_reference_audio,
                    )

                chunk_responses = list(
                    await asyncio.gather(
                        *[_diarize_chunk_async(i, p) for i, p in enumerate(chunk_paths)]
                    )
                )
            finally:
                for path in chunk_paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

            text_parts = [resp.text for resp in chunk_responses]
            u_in = 0
            u_out = 0
            for resp in chunk_responses:
                ud = _diarize_usage_tokens_dict(resp.usage)
                u_in += int(ud.get("input_tokens") or 0)
                u_out += int(ud.get("output_tokens") or 0)
            usage_merged = None
            if u_in or u_out:
                usage_merged = {
                    "type": "tokens",
                    "input_tokens": u_in,
                    "output_tokens": u_out,
                    "total_tokens": u_in + u_out,
                }
            merged_segments = _merge_diarized_segments_from_chunks(
                chunk_responses, offsets
            )
            response_dict = {
                "duration": duration,
                "segments": merged_segments,
                "task": "transcribe",
                "text": " ".join(text_parts),
                "usage": usage_merged,
                "diarization_chunk_count": n,
            }

        usage = response_dict.get("usage") or {}
        if isinstance(usage, dict):
            usage_d = usage
        else:
            usage_d = _diarize_usage_tokens_dict(usage)
        response_dict["total_cost"] = _diarize_token_cost(usage_d, context)
        if response_dict["total_cost"] == 0 and response_dict.get("duration"):
            response_dict["total_cost"] = (
                float(response_dict["duration"])
                * context.audio_diarization_estimated_price_per_minute
            )

        model = context.audio_diarization_model or "gpt-4o-transcribe-diarize"
        response_dict.update({"inference_type": "diarization", "model": model})
        inp = usage_d.get("input_tokens")
        out = usage_d.get("output_tokens")
        tot = usage_d.get("total_tokens")
        if tot is None and (inp is not None or out is not None):
            tot = (inp or 0) + (out or 0)
        response_dict.update(
            {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": tot,
            }
        )

        response_dict["latency_ms"] = (time_ns() - start_time) / 1e6
        response_dict["encoded_audio_base64"] = preprocessed_audio["audio_base64"]
        return response_dict
    finally:
        for pth in (source_path, audio_path):
            if pth:
                try:
                    os.unlink(pth)
                except OSError:
                    pass


async def extract_base64_str_from_image(image_base64: str, filename: str) -> str:
    """Build a ``data:image/...;base64,...`` URI from raw or data-URI base64 input."""
    suffix = Path(filename).suffix
    if suffix not in [".png", ".jpeg", ".jpg", ".webp"]:
        raise ValueError(
            "Unsupported file type. Only PNG, JPEG, JPG, and WEBP are allowed."
        )
    raw = _decode_base64_media_payload(image_base64)
    base64_image = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{suffix.lstrip('.')};base64,{base64_image}"


async def resize_image_bytes(image_bytes: bytes) -> tuple[bytes, Optional[str]]:
    """Downscale large images to JPEG. Returns ``(bytes, content_type)``; ``content_type`` is ``None`` when unchanged."""
    with Image.open(io.BytesIO(image_bytes)) as original_img:
        original_w, original_h = original_img.size
        longest_side = max(original_w, original_h)
        MAX_DIMENSION = 512
        if longest_side <= MAX_DIMENSION:
            return image_bytes, None
        image = original_img.convert("RGB")
        scale = MAX_DIMENSION / longest_side
        resized_width = max(1, int(round(original_w * scale)))
        resized_height = max(1, int(round(original_h * scale)))
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue(), "image/jpeg"


async def load_baseline_features_explainer_model(store: BaseStore):
    import base64
    import json
    import pickle

    import aiofiles
    import numpy as np
    import shap

    # Attempt to pull stored model and data from store
    baseline_features_namespace = ("baseline_features_arr_list_str",)
    baseline_features_model_namespace = ("baseline_features_model_b64_pkl",)

    baseline_features_arr_list_str_ITEM = await store.aget(
        baseline_features_namespace, key="baseline_features_arr_list_str"
    )
    baseline_features_arr_list_str = (
        getattr(baseline_features_arr_list_str_ITEM, "value", None) or {}
    ).get("value", None)

    baseline_features_model_b64_pkl_ITEM = await store.aget(
        baseline_features_model_namespace, key="baseline_features_model_b64_pkl"
    )
    baseline_features_model_b64_pkl = (
        getattr(baseline_features_model_b64_pkl_ITEM, "value", None) or {}
    ).get("value", None)

    # If the baseline_features_model has not yet been stored, load from disk and store the model:
    if not baseline_features_model_b64_pkl:
        _MODEL_PATH = "src/anubis/utils/dataset/baseline_features_model_b64.pkl"
        # Load model
        async with aiofiles.open(_MODEL_PATH, "rb") as fp:
            baseline_features_model_b64_pkl = await fp.read()
        baseline_features_model_b64_pkl_str = (baseline_features_model_b64_pkl).decode(
            "utf-8"
        )
        await store.aput(
            baseline_features_model_namespace,
            key="baseline_features_model_b64_pkl",
            value={"value": baseline_features_model_b64_pkl_str},
        )

    # Convert from pickled string to Isolation Forest model
    model = pickle.loads(base64.b64decode(baseline_features_model_b64_pkl))

    # If the baseline_features_arr has not yet been stored, load from disk and store the array:
    if not baseline_features_arr_list_str:
        _BASELINE_ANSWERS_RESPONSES_ARR_DIR = (
            "src/anubis/utils/dataset/baseline_features_arr.npy"
        )
        baseline_features_arr = np.load(
            _BASELINE_ANSWERS_RESPONSES_ARR_DIR, allow_pickle=False
        )

        baseline_features_arr_list_str = json.dumps(baseline_features_arr.tolist())

        await store.aput(
            baseline_features_namespace,
            key="baseline_features_arr_list_str",
            value={"value": baseline_features_arr_list_str},
        )

    # Convert from str to np.array
    baseline_features_arr = np.array(json.loads(baseline_features_arr_list_str))

    explainer = shap.KernelExplainer(
        model.predict, shap.kmeans(baseline_features_arr, 100)
    )
    return explainer, model


async def compute_shap_values_against_baseline(
    feature_values, store: BaseStore
) -> dict:
    import pandas as pd

    from src.anubis.utils.dataset.style_features import FEATURE_NAMES

    _explainer, _model = await load_baseline_features_explainer_model(store)
    prediction = bool(_model.predict(feature_values.reshape(1, -1)) == 1)

    shap_values = _explainer.shap_values(feature_values.reshape(1, -1))

    df = pd.DataFrame(
        shap_values.flatten(),
        index=FEATURE_NAMES,
        columns=["unmodified_llm_comparison_isolation_forest_shap_values"],
    )
    shap_dict = df[
        df["unmodified_llm_comparison_isolation_forest_shap_values"] != 0.0
    ].to_dict()
    shap_dict["unmodified_llm_comparison_isolation_forest_shap_values_description"] = (
        "Negative values indicate dissimilarity from unmodified llm dataset. Positive values indicate similarity to unmodified llm responses. Scale is -1 to 1."
    )
    shap_dict[
        "no_statistically_significant_difference_between_sample_and_unmodified_llm_according_to_isolation_forest"
    ] = prediction

    return shap_dict
