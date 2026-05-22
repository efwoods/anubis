# src/anubis/utils/helper_functions

# Vectore store helper functions
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.anubis.utils.model import init_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from src.anubis.utils.context import GlobalContext

from langgraph.store.base import BaseStore


from src.anubis.utils.tokenizer import count_tokens

import logging
import re

from datetime import datetime
from datetime import timezone

from src.anubis.utils.prompts.system_prompts import TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
from typing import Optional

import math
import tempfile
from pathlib import Path
from moviepy import AudioFileClip, VideoFileClip
from time import time_ns
from openai import OpenAI
from openai.types.audio.transcription_diarized import TranscriptionDiarized
import base64

import io
from PIL import Image

logger = logging.getLogger(__name__)

from pydantic import BaseModel
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

def add_queries(existing: Sequence[str], new:Sequence[str]) -> Sequence[str]:
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
from typing import Union, Any, Literal

from langgraph.store.base import SearchItem, Item

def reduce_docs(
    existing: Sequence[Document] | None,
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        Sequence[SearchItem],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            elif isinstance(item, SearchItem) or isinstance(item, Item):
                logger.info("breakpoint")
                page_content = getattr(item,'value', {}).get("document", {}).get("kwargs", {}).get("page_content", "")
                document_metadata = getattr(item, 'value', {}).get("document", {}).get("kwargs", {}).get("metadata", {})
                document = Document(page_content=page_content, metadata=document_metadata)
                coerced.append(document)
            else:
                coerced.append(item)
        return coerced
    return existing or []


from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from src.anubis.utils.context import GlobalContext


async def extract_user_id_assistant_id(config: RunnableConfig):
    user_state = {}
    assistant_state = {}

    user_id = config.get("configurable",{}).get("user_id", '')

    if user_id != '':
        user_state.update({"user_id": user_id})
    else:
        """anonymous_user_id is 'str(uuid5(NAMESPACE_URL, 'anonymous_user_id"""
        user_state.update({"user_id":'9977df19-9ceb-5f87-a130-55f6a6282069'})
        
    assistant_id = config.get("configurable", {}).get("assistant_id", "")

    if assistant_id != "":
        assistant_state.update({"assistant_id":assistant_id})
    else:
        raise Exception("Assistant does not have an id from the context. Provide an assistant_id in config['configurable']['assistant_id'].")

    return user_state, assistant_state

async def configure_assistant_context(config: RunnableConfig, store: BaseStore):
        user_id, assistant_id = await extract_user_id_assistant_id(config)
        
        namespace=(user_id, assistant_id, "assistant_ctx")
        ai_context_item = await store.aget(namespace, key=assistant_id)
        logger.info(f"ai_context_item: {ai_context_item}")

        # Load/UPDATE AI SELF IDENTITY
        logger.info("item object breakpoint")

        # get the current assistant context as a dict

        configurable_assistant_ctx = config.get("configurable", {}).get("assistant_ctx", None)

        if configurable_assistant_ctx is not None:
            if ai_context_item is not None:
                for key, value in configurable_assistant_ctx:
                    if (value != "" and value != None) and key != "metadata":
                        ai_context_item.value['assistant_ctx'].update({key: value})
                if configurable_assistant_ctx.get("metadata", None) is not None:
                    update_metadata = configurable_assistant_ctx.get("metadata")
                    ai_context_item.value['assistant_ctx']['metadata'].update(update_metadata)                 

                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":ai_context_item.value["assistant_ctx"]})
            else:
                init_assistant_ctx = {
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "name":configurable_assistant_ctx.get("name", ""),
                    "description":configurable_assistant_ctx.get("description", ""),
                    "metadata": configurable_assistant_ctx.get("metadata", {})
                }
                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":init_assistant_ctx})
        else:
            if ai_context_item is None:
                init_assistant_ctx = {
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "name": "",
                    "description": "",
                    "metadata": {}
                }
                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":init_assistant_ctx})

        ai_context_item = await store.aget(namespace, key=assistant_id)

        return ai_context_item


async def image_to_text(target_image_url: str, 
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
            reference_message = {"type": "image_url", "image_url":{"url":reference_image_url}}
        else:
            # base 64 encoding
            reference_message = {"type": "image_url", "image_url":{"url":f"data:image/jpeg;base64,{reference_image_url}"}}

    if "." in target_image_url:
        target_message = {"type": "image_url", "image_url": {"url":target_image_url}}
    else:
        target_message = {"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{target_image_url}"}}

    # Compile the message
    content = [reference_message, target_message] if reference_image_url is not None else [target_message]
    system_message = [SystemMessage(content=TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION)]
    human_message = [{"role": "user", "content": content}]

    messages = system_message + human_message

    # TODO: response_metrics_aggregation
    model = init_model()

    response = await model.ainvoke(input=messages)

    description  = response.content

    return description 


############################  CHUNK LONG MESSAGES  #############################

async def chunk_long_messages(human_message_list, context) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=0)
    # Chunk Long Messages
    chunked_message_list = []
    for message in human_message_list:
        message_token_len = count_tokens([message])
    if message_token_len > context.model_token_limit:
        text_chunks = text_splitter.split_text(getattr(message, "text", ""))
        message = [HumanMessage(content=[{'type':'text', 'text':chunk}]) for chunk in text_chunks]
    if isinstance(message, list):
        chunked_message_list += message
    else:
        chunked_message_list += [message]
    
    human_message_list = chunked_message_list
    return human_message_list 


############################  Summarize Messages  #############################

from typing import Optional


""" YOUTUBE HELPER FUNCTIONS """
import yt_dlp
import os
import re


async def download_transcript(url: str, lang: str = "en", auto_subs: bool = True, output_dir: str = ".") -> str:
    """
    Download transcript/subtitles from a YouTube video.

    Args:
        url: YouTube video URL
        lang: Language code (e.g., 'en', 'es', 'fr')
        auto_subs: Fall back to auto-generated subtitles if manual not found
        output_dir: Directory to save files

    Returns:
        Path to the downloaded subtitle file
    """
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": auto_subs,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
    }

    async with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")

    # Find the downloaded .vtt file
    for f in os.listdir(output_dir):
        if f.endswith(".vtt"):
            return os.path.join(output_dir, f)

    raise FileNotFoundError(f"No subtitle file found for '{title}'. "
                            "Try listing available languages first.")


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
def get_transcript(url: str, lang: str = "en", save_txt: bool = True) -> str:
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
    vtt_path = download_transcript(url, lang=lang)

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
    return OpenAI(api_key=key)


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
        content = client.audio.transcriptions.create(
            file=(upload_filename, audio_f),
            model=model,
            response_format="text",
        )
    return {
        "content": content,
        "file_duration_s": duration_seconds,
        "total_cost": transcription_cost,
        "latency_ms": (time_ns() - start) / 1e6,
        "model": model,
        "inference_type": "transcription",
    }


async def _transcribe_saved_path(path: str, upload_filename: str, context: GlobalContext) -> dict:
    """Transcribe audio on disk; split by time when the file exceeds the Whisper 25 MiB limit."""
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
                parts.append(seg["content"])
                total_cost += seg["transcription_cost"]
                inner_latency += seg["latency_ms"]
            finally:
                sub.close()
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
        return {
            "content": " ".join(parts),
            "file_duration_s": duration,
            "total_cost": total_cost,
            "latency_ms": inner_latency,
            "whisper_chunk_count": n,
            "model": context.audio_transcription_model, 
            "inference_type": "transcription"
        }
    finally:
        clip.close()


async def transcribe_audio(
    audio_base64: str, context: GlobalContext, filename: Optional[str] = None, 
    reference_audio: bool = False,
    max_duration_seconds: Optional[float] = 9.0,
) -> dict:

    # Remove noise and isolate the vocals; if reference audio, truncate to 9 seconds:
    preprocessed_audio = await preprocess_audio(audio_base64, truncate_only=False, reference_audio=reference_audio, max_duration_seconds=max_duration_seconds)
    audio_base64 = preprocessed_audio["audio_base64"]

    raw = _decode_base64_media_payload(audio_base64)
    
    # Update preprocessed filename to mp3 codec
    suffix = ".mp3" # mp3 after preprocessing to mp3 codec

    filename = Path(filename).stem + ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        name = Path(filename or f"audio{suffix}").name
        result = await _transcribe_saved_path(path, name, context)
        result['audio_base64_preprocessed'] = audio_base64
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
    raw = _decode_base64_media_payload(audio_base64)
    in_suffix = Path(filename or "audio.mp3").suffix or ".mp3"
    max_seconds = float(max_duration_seconds or 9.0)

    src_path: Optional[str] = None
    wav_path: Optional[str] = None
    out_mp3_path: Optional[str] = None
    saved_dev_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=in_suffix, delete=False) as tmp_in:
            tmp_in.write(raw)
            tmp_in.flush()
            src_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            out_mp3_path = tmp_mp3.name

        sample_rate: Optional[int] = None

        if truncate_only:
            clip = AudioFileClip(src_path)
            try:
                full_duration = float(clip.duration or 0.0)
                if full_duration > max_seconds:
                    sub = clip.subclipped(0, max_seconds)
                    duration_seconds = max_seconds
                else:
                    sub = clip
                    duration_seconds = full_duration
                try:
                    sub.write_audiofile(out_mp3_path, codec="mp3", logger=None)
                finally:
                    if sub is not clip:
                        sub.close()
                fps = getattr(clip, "fps", None)
                sample_rate = int(fps) if fps else None
            finally:
                clip.close()
        else:
            import librosa
            import noisereduce as nr
            import numpy as np
            import soundfile as sf

            # noisereduce: spectral-gating noise reduction (suppresses background
            # noise while preserving the voice). Decode the input MP3 to a mono
            # float waveform with ffmpeg (moviepy) — libsndfile/librosa can't
            # reliably decode compressed containers — then re-encode the cleaned
            # audio back to MP3. moviepy is already loaded; numpy/noisereduce/
            # librosa/soundfile are imported lazily so the truncate-only path
            # doesn't pay for them.
            in_clip = AudioFileClip(src_path)
            try:
                sample_rate = int(in_clip.fps or 44100)
                samples = in_clip.to_soundarray(fps=sample_rate)
            finally:
                in_clip.close()
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            mono = np.ascontiguousarray(samples, dtype=np.float32)

            # Enhance vocals and remove noise
            if enhance_vocals_and_remove_noise:
                logger.info(f"preprocess_audio noisereduce sr: {sample_rate}")
                waveform = nr.reduce_noise(y=mono, sr=sample_rate).astype(np.float32)
            else: 
                waveform = mono

            # Identify the longest window of audio that contains potential speech
            duration_seconds = waveform.shape[-1] / float(sample_rate)

            if reference_audio and duration_seconds > max_seconds:
                frame_length = 2048
                hop_length = 512
                rms = librosa.feature.rms(
                    y=waveform, frame_length=frame_length, hop_length=hop_length
                )[0]
                window_frames = max(1, int(round((max_seconds * sample_rate) / hop_length)))
                if window_frames < len(rms):
                    # Sliding-window energy via cumulative sum on rms^2.
                    energy = rms.astype(np.float64) ** 2
                    csum = np.concatenate(([0.0], np.cumsum(energy)))
                    window_sums = csum[window_frames:] - csum[:-window_frames]
                    start_frame = int(np.argmax(window_sums))
                    start_sample = start_frame * hop_length
                    end_sample = min(
                        start_sample + int(round(max_seconds * sample_rate)),
                        waveform.shape[-1],
                    )
                    waveform = waveform[start_sample:end_sample]

            duration_seconds = waveform.shape[-1] / float(sample_rate)

            # Create MP3 Codec
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                wav_path = tmp_wav.name
            sf.write(wav_path, waveform, int(sample_rate))
            clip = AudioFileClip(wav_path)
            try:
                clip.write_audiofile(out_mp3_path, codec="mp3", logger=None)
            finally:
                clip.close()

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
            result = client.audio.transcriptions.create(
                model=model,
                file=(upload_name, audio_f),
                response_format="diarized_json",
                chunking_strategy="auto",
                extra_body=extra,
            )
        else:
            result = client.audio.transcriptions.create(
                model=model,
                file=(upload_name, audio_f),
                response_format="diarized_json",
                chunking_strategy="auto",
            )
    if not isinstance(result, TranscriptionDiarized):
        msg = "Expected diarized_json transcription response from OpenAI."
        raise TypeError(msg)
    return result


def _merge_diarized_segments_from_chunks(
    chunk_responses: list[TranscriptionDiarized],
    time_offsets: list[float],
) -> list[dict]:
    merged: list[dict] = []
    for idx, (resp, t0) in enumerate(zip(chunk_responses, time_offsets, strict=True)):
        for seg in resp.segments:
            sd = seg.model_dump()
            merged.append(
                {
                    **sd,
                    "start": float(sd["start"]) + t0,
                    "end": float(sd["end"]) + t0,
                    "speaker": f"{idx}:{sd.get('speaker', '')}",
                }
            )
    return merged

async def transcribe_video(
    video_base64: str,
    context: GlobalContext,
    filename: Optional[str] = None,
    reference_audio: bool = False,
    max_duration_seconds: Optional[float] = 9.0,
) -> dict:
    raw = _decode_base64_media_payload(video_base64)
    orig_name = filename or "upload.mp4"
    suffix = Path(orig_name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_upload:
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

        size_bytes = os.path.getsize(audio_path)
        diarize_upload_name = (
            Path(orig_name).stem + ".mp3"
        )
        
        with open(audio_path, "rb") as audio_f:
            b64_encoded_reference_audio = (
                f"data:audio/mp3;base64,{base64.b64encode(audio_f.read()).decode('utf-8')}"
            )
            response = await transcribe_audio(b64_encoded_reference_audio, context, diarize_upload_name, reference_audio=reference_audio, max_duration_seconds=max_duration_seconds)
        return response

# TODO: when chunking, the total tokens, and input and output tokens should be calculated and returned in the response (aggregated from all chunks)

# TODO: when diarizing The model cost needs to be calculated and returned in the response (using total aggregated tokens from all chunks for input and output tokens)

async def transcribe_audio_diarize(
    media_base64: str,
    context: GlobalContext,
    encoded_reference_audio: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict:
    """Diarize video or audio from base64 (chunked when audio exceeds whisper_max_bytes)."""
    start_time = time_ns()
    client = _openai_client_for_speech(context)
    
    orig_name = filename or "upload.mp4"
    suffix = Path(orig_name).suffix or ".mp4"
    is_audio = _upload_is_audio_for_diarize(filename, content_type)

    # Preprocess audio or video to isolate the vocals:
    if is_audio:
        preprocessed_audio = await preprocess_audio(media_base64, truncate_only=False, reference_audio=False)
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_upload:
            raw = _decode_base64_media_payload(media_base64)
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
            os.unlink(source_path)

        with open(audio_path, "rb") as audio_f:
            b64_encoded_reference_audio = (
                f"data:audio/mp3;base64,{base64.b64encode(audio_f.read()).decode('utf-8')}"
            )
            preprocessed_audio = await preprocess_audio(b64_encoded_reference_audio, truncate_only=False, reference_audio=False)

    raw = _decode_base64_media_payload(preprocessed_audio['audio_base64'])
    if not is_audio:
        os.unlink(audio_path) # Non preprocessed audio from video

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(raw)
        temp_audio.flush()
        audio_path = temp_audio.name # Preprocessed audio
    
    try:
        size_bytes = len(raw)
        diarize_upload_name = (
            Path(orig_name).stem + ".mp3" # Preprocessed audio codec
        )

        if size_bytes <= context.whisper_max_bytes:
            response = _diarize_one_mp3_path(
                audio_path,
                diarize_upload_name,
                context,
                client,
                encoded_reference_audio,
            )
            response_dict = response.model_dump()
        else:
            clip = AudioFileClip(audio_path)
            try:
                duration = float(clip.duration or 0.0)
                n = max(2, math.ceil(size_bytes / context.chunk_source_bytes_target))
                chunk_dur = duration / n
                text_parts: list[str] = []
                chunk_responses: list[TranscriptionDiarized] = []
                offsets: list[float] = []
                u_in = 0
                u_out = 0
                for i in range(n):
                    t_off = i * chunk_dur
                    t_end = duration if i == n - 1 else (i + 1) * chunk_dur
                    sub = clip.subclipped(t_off, t_end)
                    fd, chunk_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    try:
                        sub.write_audiofile(chunk_path, logger=None)
                        resp = _diarize_one_mp3_path(
                            chunk_path,
                            f"chunk_{i}.mp3",
                            context,
                            client,
                            encoded_reference_audio,
                        )
                        text_parts.append(resp.text)
                        chunk_responses.append(resp)
                        offsets.append(t_off)
                        ud = _diarize_usage_tokens_dict(resp.usage)
                        u_in += int(ud.get("input_tokens") or 0)
                        u_out += int(ud.get("output_tokens") or 0)
                    finally:
                        sub.close()
                        try:
                            os.unlink(chunk_path)
                        except OSError:
                            pass
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
            finally:
                clip.close()

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
        response_dict.update(
            {"inference_type": "diarization", "model": model}
        )
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
        raise ValueError("Unsupported file type. Only PNG, JPEG, JPG, and WEBP are allowed.")
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