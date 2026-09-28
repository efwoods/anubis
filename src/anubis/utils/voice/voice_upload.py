"""Feed the voice model from media added in the Voice section.

A file or link added in the Voice section is a deliberate statement from the
owner: this recording is the avatar speaking, and the audible voice should
change to include the recording. Two cases arrive here.

**New media** goes through the ordinary media pipeline, which diarizes the
recording once against the stored reference clip, indexes the transcript, and
cuts the avatar's turns into the voice corpus on the way.

**Media the avatar already holds** is the case this module exists for. The
ordinary pipeline skips an item whose key is already indexed, and before
2026-09-03 the pipeline indexed media without collecting any voice at all, so
an avatar can hold hours of the person speaking and still have zero seconds of
voice. The earlier diarization is already in the store: every speaker turn of a
diarized recording was written as a ``quote`` document carrying ``start``,
``end`` and ``is_target``. So the avatar's windows are read back from those
stored documents, the audio is fetched again, the windows are cut with
``cut_target_turns_to_mp3_data_uri``, and the clip goes to ``add_voice_clip``.
No diarizer call is made, and nothing is written to the store, because the
transcript documents already exist and every store row is embedded.

When a Voice-section batch adds speech to an avatar whose instant clone already
exists, ``finish_voice_upload_batch`` rebuilds the clone preferring the newest
clips, so the recording the owner just added is what the avatar sounds like.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

VOICE_UPLOAD_CLIP_SOURCE = "voice_upload"
"""``avatar_voice_clips.source`` for speech added from the Voice section."""

_STORED_TURN_NAMESPACES = ("quote", "identity")
_STORE_SEARCH_LIMIT = 100_000

ProgressEmitter = Callable[[dict[str, Any]], None]


class VoiceUploadError(RuntimeError):
    """A Voice-section item that produced no speech of the avatar, with the reason."""


def _document_metadata(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", None) or {}
    document = value.get("document") if isinstance(value, dict) else None
    if not isinstance(document, dict):
        return {}
    kwargs = document.get("kwargs") or {}
    metadata = kwargs.get("metadata") if isinstance(kwargs, dict) else None
    return metadata if isinstance(metadata, dict) else {}


async def stored_target_turns(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    namespace_filename: str,
) -> list[dict[str, Any]]:
    """Return the avatar's diarized turns recorded when the item was first processed.

    Reads the ``quote`` and ``identity`` documents stored under the item's key
    and keeps every document the diarizer attributed to the avatar
    (``is_target`` true) that carries a ``start`` and an ``end``. Duplicate
    windows (one turn chunked into several documents) are merged later by
    ``target_windows``.
    """
    turns: list[dict[str, Any]] = []
    for namespace in _STORED_TURN_NAMESPACES:
        items = await store.asearch(
            (user_id, assistant_id, namespace, namespace_filename),
            limit=_STORE_SEARCH_LIMIT,
        )
        for item in items or []:
            metadata = _document_metadata(item)
            if metadata.get("is_target") is not True:
                continue
            start = metadata.get("start")
            end = metadata.get("end")
            if start is None or end is None:
                continue
            turns.append({"start": start, "end": end, "is_target": True})
    return turns


def _is_youtube_link(url: str) -> bool:
    from urllib.parse import urlparse

    from src.anubis.utils.classes.URLDocumentLoaderClass import _YOUTUBE_HOSTS

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


async def audio_for_voice_upload(media_file: dict[str, Any]) -> str:
    """Return the item's audio track as a data URI, without any transcription.

    An uploaded audio file is used as sent; an uploaded video has the audio
    track extracted; a YouTube link has the audio downloaded again. Anything
    else carries no speech to clone.
    """
    from src.anubis.utils.utility import extract_video_audio_b64

    content_type = str(media_file.get("content_type") or "").lower()
    encoded_media = str(media_file.get("base64_encoded_str") or "")
    filename = media_file.get("filename")
    if encoded_media and content_type.startswith("audio/"):
        return encoded_media
    if encoded_media and content_type.startswith("video/"):
        audio_data_uri, _audio_filename = await asyncio.to_thread(
            extract_video_audio_b64, encoded_media, filename
        )
        return audio_data_uri
    link = str(
        media_file.get("page_url")
        or media_file.get("audio_url")
        or media_file.get("video_url")
        or ""
    )
    if link and _is_youtube_link(link):
        from src.anubis.utils.classes.URLDocumentLoaderClass import (
            _download_youtube_audio_b64,
        )

        audio_data_uri, _suffix = await _download_youtube_audio_b64(link)
        if not audio_data_uri:
            raise VoiceUploadError(f"Could not download the audio of {link}.")
        return audio_data_uri
    raise VoiceUploadError(
        f"{filename or 'This item'} is not audio or video, so the item carries no voice."
    )


async def collect_voice_from_indexed_item(
    store: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    media_file: dict[str, Any],
    avatar_name: str = "",
    is_personal_avatar: bool = False,
    emit: ProgressEmitter | None = None,
    repository: Any = None,
) -> float:
    """Cut the avatar's already-diarized turns from an indexed item into the voice corpus.

    Returns the seconds added. Raises ``VoiceUploadError`` with a sentence the
    owner can act on when the item yields no speech of the avatar.
    """
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.clips import (
        cut_target_turns_to_mp3_data_uri,
        target_windows,
    )
    from src.anubis.utils.voice.corpus import (
        active_instant_voice_id,
        add_voice_clip,
        voice_configured,
    )

    speaker_name = avatar_name or "the avatar"
    voice_repository = repository or get_media_asset_repository()
    if voice_repository is None or not voice_configured(context):
        raise VoiceUploadError("Voice cloning is not configured on this server.")

    namespace_filename = str(media_file.get("namespace_filename") or "")
    filename = media_file.get("filename")
    turns = await stored_target_turns(
        store,
        user_id=user_id,
        assistant_id=assistant_id,
        namespace_filename=namespace_filename,
    )
    if not target_windows(turns):
        raise VoiceUploadError(
            f"No speech of {speaker_name} was found when {filename} was first "
            "processed, so there is nothing to add to the voice."
        )

    if emit is not None:
        emit({"stage": "voice_reusing_diarization", "turns": len(turns)})
    audio_data_uri = await audio_for_voice_upload(media_file)
    clip_data_uri, clip_seconds = await cut_target_turns_to_mp3_data_uri(
        audio_data_uri, turns
    )
    if not clip_data_uri or clip_seconds <= 0:
        raise VoiceUploadError(
            f"The speech of {speaker_name} could not be cut from {filename}."
        )

    record = await add_voice_clip(
        voice_repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        audio_data_uri=clip_data_uri,
        duration_seconds=clip_seconds,
        source=VOICE_UPLOAD_CLIP_SOURCE,
        source_document_name=filename,
        is_personal_avatar=is_personal_avatar,
        avatar_name=avatar_name,
    )
    if emit is not None:
        emit(
            {
                "stage": "voice_clip_collected",
                "seconds": clip_seconds,
                "collected_seconds": float(record.get("collected_seconds") or 0.0),
            }
        )
        if active_instant_voice_id(record, context):
            emit({"stage": "instant_clone_created"})
    logger.info(
        "Voice upload of %s added %.1fs to %s from the stored diarization",
        filename,
        clip_seconds,
        assistant_id,
    )
    return clip_seconds


async def instant_voice_before_batch(
    assistant_id: str, *, repository: Any = None, context: Any = None
) -> str | None:
    """Return the active provider's instant clone id before a Voice-section batch runs."""
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.corpus import active_instant_voice_id

    voice_repository = repository or get_media_asset_repository()
    if voice_repository is None:
        return None
    voice_record = await voice_repository.get_voice(assistant_id)
    return active_instant_voice_id(voice_record, context)


async def finish_voice_upload_batch(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str,
    seconds_added: float,
    instant_voice_id_before: str | None,
    repository: Any = None,
    rebuild: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Make the audible voice include the speech a Voice-section batch added.

    A clone that did not exist before the batch was built from the batch's own
    clips by ``add_voice_clip``, so nothing more is needed. A clone that did
    exist is rebuilt once, preferring the newest clips, because
    ``ensure_instant_voice`` never retrains an existing clone on its own.
    Returns the voice record after a rebuild, or ``None`` when no rebuild ran.
    """
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.corpus import rebuild_instant_voice, voice_configured

    if seconds_added <= 0 or not instant_voice_id_before:
        return None
    voice_repository = repository or get_media_asset_repository()
    if voice_repository is None or not voice_configured(context):
        return None
    rebuild_voice = rebuild or rebuild_instant_voice
    try:
        rebuilt_record = await rebuild_voice(
            voice_repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            avatar_name=avatar_name,
            newest_first=True,
        )
    except Exception as rebuild_error:  # noqa: BLE001 - the clips are already stored
        logger.warning(
            "Rebuilding the instant voice of %s after a voice upload failed: %s",
            assistant_id,
            rebuild_error,
        )
        return None
    logger.info(
        "Rebuilt the instant voice of %s after a voice upload added %.1fs",
        assistant_id,
        seconds_added,
    )
    return rebuilt_record
