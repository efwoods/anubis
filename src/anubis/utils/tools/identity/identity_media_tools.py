"""The in-chat tool that feeds attached or linked media into the avatar's identity.

Built per turn for the avatar's creator (the only person allowed to update an
avatar's identity) when the caller's subscription tier permits uploads. The
tool reaches the exact same pipeline as ``POST /update_avatar_identity_with_media``
— the media graph classifies, converts, and indexes the item, and the
reference-image and voice-corpus hooks run — through the starter the FastAPI
lifespan publishes in ``runtime_handles``. Files come from the turn's
attachment record (``src/api/chat_attachments.py``); links come straight from
the conversation.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

IDENTITY_MEDIA_TOOL_NAME = "update_avatar_identity_with_media"


def _emit_media_job_started(payload: dict[str, Any]) -> None:
    """Tell the streaming client a media job started so it can follow the progress."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - outside a graph run there is no stream
        return
    try:
        writer({"type": "media_job_started", **payload})
    except Exception:  # noqa: BLE001 - a client that cannot follow changes nothing
        logger.debug("media_job_started frame not delivered", exc_info=True)


def build_identity_media_tools(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    assistant_ctx: dict[str, Any],
    thread_id: str | None,
) -> list[Any]:
    """Build the identity-update tool bound to this creator, avatar, and turn."""
    from src.anubis.utils.runtime_handles import get_identity_media_job_starter

    @tool(IDENTITY_MEDIA_TOOL_NAME)
    async def update_avatar_identity_with_media(
        filenames: list[str] | None = None,
        urls: list[str] | None = None,
        reference_image: bool = False,
        reference_audio: bool = False,
    ) -> dict[str, Any]:
        """Learn from media of the avatar: add attached files or links to the avatar's identity.

        Call this when the conversation partner attaches files or shares links
        that ARE of the avatar — photos of the avatar, recordings or videos of
        the avatar speaking, the avatar's own writing, posts, transcripts,
        documents — or when the conversation partner asks the avatar to learn
        from, remember, or absorb the media ("learn from this", "this is you",
        "add this to your memory"). Do not call this for media shared only to
        discuss or ask about; ask first when the request is ambiguous.

        The ATTACHED_MEDIA section of the system prompt lists the files attached
        to this turn by filename. Pass the filenames to learn from, or leave
        filenames empty to learn from every attached file. Pass urls for links
        (web pages, YouTube videos or playlists, social posts).

        Set reference_image to true only when the conversation partner says the
        attached image is the avatar's portrait or reference photo (exactly one
        image). Set reference_audio to true only when the conversation partner
        says the attached recording is a reference clip of the avatar's own
        voice (exactly one audio file). Both default to false.

        Processing runs in the background; the result reports what was accepted
        and what was rejected. Tell the conversation partner briefly what is
        being learned and that it takes a few minutes, without listing job ids.

        Args:
            filenames: Attached filenames to learn from; empty means every attached file.
            urls: Links to learn from.
            reference_image: The single attached image is the avatar's reference portrait.
            reference_audio: The single attached recording is the avatar's reference voice clip.
        """
        from src.api.chat_attachments import get_turn_attachments

        starter = get_identity_media_job_starter()
        if starter is None:
            return {
                "status": "unavailable",
                "detail": (
                    "Identity updates from chat are not available in this deployment. "
                    "Ask the conversation partner to use the avatar settings upload."
                ),
            }
        record = get_turn_attachments(thread_id)
        available = list(record.attachments) if record is not None else []
        wanted_names = [
            name.strip() for name in (filenames or []) if name and name.strip()
        ]
        if wanted_names:
            by_name = {attachment.filename: attachment for attachment in available}
            missing = [name for name in wanted_names if name not in by_name]
            if missing:
                return {
                    "status": "not_found",
                    "detail": (
                        "These filenames were not attached to this turn: "
                        f"{', '.join(missing)}. Attached: "
                        f"{', '.join(sorted(by_name)) or 'nothing'}."
                    ),
                }
            selected = [by_name[name] for name in wanted_names]
        else:
            selected = available
        clean_urls = [url.strip() for url in (urls or []) if url and url.strip()]
        if not selected and not clean_urls:
            return {
                "status": "nothing_to_learn",
                "detail": (
                    "No files are attached to this turn and no links were given. "
                    "Ask the conversation partner to attach the media or share a link."
                ),
            }
        if reference_image and reference_audio:
            return {
                "status": "invalid",
                "detail": "Use only one of reference_image or reference_audio.",
            }
        if (reference_image or reference_audio) and (
            len(selected) + len(clean_urls)
        ) != 1:
            return {
                "status": "invalid",
                "detail": (
                    "A reference image or reference recording is a single item; "
                    "pass exactly one filename or url with that flag."
                ),
            }
        if record is None:
            return {
                "status": "unavailable",
                "detail": (
                    "This turn's caller could not be identified for metering; "
                    "the conversation partner should send the media again."
                ),
            }
        try:
            result = await starter(
                user_id=user_id,
                assistant_id=assistant_id,
                assistant_ctx=assistant_ctx,
                current_user=record.current_user,
                attachments=selected,
                urls=clean_urls,
                reference_image=bool(reference_image),
                reference_audio=bool(reference_audio),
            )
        except Exception as start_error:  # noqa: BLE001 - never fail the turn
            logger.warning("Identity media job could not start: %s", start_error)
            return {"status": "error", "detail": str(start_error)}
        if result.get("job_id"):
            described = [attachment.filename for attachment in selected] + clean_urls
            _emit_media_job_started(
                {
                    "job_id": result["job_id"],
                    "assistant_id": assistant_id,
                    "thread_id": thread_id,
                    "description": ", ".join(described[:3])
                    + (f" and {len(described) - 3} more" if len(described) > 3 else ""),
                    "items_accepted": result.get("items_accepted", 0),
                }
            )
        return result

    return [update_avatar_identity_with_media]
