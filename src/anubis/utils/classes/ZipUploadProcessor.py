"""Expand a ``.zip`` upload into individual media-file dicts.

The webapp posts each archive member back through the existing
``process_uploaded_files_and_label_media_type`` node, which already knows how
to label image/audio/video/text/json/pdf payloads.

Identity-write rule:
    This class is consumed by the ``update_avatar_identity_with_media``
    flow. By rule, on that endpoint the ``user_id`` and the ``creator_id``
    are always the same person — only the assistant's original "parent" /
    creator may write identity material into the avatar's vectorstore.

    ``creator_id`` is therefore not a separate parameter on :method:`expand`.
    Each produced media-file dict has ``creator_id == user_id``.

    The reason for keeping ``creator_id`` on the dicts at all (instead of
    just relying on ``user_id``) is the read side: at chat time
    ``load_consciousness`` uses ``creator_id`` to (a) load the known facts
    that the creator stored in the vectorstore so a non-creator chatting
    with the avatar still sees them, and (b) prevent the avatar from
    learning new facts about itself during conversation when the speaker is
    not the original creator. The two ids only diverge on the read side; on
    the write side handled here they are equal.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPES: Dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/mp4",
    ".3gp": "video/3gpp",
}


class ZipUploadProcessor:
    """Expand a zipped archive into a list of pseudo-``UploadFile`` dicts."""

    def expand(
        self,
        zip_bytes: bytes,
        *,
        user_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return one media-file dict per non-empty member of the archive."""
        if not zip_bytes:
            return []

        media_files: List[Dict[str, Any]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    name = Path(member.filename).name
                    if not name or name.startswith("."):
                        continue
                    suffix = Path(name).suffix.lower()
                    if not suffix:
                        continue
                    content_type = _DEFAULT_CONTENT_TYPES.get(
                        suffix, "application/octet-stream"
                    )
                    try:
                        with archive.open(member) as fh:
                            payload = fh.read()
                    except Exception as exc:
                        logger.warning(
                            "Failed to read %s from zip: %s", member.filename, exc
                        )
                        continue
                    if not payload:
                        continue
                    media_files.append(
                        {
                            "filename": name,
                            "content_type": content_type,
                            "content": payload,
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                        }
                    )
        except zipfile.BadZipFile as exc:
            logger.error("Invalid zip archive: %s", exc)
            return []

        return media_files
