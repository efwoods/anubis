"""Uploaded images the owner can promote to the avatar's portrait.

The avatar holds exactly one active portrait (see ``reference_image.py``). An
ordinary image upload used to keep only a text description, so an image the
owner had already uploaded could never become the portrait without uploading
the image again. Every uploaded image is now also kept as a *portrait
candidate*:

``(user_id, assistant_id, "portrait_candidate", namespace_filename)`` with key
``"image"`` and the value::

    {
        "image_data": <data URI>,
        "filename": <upload filename>,
        "namespace_filename": <namespace_filename>,
        "portrait_key": <sha256 of the data URI>,
        # Present once the image has been analysed as a portrait:
        "reference_document": Document.to_json(),
        **assessment_store_fields(...),
    }

The namespace ends with the upload's ``namespace_filename`` so the existing
``DELETE /delete_avatar_document`` SQL (``prefix LIKE user.assistant.%.<name>``)
removes the candidate together with the upload. The row is written with
``index=False`` because the store's vector index embeds
``document.kwargs.page_content`` and a portrait candidate is not retrievable
source material.

Cost rule: the portrait analysis (a first-person description plus the subject
and moderation assessment) runs at most once per image. The analysis is kept on
the candidate, so switching back to a portrait analysed earlier costs no model
call. Generated emotion media follows the same rule: media belonging to a
previous portrait is parked under ``portrait_key`` (see
``media_assets/repository.py``) and restored when that portrait is chosen again.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

PORTRAIT_CANDIDATE_NAMESPACE_CATEGORY = "portrait_candidate"
PORTRAIT_CANDIDATE_KEY = "image"
# The parking key for media that existed while the avatar had no portrait.
UNASSIGNED_PORTRAIT_KEY = "unassigned"


def portrait_candidate_namespace(
    user_id: str, assistant_id: str, namespace_filename: str
) -> tuple[str, str, str, str]:
    """Return the store namespace of one upload's portrait candidate."""
    return (
        user_id,
        assistant_id,
        PORTRAIT_CANDIDATE_NAMESPACE_CATEGORY,
        namespace_filename,
    )


def portrait_key_for_image(image_data_uri: str | None) -> str:
    """Fingerprint an image so generated media can be tied to one portrait.

    The fingerprint is computed from the image bytes, so the same picture
    uploaded twice under two filenames still maps to the same parked media.
    """
    if not image_data_uri:
        return UNASSIGNED_PORTRAIT_KEY
    return hashlib.sha256(image_data_uri.encode("utf-8")).hexdigest()[:32]


def portrait_candidate_filenames(store_items: Any) -> set[str]:
    """Return the ``namespace_filename`` of every stored portrait candidate."""
    namespace_filenames: set[str] = set()
    for item in store_items or []:
        item_namespace = getattr(item, "namespace", None)
        if item_namespace is None and isinstance(item, dict):
            item_namespace = item.get("namespace")
        if (
            isinstance(item_namespace, (list, tuple))
            and len(item_namespace) >= 4
            and item_namespace[2] == PORTRAIT_CANDIDATE_NAMESPACE_CATEGORY
        ):
            namespace_filenames.add(str(item_namespace[3]))
    return namespace_filenames


def _store_item_value(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    return dict(value) if isinstance(value, dict) else None


async def read_portrait_candidate(
    store: Any, *, user_id: str, assistant_id: str, namespace_filename: str
) -> dict[str, Any] | None:
    """Return one upload's portrait candidate, or ``None`` when not kept."""
    try:
        item = await store.aget(
            portrait_candidate_namespace(user_id, assistant_id, namespace_filename),
            PORTRAIT_CANDIDATE_KEY,
        )
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Portrait candidate lookup failed: %s", read_error)
        return None
    candidate_value = _store_item_value(item)
    if not candidate_value or not str(candidate_value.get("image_data") or ""):
        return None
    return candidate_value


def candidate_reference_analysis(
    candidate_value: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]] | None:
    """Return ``(reference_document_json, assessment_fields)`` when cached.

    ``None`` means the image has not been analysed as a portrait yet, or the
    cached analysis belongs to different image bytes.
    """
    from src.anubis.utils.media_generation.reference_subject import (
        assessment_from_store_value,
        assessment_store_fields,
    )

    if not candidate_value:
        return None
    reference_document_json = candidate_value.get("reference_document")
    if not reference_document_json:
        return None
    analysed_portrait_key = candidate_value.get("analysed_portrait_key")
    if analysed_portrait_key and analysed_portrait_key != portrait_key_for_image(
        candidate_value.get("image_data")
    ):
        return None
    assessment = assessment_from_store_value(candidate_value)
    if assessment is None:
        return None
    return reference_document_json, assessment_store_fields(assessment)


async def store_portrait_candidate(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    namespace_filename: str,
    filename: str,
    image_data_uri: str,
    reference_document_json: Any = None,
    assessment_fields: dict[str, Any] | None = None,
) -> None:
    """Keep an uploaded image so the owner can promote the image later.

    A portrait analysis already cached for the same image bytes is kept when
    the caller supplies none, so re-uploading an image never discards work
    that was paid for.
    """
    if not namespace_filename or not image_data_uri:
        return
    portrait_key = portrait_key_for_image(image_data_uri)
    candidate_value: dict[str, Any] = {
        "image_data": image_data_uri,
        "filename": filename,
        "namespace_filename": namespace_filename,
        "portrait_key": portrait_key,
    }
    if reference_document_json is not None and assessment_fields:
        candidate_value["reference_document"] = reference_document_json
        candidate_value["analysed_portrait_key"] = portrait_key
        candidate_value.update(assessment_fields)
    else:
        existing_candidate = await read_portrait_candidate(
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            namespace_filename=namespace_filename,
        )
        if (
            existing_candidate
            and existing_candidate.get("portrait_key") == portrait_key
            and candidate_reference_analysis(existing_candidate) is not None
        ):
            candidate_value = {**existing_candidate, **candidate_value}
    await store.aput(
        portrait_candidate_namespace(user_id, assistant_id, namespace_filename),
        PORTRAIT_CANDIDATE_KEY,
        candidate_value,
        index=False,
    )


async def switch_portrait_media(
    repository: Any,
    *,
    assistant_id: str,
    previous_image_data_uri: str | None,
    next_image_data_uri: str | None,
) -> dict[str, int]:
    """Park the previous portrait's generated media; restore the next one's.

    Generated stills, idle loops and lip-sync clips show one face, so the
    media of the previous portrait must stop showing against the new portrait.
    The media is parked, never deleted: the owner paid for the generation, and
    switching back to the previous portrait brings the media back.
    """
    if repository is None:
        return {"parked": 0, "restored": 0}
    previous_portrait_key = portrait_key_for_image(previous_image_data_uri)
    next_portrait_key = portrait_key_for_image(next_image_data_uri)
    try:
        return await repository.switch_portrait_media(
            assistant_id, previous_portrait_key, next_portrait_key
        )
    except Exception:  # noqa: BLE001 - parking media must not fail the switch
        logger.warning(
            "Could not park generated media for %s", assistant_id, exc_info=True
        )
        return {"parked": 0, "restored": 0}


__all__ = [
    "PORTRAIT_CANDIDATE_KEY",
    "PORTRAIT_CANDIDATE_NAMESPACE_CATEGORY",
    "UNASSIGNED_PORTRAIT_KEY",
    "candidate_reference_analysis",
    "portrait_candidate_filenames",
    "portrait_candidate_namespace",
    "portrait_key_for_image",
    "read_portrait_candidate",
    "store_portrait_candidate",
    "switch_portrait_media",
]
