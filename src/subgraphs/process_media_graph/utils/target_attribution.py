"""Post-diarization target-speaker attribution.

A recording longer than the single-request diarization size limit is split into
chunks that are diarized independently, so the target's speech is scattered
across per-chunk speaker labels (``chunk_1.speaker_0``) and only the segments
the diarizer confidently voice-matched to the short reference clip carry the
known-speaker label. This module runs one structured-output adjudication call
that reads the whole labeled transcript and decides, for each distinct speaker
label, whether the person behind that label is the single target speaker.

The caller applies the returned map as a UNION with the diarizer's own
known-speaker votes (a label is only ever promoted to the target, never
demoted) and then re-coalesces so the target's turns merge under one label.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

try:  # Literal is in typing on Python 3.11; keep the import defensive.
    from typing import Literal
except ImportError:  # pragma: no cover - Python < 3.8 not supported here
    from typing_extensions import Literal  # type: ignore

logger = logging.getLogger(__name__)


class SpeakerLabelAttribution(BaseModel):
    """Whether one diarized speaker label belongs to the target speaker."""

    speaker_label: str
    belongs_to_target: bool
    confidence: Literal["high", "medium", "low"]
    evidence_summary: str


class TargetSpeakerAttributionResponse(BaseModel):
    """One attribution per distinct speaker label in the transcript."""

    attributions: List[SpeakerLabelAttribution]


def _render_labeled_transcript(turns: List[Dict[str, Any]]) -> str:
    """Render coalesced turns as ``speaker_label: text`` lines for the model."""
    lines: List[str] = []
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "unknown")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _distinct_labels_in_order(turns: List[Dict[str, Any]]) -> List[str]:
    """Distinct speaker labels, preserving first-appearance order."""
    seen: Dict[str, None] = {}
    for turn in turns:
        speaker = str(turn.get("speaker") or "unknown")
        if speaker not in seen:
            seen[speaker] = None
    return list(seen.keys())


async def _adjudicate_one_transcript(
    turns: List[Dict[str, Any]],
    *,
    reference_transcript_text: str,
    target_name: Optional[str],
    target_speaker_label: str,
) -> Optional[Dict[str, bool]]:
    """Run one adjudication call over a single (already-scoped) turn list.

    Returns ``{speaker_label: belongs_to_target}`` for the labels present, or
    ``None`` on any failure so the caller can fall back to diarizer votes only.
    """
    distinct_labels = _distinct_labels_in_order(turns)
    if not distinct_labels:
        return {}

    normalized_target_label = target_speaker_label.strip().lower()
    confirmed_target_labels = [
        label
        for label in distinct_labels
        if label.strip().lower() == normalized_target_label
    ]

    # Lazy import so a cold graph start does not pull the model SDK.
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.target_speaker_attribution_prompt import (
        TARGET_SPEAKER_ATTRIBUTION_SYSTEM_PROMPT,
    )

    human_message_sections = [
        f"Target name (may be empty): {target_name or 'unknown'}",
        (
            "Speaker labels already confirmed as the target by the voice "
            f"matcher: {confirmed_target_labels or 'none'}"
        ),
        (
            "Verbatim transcript of the target's own reference clip (may be a "
            "generic calibration sentence with no biographical content):\n"
            f"{(reference_transcript_text or '').strip() or 'unavailable'}"
        ),
        (
            "Every distinct speaker label you must attribute exactly once:\n"
            + "\n".join(f"- {label}" for label in distinct_labels)
        ),
        (
            "Full labeled transcript (one turn per line as "
            "'speaker_label: text'):\n" + _render_labeled_transcript(turns)
        ),
    ]
    human_message = "\n\n".join(human_message_sections)

    model = init_model(
        model_without_tools=False,
        response_format=TargetSpeakerAttributionResponse,
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=TARGET_SPEAKER_ATTRIBUTION_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]
    )

    attributions = getattr(response, "attributions", None)
    if not attributions:
        return None

    valid_labels = set(distinct_labels)
    attribution_map: Dict[str, bool] = {label: False for label in distinct_labels}
    for attribution in attributions:
        label = getattr(attribution, "speaker_label", None)
        if label not in valid_labels:
            # The model returned a label that is not in the provided set; drop
            # it rather than trust a hallucinated label.
            continue
        confidence = getattr(attribution, "confidence", "low")
        belongs = bool(getattr(attribution, "belongs_to_target", False))
        # Only "high" / "medium" confidence promotes a label to the target;
        # "low" confidence stays non-target so ambiguous speech is never
        # credited to the target.
        attribution_map[label] = belongs and confidence in ("high", "medium")

    # A voice-matcher-confirmed label is the target by definition regardless of
    # what the model said, so union those in.
    for label in confirmed_target_labels:
        attribution_map[label] = True

    return attribution_map


async def adjudicate_target_speaker_labels(
    turns: List[Dict[str, Any]],
    *,
    reference_transcript_text: str,
    target_name: Optional[str],
    target_speaker_label: str,
    context: Any,
) -> Optional[Dict[str, bool]]:
    """Map each distinct speaker label to whether the speaker is the target.

    ``turns`` is the coalesced turn list (each carrying ``speaker``, ``text``,
    and ``chunk_idx``). Returns ``{speaker_label: belongs_to_target}`` for every
    distinct label, or ``None`` on failure so the caller keeps the diarizer's
    own known-speaker votes only.

    When the rendered transcript exceeds
    ``target_speaker_attribution_transcript_character_limit``, the turns are
    grouped by ``chunk_idx`` and adjudicated one chunk at a time; the per-chunk
    maps are merged. Per-chunk speaker labels are namespaced with the chunk
    index upstream, so labels never collide across chunk groups.
    """
    if not turns:
        return {}

    character_limit = int(
        getattr(context, "target_speaker_attribution_transcript_character_limit", 0)
        or 100000
    )

    try:
        full_transcript = _render_labeled_transcript(turns)
        if len(full_transcript) <= character_limit:
            return await _adjudicate_one_transcript(
                turns,
                reference_transcript_text=reference_transcript_text,
                target_name=target_name,
                target_speaker_label=target_speaker_label,
            )

        # Length fallback: adjudicate per chunk group and merge. Labels are
        # chunk-namespaced upstream, so the merged map has no collisions.
        chunk_groups: Dict[int, List[Dict[str, Any]]] = {}
        for turn in turns:
            chunk_index = int(turn.get("chunk_idx") or 0)
            chunk_groups.setdefault(chunk_index, []).append(turn)

        merged_map: Dict[str, bool] = {}
        for chunk_index in sorted(chunk_groups.keys()):
            chunk_map = await _adjudicate_one_transcript(
                chunk_groups[chunk_index],
                reference_transcript_text=reference_transcript_text,
                target_name=target_name,
                target_speaker_label=target_speaker_label,
            )
            if chunk_map is None:
                # A single chunk failing must not discard the whole pass; keep
                # what succeeded and default the rest to diarizer votes only.
                logger.warning(
                    "target attribution failed for chunk %s; keeping diarizer "
                    "votes for that chunk",
                    chunk_index,
                )
                continue
            merged_map.update(chunk_map)
        return merged_map or None
    except Exception as attribution_error:  # noqa: BLE001 - best-effort pass
        logger.warning(
            "target speaker attribution failed (%s); falling back to diarizer "
            "votes only",
            attribution_error,
        )
        return None


class AvatarSpeakerIdentification(BaseModel):
    """Which diarized speaker labels are the named avatar, judged from text."""

    verdict: Literal["identified", "absent", "undetermined"]
    avatar_speaker_labels: List[str]
    confidence: Literal["high", "medium", "low"]
    evidence_summary: str


AVATAR_SPEAKER_IDENTIFICATION_TRANSCRIPT_CHARACTER_LIMIT = 60000


def _transcript_excerpt_for_identification(
    turns: List[Dict[str, Any]], character_limit: int
) -> str:
    """Render the transcript, keeping the opening and closing when too long.

    Introductions and sign-offs carry most of the naming evidence, so an
    over-long transcript keeps the first two thirds of the character limit
    from the opening and the last third from the closing.
    """
    full_transcript = _render_labeled_transcript(turns)
    if len(full_transcript) <= character_limit:
        return full_transcript
    opening_character_count = (character_limit * 2) // 3
    closing_character_count = character_limit - opening_character_count
    return (
        full_transcript[:opening_character_count]
        + "\n[... middle of the transcript omitted ...]\n"
        + full_transcript[-closing_character_count:]
    )


async def identify_avatar_speaker_labels(
    turns: List[Dict[str, Any]],
    *,
    avatar_name: Optional[str],
) -> Optional[AvatarSpeakerIdentification]:
    """Decide from the transcript which speaker labels are the named avatar.

    The reference identity check. A promoted reference upload has no voice
    reference yet, and the dominant speaker by talk time can be the
    interviewer. One text-only structured-output call over the transcript the
    diarizer call already paid for reads self-introductions and address by
    name. No audio is sent, so the identity check adds no speech cost.

    Returns ``None`` when the avatar has no name, the transcript is empty, or
    the call fails; the caller then keeps the dominant-speaker rule. Only a
    "high" or "medium" confidence verdict is returned as decisive: a "low"
    confidence verdict is downgraded to "undetermined" with no labels.
    """
    cleaned_avatar_name = (avatar_name or "").strip()
    distinct_labels = _distinct_labels_in_order(turns)
    if not cleaned_avatar_name or not distinct_labels:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_model
        from src.anubis.utils.prompts.target_speaker_attribution_prompt import (
            AVATAR_SPEAKER_IDENTIFICATION_SYSTEM_PROMPT,
        )

        human_message = "\n\n".join(
            [
                f"Avatar name: {cleaned_avatar_name}",
                "Every distinct speaker label:\n"
                + "\n".join(f"- {label}" for label in distinct_labels),
                "Labeled transcript (one turn per line as 'speaker_label: text'):\n"
                + _transcript_excerpt_for_identification(
                    turns, AVATAR_SPEAKER_IDENTIFICATION_TRANSCRIPT_CHARACTER_LIMIT
                ),
            ]
        )
        model = init_model(
            model_without_tools=False,
            response_format=AvatarSpeakerIdentification,
        )
        response = await model.ainvoke(
            input=[
                SystemMessage(content=AVATAR_SPEAKER_IDENTIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=human_message),
            ]
        )
    except Exception as identification_error:  # noqa: BLE001 - best-effort pass
        logger.warning(
            "avatar speaker identification failed (%s); keeping the "
            "dominant-speaker rule",
            identification_error,
        )
        return None

    verdict = getattr(response, "verdict", None)
    if verdict not in ("identified", "absent", "undetermined"):
        return None
    confidence = getattr(response, "confidence", "low")
    evidence_summary = str(getattr(response, "evidence_summary", "") or "")
    valid_labels = set(distinct_labels)
    avatar_speaker_labels = [
        label
        for label in (getattr(response, "avatar_speaker_labels", None) or [])
        if label in valid_labels
    ]
    if confidence not in ("high", "medium") or (
        verdict == "identified" and not avatar_speaker_labels
    ):
        return AvatarSpeakerIdentification(
            verdict="undetermined",
            avatar_speaker_labels=[],
            confidence="low",
            evidence_summary=evidence_summary,
        )
    return AvatarSpeakerIdentification(
        verdict=verdict,
        avatar_speaker_labels=avatar_speaker_labels if verdict == "identified" else [],
        confidence=confidence,
        evidence_summary=evidence_summary,
    )
