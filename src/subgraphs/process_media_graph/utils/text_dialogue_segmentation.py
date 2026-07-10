"""Turn long-form text into golden-format speaker turns with an inferred target.

Text sources (film and interview transcripts, scripture-style narrative) are
segmented into speaker turns so the SAME pipeline that consumes diarized audio
(:func:`process_dialogue_json_to_documents`) can consume text. The target
speaker is inferred from content — no caller-supplied parameter names the
target. The produced segments follow the golden format: the target's speech is
relabeled ``"avatar"`` and coalesced into long turns; every other individual
keeps their inferred name.

Long inputs are processed in sequential character windows. Speaker roster and
last-attributed-speaker state chain from one window to the next so an unmarked
continuation line at a window boundary is attributed correctly, then the whole
concatenated turn list is folded (narration into the surrounding speaker),
relabeled (target -> avatar), and coalesced once.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Speaker label used for narration, stage directions, and sound cues before
# folding relabels them onto the surrounding speaker.
NARRATOR_SPEAKER_LABEL = "narrator"


class SegmentedSpeakerTurn(BaseModel):
    """One attributed unit of text within a window."""

    speaker: str
    text: str
    is_speech: bool


class SpeakerRosterEntry(BaseModel):
    """A discovered speaker and a one-line identifying descriptor."""

    name: str
    description: str


class WindowSegmentationResult(BaseModel):
    """Structured result of segmenting one window of source text."""

    reasoning: str
    segments: List[SegmentedSpeakerTurn]
    updated_roster: List[SpeakerRosterEntry]
    final_attributed_speaker: str


class TargetSpeakerInference(BaseModel):
    """Which single individual is the inferred target of the content."""

    reasoning: str
    has_identifiable_target: bool
    target_name: Optional[str]
    matching_roster_names: List[str]


def split_text_into_dialogue_windows(
    text: str, *, window_characters: int
) -> List[str]:
    """Split ``text`` into windows on line boundaries, never mid-line.

    Prefers a blank-line boundary near the target size, then any newline; only
    when a single line already exceeds ``window_characters`` is that line
    emitted on its own. Deterministic so the same input always windows the same
    way.
    """
    if not text:
        return []
    if window_characters <= 0:
        return [text]

    lines = text.splitlines(keepends=True)
    windows: List[str] = []
    current: List[str] = []
    current_length = 0
    for line in lines:
        if current_length + len(line) > window_characters and current:
            windows.append("".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line)
    if current:
        windows.append("".join(current))
    return windows


async def segment_dialogue_window(
    window_text: str,
    *,
    roster: List[Dict[str, str]],
    last_attributed_speaker: Optional[str],
    previous_turn_tail: List[Dict[str, Any]],
) -> WindowSegmentationResult:
    """Segment one window into attributed speaker turns via structured output."""
    import json

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.text_dialogue_segmentation_prompt import (
        TEXT_DIALOGUE_SEGMENTATION_SYSTEM_PROMPT,
    )

    previous_turns_rendered = "\n".join(
        f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}"
        for turn in previous_turn_tail
    )
    human_message = "\n\n".join(
        [
            "Known-speaker roster (JSON list of {name, description}):\n"
            + json.dumps(roster, ensure_ascii=False),
            f"Last attributed speaker from the previous window: "
            f"{last_attributed_speaker or 'none'}",
            "Final turns of the previous window (READ-ONLY context; do NOT "
            "re-emit these):\n" + (previous_turns_rendered or "none"),
            "Current window text to segment:\n" + window_text,
        ]
    )

    model = init_model(
        model_without_tools=False, response_format=WindowSegmentationResult
    )
    return await model.ainvoke(
        input=[
            SystemMessage(content=TEXT_DIALOGUE_SEGMENTATION_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]
    )


def _merge_roster(
    existing: List[Dict[str, str]], updates: List[SpeakerRosterEntry]
) -> List[Dict[str, str]]:
    """Case-insensitive roster union; first-seen description wins."""
    by_lower_name: Dict[str, Dict[str, str]] = {
        entry["name"].strip().lower(): entry for entry in existing
    }
    for update in updates:
        name = (update.name or "").strip()
        if not name:
            continue
        key = name.lower()
        if key not in by_lower_name:
            by_lower_name[key] = {
                "name": name,
                "description": (update.description or "").strip(),
            }
    return list(by_lower_name.values())


async def segment_text_into_speaker_turns(
    text: str, *, window_characters: int, max_characters: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Segment ``text`` sequentially into ``(raw_segments, roster)``.

    Windows are processed in order because the roster and the last-attributed
    speaker chain across windows. Content beyond ``max_characters`` is skipped
    with a warning; the prefix still yields documents.
    """
    if not (text or "").strip():
        return [], []

    if max_characters and len(text) > max_characters:
        logger.warning(
            "text length %d exceeds segmentation cap %d; segmenting the prefix "
            "only",
            len(text),
            max_characters,
        )
        text = text[:max_characters]

    windows = split_text_into_dialogue_windows(
        text, window_characters=window_characters
    )
    raw_segments: List[Dict[str, Any]] = []
    roster: List[Dict[str, str]] = []
    last_attributed_speaker: Optional[str] = None

    for window_text in windows:
        if not window_text.strip():
            continue
        try:
            result = await segment_dialogue_window(
                window_text,
                roster=roster,
                last_attributed_speaker=last_attributed_speaker,
                previous_turn_tail=raw_segments[-2:],
            )
        except Exception as window_error:  # noqa: BLE001 - skip a bad window
            logger.warning(
                "window segmentation failed (%s); skipping this window",
                window_error,
            )
            continue

        for turn in result.segments:
            turn_text = (turn.text or "").strip()
            if not turn_text:
                continue
            # Fuzzy verbatim-fidelity guard: the model should echo the window
            # verbatim. Log (do not drop) when a turn is not found in the source
            # window so mutations are observable.
            if turn_text not in window_text and turn_text[:40] not in window_text:
                logger.warning(
                    "segmented turn text not found verbatim in source window "
                    "(speaker=%r): %r",
                    turn.speaker,
                    turn_text[:80],
                )
            raw_segments.append(
                {
                    "speaker": str(turn.speaker or NARRATOR_SPEAKER_LABEL),
                    "text": turn_text,
                    "is_speech": bool(turn.is_speech),
                }
            )
        roster = _merge_roster(roster, result.updated_roster)
        if (result.final_attributed_speaker or "").strip():
            last_attributed_speaker = result.final_attributed_speaker.strip()

    return raw_segments, roster


def fold_narrator_segments(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Relabel non-speech ("narrator") segments onto the surrounding speaker.

    A non-speech segment (stage direction, sound cue, narration) folds into the
    nearest PRECEDING speaking segment's speaker so the action line lands inside
    that turn after coalescing; a leading non-speech segment folds into the
    nearest following speaking segment. This matches the worked example in
    ``_PREPROCESSING_PROCESS.md`` where "[line clicks]" belongs to the
    surrounding non-target (user) turn. When no speaking segment exists at all,
    segments are left under the narrator label.
    """
    if not segments:
        return []

    # Nearest following speaking speaker for each index (for leading narration).
    following_speaker: List[Optional[str]] = [None] * len(segments)
    next_speaker: Optional[str] = None
    for index in range(len(segments) - 1, -1, -1):
        following_speaker[index] = next_speaker
        if segments[index].get("is_speech"):
            next_speaker = str(segments[index].get("speaker") or NARRATOR_SPEAKER_LABEL)

    folded: List[Dict[str, Any]] = []
    preceding_speaker: Optional[str] = None
    for index, segment in enumerate(segments):
        if segment.get("is_speech"):
            preceding_speaker = str(segment.get("speaker") or NARRATOR_SPEAKER_LABEL)
            folded.append(dict(segment))
            continue
        # Non-speech: fold onto the surrounding speaker.
        target_speaker = preceding_speaker or following_speaker[index]
        new_segment = dict(segment)
        if target_speaker is not None:
            new_segment["speaker"] = target_speaker
        folded.append(new_segment)
    return folded


async def infer_target_speaker(
    *,
    roster: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
    classification_target_name: Optional[str],
    filename: str,
) -> Dict[str, Any]:
    """Infer the single target speaker from the roster and segment evidence."""
    import json
    from collections import Counter

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.text_dialogue_segmentation_prompt import (
        TARGET_SPEAKER_INFERENCE_SYSTEM_PROMPT,
    )

    turn_counts = Counter(str(seg.get("speaker") or "unknown") for seg in segments)
    sample_turns: Dict[str, List[str]] = {}
    for seg in segments:
        speaker = str(seg.get("speaker") or "unknown")
        if len(sample_turns.setdefault(speaker, [])) < 3:
            sample_turns[speaker].append((seg.get("text") or "")[:200])

    roster_summary = [
        {
            "name": entry.get("name"),
            "description": entry.get("description"),
            "turn_count": turn_counts.get(entry.get("name", ""), 0),
            "sample_turns": sample_turns.get(entry.get("name", ""), []),
        }
        for entry in roster
    ]
    # Include any speaker labels that appear in segments but not the roster.
    for speaker, count in turn_counts.items():
        if not any(entry.get("name") == speaker for entry in roster_summary):
            roster_summary.append(
                {
                    "name": speaker,
                    "description": "",
                    "turn_count": count,
                    "sample_turns": sample_turns.get(speaker, []),
                }
            )

    human_message = "\n\n".join(
        [
            f"Source: {filename or 'unknown'}",
            f"Prior classifier target guess (may be empty): "
            f"{classification_target_name or 'none'}",
            "Speaker roster with turn counts and sample turns (JSON):\n"
            + json.dumps(roster_summary, ensure_ascii=False),
        ]
    )

    model = init_model(
        model_without_tools=False, response_format=TargetSpeakerInference
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=TARGET_SPEAKER_INFERENCE_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]
    )
    return {
        "has_identifiable_target": bool(
            getattr(response, "has_identifiable_target", False)
        ),
        "target_name": getattr(response, "target_name", None),
        "matching_roster_names": list(
            getattr(response, "matching_roster_names", []) or []
        ),
    }


def relabel_target_segments(
    segments: List[Dict[str, Any]], *, target_roster_names: List[str]
) -> List[Dict[str, Any]]:
    """Relabel the target's segments to ``avatar``/is_target; others unchanged."""
    target_lower = {name.strip().lower() for name in target_roster_names if name}
    relabeled: List[Dict[str, Any]] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "unknown")
        new_segment = dict(segment)
        if speaker.strip().lower() in target_lower:
            new_segment["speaker"] = "avatar"
            new_segment["is_target"] = True
        else:
            new_segment["is_target"] = False
        relabeled.append(new_segment)
    return relabeled


def _target_not_identifiable_document(
    media_item: Dict[str, Any],
) -> Document:
    """Error Document surfaced when no single target can be inferred from text."""
    metadata = media_item.get("metadata", {}) or {}
    return Document(
        page_content=(
            "[No single target speaker could be inferred from this text, so it "
            "cannot be segmented into target quotes and biographical facts.]"
        ),
        metadata={
            "status": "error",
            "error": "dialogue_text_target_not_identifiable",
            "filename": metadata.get("filename", ""),
            "namespace_filename": metadata.get("namespace_filename", ""),
        },
    )


async def convert_text_dialogue_to_documents(
    text_content: str,
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    classification_target_name: Optional[str] = None,
) -> List[Document]:
    """Segment text, infer the target, and route through the dialogue pipeline.

    Produces the golden-format ``dialogue_payload`` (target speech relabeled
    ``avatar`` and coalesced, other individuals keeping their names) and feeds
    it to :func:`process_dialogue_json_to_documents` so text yields the same
    quote / adapter-conversation / biographical Documents that diarized audio
    does. Returns a single error Document when no target is inferable.
    """
    from src.anubis.utils.context import GlobalContext
    from src.subgraphs.process_media_graph.utils.helper_functions import (
        coalesce_segments_by_speaker,
        process_dialogue_json_to_documents,
    )

    context = GlobalContext()
    metadata = media_item.get("metadata", {}) or {}
    filename = metadata.get("filename", "")

    raw_segments, roster = await segment_text_into_speaker_turns(
        text_content,
        window_characters=int(
            context.text_dialogue_segmentation_window_characters or 4000
        ),
        max_characters=int(
            context.text_dialogue_segmentation_max_characters or 250000
        ),
    )
    if not raw_segments:
        return [_target_not_identifiable_document(media_item)]

    folded_segments = fold_narrator_segments(raw_segments)
    inference = await infer_target_speaker(
        roster=roster,
        segments=folded_segments,
        classification_target_name=classification_target_name,
        filename=filename,
    )
    if not inference["has_identifiable_target"]:
        return [_target_not_identifiable_document(media_item)]

    target_roster_names = list(inference["matching_roster_names"])
    if inference["target_name"]:
        target_roster_names.append(inference["target_name"])
    relabeled_segments = relabel_target_segments(
        folded_segments, target_roster_names=target_roster_names
    )
    coalesced_segments = coalesce_segments_by_speaker(relabeled_segments)

    dialogue_payload = {
        "segments": coalesced_segments,
        "target_name": "avatar",
        "speakers": roster,
    }
    return await process_dialogue_json_to_documents(
        dialogue_payload=dialogue_payload,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
    )
