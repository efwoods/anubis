"""Score a diarizer's output against a transcript whose speakers are known.

Used by ``scripts/diarization_bakeoff.py`` to compare hosted diarizers on the
same recordings. The reference is a transcript published by the recording's
source with every turn named (``published_transcripts.py``); the hypothesis is
one provider's diarized segments. Everything is computed on words, so no audio
tooling is needed:

- **word error rate**: how many word substitutions, deletions and insertions
  turn the provider's words into the reference words, per reference word;
- **speaker attribution accuracy**: of the words both transcripts share, the
  share the provider placed under the right speaker, after each provider label
  is mapped to the reference speaker the label's words most often belong to;
- **avatar precision and recall**: the same, for the one speaker whose turns
  become the avatar's own words.

All functions are pure.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

_WORD_PATTERN = re.compile(r"[a-z0-9']+")


def normalized_words(text: str) -> List[str]:
    """Lower-case words with punctuation removed, apostrophes dropped."""
    return [word.replace("'", "") for word in _WORD_PATTERN.findall((text or "").lower()) if word.replace("'", "")]


def speaker_labelled_words(segments: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Flatten segments into ``(word, speaker)`` pairs in spoken order."""
    labelled: List[Tuple[str, str]] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "unknown")
        labelled.extend((word, speaker) for word in normalized_words(segment.get("text") or ""))
    return labelled


def segments_within(
    segments: Sequence[Dict[str, Any]], maximum_seconds: float | None
) -> List[Dict[str, Any]]:
    """Keep the segments that start before ``maximum_seconds``."""
    if maximum_seconds is None:
        return list(segments)
    return [
        segment
        for segment in segments
        if segment.get("start") is not None and float(segment["start"]) < maximum_seconds
    ]


def word_error_rate(reference_words: Sequence[str], hypothesis_words: Sequence[str]) -> float | None:
    """Word-level edit distance divided by the number of reference words."""
    if not reference_words:
        return None
    previous_row = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index] + [0] * len(hypothesis_words)
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous_row[hypothesis_index - 1] + (reference_word != hypothesis_word)
            deletion = previous_row[hypothesis_index] + 1
            insertion = current_row[hypothesis_index - 1] + 1
            current_row[hypothesis_index] = min(substitution, deletion, insertion)
        previous_row = current_row
    return previous_row[-1] / len(reference_words)


def speaker_attribution_scores(
    reference_segments: Sequence[Dict[str, Any]],
    hypothesis_segments: Sequence[Dict[str, Any]],
    *,
    avatar_speaker_name: str | None = None,
) -> Dict[str, Any]:
    """Score who-said-what on the words the two transcripts share.

    The two word sequences are aligned on identical words. Each hypothesis
    label is mapped to the reference speaker who owns the most of that label's
    aligned words. ``speaker_attribution_accuracy`` is the share of aligned
    words whose mapped label equals the reference speaker.

    With ``avatar_speaker_name`` given, ``avatar_precision`` is the share of
    aligned words placed under a label mapped to the avatar that the avatar
    really spoke, and ``avatar_recall`` is the share of the avatar's aligned
    words that were placed under such a label. A value that cannot be computed
    (no aligned words, or no label mapped to the avatar) is ``None``.
    """
    reference = speaker_labelled_words(reference_segments)
    hypothesis = speaker_labelled_words(hypothesis_segments)
    matcher = difflib.SequenceMatcher(
        None,
        [word for word, _speaker in reference],
        [word for word, _speaker in hypothesis],
        autojunk=False,
    )
    aligned: List[Tuple[str, str]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            aligned.append(
                (reference[block.a + offset][1], hypothesis[block.b + offset][1])
            )

    votes: Dict[str, Counter] = {}
    for reference_speaker, hypothesis_label in aligned:
        votes.setdefault(hypothesis_label, Counter())[reference_speaker] += 1
    label_to_speaker = {
        label: counter.most_common(1)[0][0] for label, counter in votes.items()
    }

    correct = sum(
        1
        for reference_speaker, hypothesis_label in aligned
        if label_to_speaker[hypothesis_label] == reference_speaker
    )
    scores: Dict[str, Any] = {
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        "aligned_words": len(aligned),
        "speaker_attribution_accuracy": (correct / len(aligned)) if aligned else None,
        "label_to_speaker": label_to_speaker,
        "reference_speaker_count": len({speaker for _word, speaker in reference}),
        "hypothesis_label_count": len({speaker for _word, speaker in hypothesis}),
        "avatar_precision": None,
        "avatar_recall": None,
    }
    if avatar_speaker_name:
        placed_under_avatar = [
            reference_speaker
            for reference_speaker, hypothesis_label in aligned
            if label_to_speaker[hypothesis_label] == avatar_speaker_name
        ]
        avatar_words = [
            hypothesis_label
            for reference_speaker, hypothesis_label in aligned
            if reference_speaker == avatar_speaker_name
        ]
        if placed_under_avatar:
            scores["avatar_precision"] = (
                placed_under_avatar.count(avatar_speaker_name) / len(placed_under_avatar)
            )
        if avatar_words:
            scores["avatar_recall"] = sum(
                1 for label in avatar_words if label_to_speaker[label] == avatar_speaker_name
            ) / len(avatar_words)
    return scores


def turn_structure(turns: Sequence[Dict[str, Any]], *, short_turn_characters: int = 40) -> Dict[str, Any]:
    """Count the turns and the turns too short to be a usable quote."""
    texts = [(turn.get("text") or "").strip() for turn in turns]
    texts = [text for text in texts if text]
    short_turns = sum(1 for text in texts if len(text) < short_turn_characters)
    return {
        "turns": len(texts),
        "turns_under_short_limit": short_turns,
        "share_of_turns_under_short_limit": (short_turns / len(texts)) if texts else None,
        "short_turn_characters": short_turn_characters,
    }
