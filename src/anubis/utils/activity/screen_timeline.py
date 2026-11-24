"""Turn the hidden screen observations of a period into a timeline of time blocks.

Each watched screen snapshot already carries what the activity review needs:
``captured_at`` (when), and the triage fields ``observation_kind`` (a free
lowercase label such as ``writing_code``, ``youtube_results`` or
``organization_page``) and ``summary`` (one line). The description's
``screen:`` section names the application, site, and window title.

Duration is measured from observation spacing, not from a clock on the
screen. An observation is taken to last until the next observation, capped at
``maximum_gap_seconds``: the browser sends a changed screen on the capture
interval and an unchanged screen once per quiet heartbeat, so a longer gap
means the share was off, the chat page was closed, or the machine slept. Time
beyond the cap is reported as not observed and never counted as activity.

Observations are grouped into fixed local-time blocks rather than into runs of
one ``observation_kind``. The kind label is chosen freely per snapshot, so the
label changes on nearly every snapshot of one piece of work (measured on
2026-11-17: five consecutive snapshots of one GitHub page came back as five
different labels), and runs of one label fragment into one segment per
snapshot. A block lists the seconds per label instead. The block length is the
shortest of ``BLOCK_MINUTES_CHOICES`` that keeps the number of non-empty blocks
within ``maximum_blocks``, so a day reads in quarter hours and a week in
longer blocks, and no observed second is ever dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.anubis.utils.ambient.observations import (
    ambient_details,
    is_ambient_observation,
    message_text,
    split_observation_text,
    strip_instruction,
)

SCREEN_SOURCE = "screen"
UNLABELLED_KIND = "unlabelled"
SCREEN_SECTION_PREFIX = "screen:"

BLOCK_MINUTES_CHOICES = (15, 30, 60, 120, 240, 480, 1440)
"""Block lengths tried in order; every length divides a day evenly."""

SUMMARIES_PER_BLOCK = 3
SCREEN_DESCRIPTIONS_PER_BLOCK = 2

DURATION_METHOD = (
    "Seconds are measured from the spacing between screen observations: each "
    "observation counts until the next one, capped at maximum_gap_seconds, and "
    "is credited to the block the observation was captured in. Time past the "
    "cap is not_observed_seconds and is not counted as activity. The figures "
    "are as exact as the observation spacing, not a stopwatch. Activity labels "
    "are chosen freely per snapshot, so read the labels and summaries of a "
    "block together rather than treating each label as a separate task."
)


@dataclass
class ScreenObservation:
    """One watched screen snapshot, as the timeline needs the snapshot."""

    captured_at: datetime
    observation_kind: str
    summary: str
    screen_description: str
    thread_id: str
    counted_seconds: float = 0.0


@dataclass
class ScreenBlock:
    """Every observation captured inside one local-time block."""

    started_at: datetime
    block_minutes: int
    observed_seconds: float = 0.0
    observation_count: int = 0
    seconds_by_activity: dict[str, float] = field(default_factory=dict)
    summaries: list[str] = field(default_factory=list)
    screen_descriptions: list[str] = field(default_factory=list)


def _parse_captured_at(stamp: Any) -> datetime | None:
    """Parse an ISO ``captured_at``; a naive stamp is read as UTC, the browser's format."""
    text = str(stamp or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def screen_section_of(body: str, maximum_characters: int) -> str:
    """Return the ``screen:`` section of an observation body, trimmed to a ceiling."""
    screen_lines: list[str] = []
    inside_screen_section = False
    for line in body.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith(SCREEN_SECTION_PREFIX):
            inside_screen_section = True
            stripped = stripped[len(SCREEN_SECTION_PREFIX) :].strip()
        elif inside_screen_section and lowered.split(":", 1)[0] in {
            "webcam",
            "microphone",
            "image",
        }:
            break
        if inside_screen_section and stripped:
            screen_lines.append(stripped)
    screen_text = " ".join(screen_lines)
    if len(screen_text) > maximum_characters:
        screen_text = screen_text[: max(0, maximum_characters - 1)].rstrip() + "…"
    return screen_text


def screen_observations_in_period(
    messages_by_thread: dict[str, list[Any]],
    period_start: datetime,
    period_end: datetime,
    *,
    maximum_description_characters: int,
) -> list[ScreenObservation]:
    """Collect every screen observation captured inside the period, oldest first."""
    observations: list[ScreenObservation] = []
    seen_observation_ids: set[str] = set()
    for thread_id, messages in messages_by_thread.items():
        for message in messages:
            if not is_ambient_observation(message):
                continue
            details = ambient_details(message) or {}
            sources = [str(source) for source in (details.get("sources") or [])]
            if SCREEN_SOURCE not in sources:
                continue
            captured_at = _parse_captured_at(details.get("captured_at"))
            if captured_at is None or not (period_start <= captured_at < period_end):
                continue
            observation_id = str(details.get("observation_id") or "")
            if observation_id and observation_id in seen_observation_ids:
                continue
            if observation_id:
                seen_observation_ids.add(observation_id)
            _header, body = split_observation_text(message_text(message))
            observations.append(
                ScreenObservation(
                    captured_at=captured_at,
                    observation_kind=str(
                        details.get("observation_kind") or UNLABELLED_KIND
                    )
                    .strip()
                    .lower()
                    or UNLABELLED_KIND,
                    summary=str(details.get("summary") or "").strip(),
                    screen_description=screen_section_of(
                        strip_instruction(body), maximum_description_characters
                    ),
                    thread_id=thread_id,
                )
            )
    observations.sort(key=lambda observation: observation.captured_at)
    return observations


def count_observed_seconds(
    observations: list[ScreenObservation],
    period_end: datetime,
    *,
    maximum_gap_seconds: float,
) -> float:
    """Set each observation's ``counted_seconds``; return the not-observed seconds.

    The not-observed seconds are the parts of the gaps between observations
    (and after the last observation, up to ``period_end``) beyond
    ``maximum_gap_seconds``. Time before the first observation of the period
    is not counted either way.
    """
    not_observed_seconds = 0.0
    for index, observation in enumerate(observations):
        next_moment = (
            observations[index + 1].captured_at
            if index + 1 < len(observations)
            else period_end
        )
        gap_seconds = max(0.0, (next_moment - observation.captured_at).total_seconds())
        observation.counted_seconds = min(gap_seconds, maximum_gap_seconds)
        not_observed_seconds += gap_seconds - observation.counted_seconds
    return not_observed_seconds


def _block_start(moment: datetime, owner_zone: Any, block_minutes: int) -> datetime:
    """Local start of the block holding ``moment``, aligned to the owner's local midnight."""
    local_moment = moment.astimezone(owner_zone)
    midnight = local_moment.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_since_midnight = local_moment.hour * 60 + local_moment.minute
    block_index = minutes_since_midnight // block_minutes
    return midnight + timedelta(minutes=block_index * block_minutes)


def choose_block_minutes(
    observations: list[ScreenObservation], owner_zone: Any, maximum_blocks: int
) -> int:
    """Return the shortest block length whose non-empty block count fits ``maximum_blocks``."""
    for block_minutes in BLOCK_MINUTES_CHOICES:
        block_starts = {
            _block_start(observation.captured_at, owner_zone, block_minutes)
            for observation in observations
        }
        if len(block_starts) <= max(1, maximum_blocks):
            return block_minutes
    return BLOCK_MINUTES_CHOICES[-1]


def _append_unique(values: list[str], value: str, keep: int) -> None:
    """Append ``value`` when non-empty, not already listed, and fewer than ``keep`` are listed."""
    if value and value not in values and len(values) < keep:
        values.append(value)


def build_screen_blocks(
    observations: list[ScreenObservation], owner_zone: Any, block_minutes: int
) -> list[ScreenBlock]:
    """Group counted observations into local-time blocks, oldest first."""
    blocks: dict[datetime, ScreenBlock] = {}
    for observation in observations:
        block_start = _block_start(observation.captured_at, owner_zone, block_minutes)
        block = blocks.get(block_start)
        if block is None:
            block = ScreenBlock(started_at=block_start, block_minutes=block_minutes)
            blocks[block_start] = block
        block.observed_seconds += observation.counted_seconds
        block.observation_count += 1
        block.seconds_by_activity[observation.observation_kind] = (
            block.seconds_by_activity.get(observation.observation_kind, 0.0)
            + observation.counted_seconds
        )
        _append_unique(block.summaries, observation.summary, SUMMARIES_PER_BLOCK)
        _append_unique(
            block.screen_descriptions,
            observation.screen_description,
            SCREEN_DESCRIPTIONS_PER_BLOCK,
        )
    return [blocks[block_start] for block_start in sorted(blocks)]


LABELS_PER_PERIOD = 25
"""Labels listed in the period-wide totals; the rest are summed under ``other_labels``."""

OTHER_LABELS = "other_labels"


def _seconds_sorted(
    seconds_by_label: dict[str, float], keep: int | None = None
) -> dict[str, int]:
    """Round seconds per label, order the labels longest first, and fold the tail past ``keep``."""
    ordered = sorted(seconds_by_label.items(), key=lambda item: item[1], reverse=True)
    kept = ordered if keep is None else ordered[:keep]
    rounded = {label: round(seconds) for label, seconds in kept}
    folded_seconds = sum(seconds for _label, seconds in ordered[len(kept) :])
    if folded_seconds:
        rounded[OTHER_LABELS] = round(folded_seconds)
    return rounded


def screen_timeline(
    messages_by_thread: dict[str, list[Any]],
    period_start: datetime,
    period_end: datetime,
    *,
    owner_zone: Any,
    maximum_gap_seconds: float,
    maximum_blocks: int,
    maximum_description_characters: int,
) -> dict[str, Any]:
    """Return the screen section of an activity review for one period."""
    observations = screen_observations_in_period(
        messages_by_thread,
        period_start,
        period_end,
        maximum_description_characters=maximum_description_characters,
    )
    if not observations:
        return {
            "observation_count": 0,
            "threads_read": len(messages_by_thread),
            "detail": "No watched screen observation was captured in this period.",
        }
    not_observed_seconds = count_observed_seconds(
        observations, period_end, maximum_gap_seconds=maximum_gap_seconds
    )
    block_minutes = choose_block_minutes(observations, owner_zone, maximum_blocks)
    blocks = build_screen_blocks(observations, owner_zone, block_minutes)
    seconds_by_activity: dict[str, float] = {}
    for observation in observations:
        seconds_by_activity[observation.observation_kind] = (
            seconds_by_activity.get(observation.observation_kind, 0.0)
            + observation.counted_seconds
        )
    observed_seconds = sum(observation.counted_seconds for observation in observations)
    return {
        "observation_count": len(observations),
        "threads_read": len(messages_by_thread),
        "threads_with_screen_observations": len(
            {observation.thread_id for observation in observations}
        ),
        "first_observation_local": observations[0]
        .captured_at.astimezone(owner_zone)
        .isoformat(),
        "last_observation_local": observations[-1]
        .captured_at.astimezone(owner_zone)
        .isoformat(),
        "observed_seconds": round(observed_seconds),
        "not_observed_seconds": round(not_observed_seconds),
        "maximum_gap_seconds": round(maximum_gap_seconds),
        "duration_method": DURATION_METHOD,
        "seconds_by_activity": _seconds_sorted(seconds_by_activity, LABELS_PER_PERIOD),
        "distinct_activity_labels": len(seconds_by_activity),
        "block_minutes": block_minutes,
        "blocks": [
            {
                "started_local": block.started_at.strftime("%Y-%m-%d %H:%M"),
                "observed_seconds": round(block.observed_seconds),
                "observation_count": block.observation_count,
                "seconds_by_activity": _seconds_sorted(block.seconds_by_activity),
                "summaries": block.summaries,
                "on_screen": block.screen_descriptions,
            }
            for block in blocks
        ],
    }
