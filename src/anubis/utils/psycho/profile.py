"""The consolidated psychological profile: accumulate, render, read.

Raw analysis findings already reach the avatar through the ``analysis`` namespace
and the ANALYZED TRAITS section, retrieved by similarity against whatever the
conversation happens to be about. That works for a fact and fails for a trait: a
person's love languages, attachment style and values bear on EVERY reply, not only
on replies whose topic resembles the finding. So alongside the raw findings this
module keeps one consolidated profile per avatar, fetched by key on every turn and
rendered into its own prompt section.

The profile ACCUMULATES. A second upload must never erase what the first one
learned, so:

* a graded trait keeps a confidence-weighted running mean of its score, and the
  count of how many readings went into it — two uploads that agree reinforce each
  other, and a disagreement moves the score partway rather than replacing it;
* a narrative statement is appended if it is new and dropped if the profile already
  holds a near-identical one.

Nothing here calls a model. Consolidation is arithmetic over findings the analyzer
nodes already produced, which is what makes it cheap enough to run on every upload.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from src.anubis.utils.psycho.namespaces import (
    CURRENT_RECORD_KEY,
    psychological_profile_namespace,
)

logger = logging.getLogger(__name__)

# How many narrative statements one dimension keeps. The profile is read into the
# system prompt on every turn, so it is bounded by construction rather than by
# hoping the analyzers stay terse.
MAX_STATEMENTS_PER_DIMENSION = 12
# Traits scoring below this are not rendered: a profile that lists every trait the
# target does NOT have buries the ones the target does.
MIN_RENDERED_TRAIT_SCORE = 0.25
# How many traits of one graded dimension are rendered, strongest first.
MAX_RENDERED_TRAITS_PER_DIMENSION = 5

DIMENSION_KIND_GRADED = "graded"
DIMENSION_KIND_NARRATIVE = "narrative"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _normalized(text: str) -> str:
    """Lowercased, punctuation-free form used to tell two statements apart."""
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def empty_profile() -> dict[str, Any]:
    """The profile record of an avatar nothing has been read about yet."""
    return {"value": "", "dimensions": {}, "updated_at": _now(), "upload_count": 0}


def _merge_graded_dimension(
    existing: Mapping[str, Any] | None, finding: Mapping[str, Any]
) -> dict[str, Any]:
    """Fold one graded reading into the running, confidence-weighted profile."""
    merged_traits: dict[str, Any] = dict((existing or {}).get("traits") or {})
    for trait_name, reading in (finding.get("traits") or {}).items():
        try:
            score = float(reading.get("score", 0.0))
            confidence = float(reading.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        score = min(max(score, 0.0), 1.0)
        confidence = min(max(confidence, 0.0), 1.0)
        # A reading with no confidence still counts as an observation, but it must
        # not be able to drag the running mean around; give it a small floor.
        weight = max(confidence, 0.05)
        previous = merged_traits.get(trait_name) or {}
        previous_weight = float(previous.get("weight_total") or 0.0)
        previous_score = float(previous.get("score") or 0.0)
        total_weight = previous_weight + weight
        merged_traits[trait_name] = {
            "score": ((previous_score * previous_weight) + (score * weight))
            / total_weight,
            "confidence": max(float(previous.get("confidence") or 0.0), confidence),
            "weight_total": total_weight,
            "observations": int(previous.get("observations") or 0) + 1,
            # The most recent reading's words win: they were drawn from the newest
            # material, and the score already carries the whole history.
            "statement": reading.get("statement") or previous.get("statement") or "",
            "evidence": reading.get("evidence") or previous.get("evidence") or "",
        }
    summaries = list((existing or {}).get("summaries") or [])
    summary = (finding.get("summary") or "").strip()
    if summary and _normalized(summary) not in {_normalized(s) for s in summaries}:
        summaries.append(summary)
    return {
        "kind": DIMENSION_KIND_GRADED,
        "traits": merged_traits,
        "summaries": summaries[-3:],
        "updated_at": _now(),
    }


def _merge_narrative_dimension(
    existing: Mapping[str, Any] | None, finding: Mapping[str, Any]
) -> dict[str, Any]:
    """Append the narrative statements this reading found that are genuinely new."""
    statements = list((existing or {}).get("statements") or [])
    seen = {_normalized(entry.get("statement", "")) for entry in statements}
    for entry in finding.get("statements") or []:
        statement = (entry.get("statement") or "").strip()
        if not statement:
            continue
        key = _normalized(statement)
        if not key or key in seen:
            continue
        seen.add(key)
        statements.append(
            {"statement": statement, "evidence": (entry.get("evidence") or "").strip()}
        )
    return {
        "kind": DIMENSION_KIND_NARRATIVE,
        "statements": statements[-MAX_STATEMENTS_PER_DIMENSION:],
        "updated_at": _now(),
    }


def merge_findings_into_profile(
    profile: Mapping[str, Any] | None, findings: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fold one upload's dimension findings into the accumulated profile.

    ``findings`` are the per-dimension results produced by the psycho-analysis
    graph: ``{"dimension": str, "kind": "graded"|"narrative", ...}``.
    """
    merged = dict(profile or empty_profile())
    dimensions: dict[str, Any] = dict(merged.get("dimensions") or {})
    for finding in findings:
        dimension = (finding or {}).get("dimension")
        if not dimension:
            continue
        existing = dimensions.get(dimension)
        if finding.get("kind") == DIMENSION_KIND_GRADED:
            dimensions[dimension] = _merge_graded_dimension(existing, finding)
        else:
            dimensions[dimension] = _merge_narrative_dimension(existing, finding)
    merged["dimensions"] = dimensions
    merged["upload_count"] = int(merged.get("upload_count") or 0) + 1
    merged["updated_at"] = _now()
    merged["value"] = render_profile(merged)
    return merged


def _title(dimension: str) -> str:
    return dimension.replace("_", " ").upper()


def render_profile(profile: Mapping[str, Any], max_characters: int = 6000) -> str:
    """Render the profile as the prose the avatar's system prompt carries.

    Written in the first person, because everything else in the ROLE block is: the
    avatar reads this as a description of itself, not as a report about somebody.
    """
    dimensions = (profile or {}).get("dimensions") or {}
    if not dimensions:
        return ""
    sections: list[str] = []
    for dimension in sorted(dimensions):
        entry = dimensions[dimension] or {}
        lines: list[str] = []
        if entry.get("kind") == DIMENSION_KIND_GRADED:
            traits = entry.get("traits") or {}
            ranked = sorted(
                traits.items(),
                key=lambda item: float((item[1] or {}).get("score") or 0.0),
                reverse=True,
            )
            for trait_name, reading in ranked[:MAX_RENDERED_TRAITS_PER_DIMENSION]:
                score = float((reading or {}).get("score") or 0.0)
                if score < MIN_RENDERED_TRAIT_SCORE:
                    continue
                statement = (reading or {}).get("statement") or trait_name.replace(
                    "_", " "
                )
                lines.append(f"- {statement} ({trait_name}: {score:.2f})")
            for summary in (entry.get("summaries") or [])[-1:]:
                lines.append(f"- {summary}")
        else:
            for record in entry.get("statements") or []:
                statement = (record or {}).get("statement")
                if statement:
                    lines.append(f"- {statement}")
        if lines:
            sections.append(f"{_title(dimension)}\n" + "\n".join(lines))
    if not sections:
        return ""
    rendered = "\n\n".join(sections)
    if max_characters and len(rendered) > max_characters:
        # Truncate on a section boundary so the profile never ends mid-sentence.
        kept: list[str] = []
        used = 0
        for section in sections:
            if used + len(section) + 2 > max_characters:
                break
            kept.append(section)
            used += len(section) + 2
        rendered = "\n\n".join(kept) or rendered[:max_characters]
    return rendered


async def read_profile_record(
    store: Any, creator_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Fetch the profile record through the store cache. Best effort."""
    if store is None or not creator_id or not assistant_id:
        return None
    try:
        from src.anubis.utils.store_cache import aget_through_cache

        item = await aget_through_cache(
            store,
            psychological_profile_namespace(creator_id, assistant_id),
            CURRENT_RECORD_KEY,
        )
    except Exception as read_error:  # noqa: BLE001 - never cost a turn its prompt
        logger.warning("Could not read the psychological profile: %s", read_error)
        return None
    value = getattr(item, "value", None)
    return value if isinstance(value, dict) else None


async def read_profile_text(
    store: Any, creator_id: str, assistant_id: str, max_characters: int = 6000
) -> str:
    """The rendered profile for the system prompt, or an empty string."""
    record = await read_profile_record(store, creator_id, assistant_id)
    if not record:
        return ""
    rendered = record.get("value") or render_profile(record, max_characters)
    if max_characters and len(rendered) > max_characters:
        rendered = render_profile(record, max_characters)
    return rendered


async def write_profile_record(
    store: Any, creator_id: str, assistant_id: str, profile: Mapping[str, Any]
) -> bool:
    """Persist the profile and drop the cached copy so the next turn sees it."""
    if store is None or not creator_id or not assistant_id:
        return False
    namespace = psychological_profile_namespace(creator_id, assistant_id)
    try:
        await store.aput(namespace, key=CURRENT_RECORD_KEY, value=dict(profile))
    except Exception as write_error:  # noqa: BLE001 - an upload must still finish
        logger.error("Could not write the psychological profile: %s", write_error)
        return False
    try:
        from src.anubis.utils.store_cache import invalidate_store_cache_entry

        invalidate_store_cache_entry(namespace, CURRENT_RECORD_KEY)
    except Exception as cache_error:  # noqa: BLE001 - a stale read expires on its own
        logger.warning(
            "Could not invalidate the cached psychological profile: %s", cache_error
        )
    return True


__all__ = [
    "DIMENSION_KIND_GRADED",
    "DIMENSION_KIND_NARRATIVE",
    "MAX_STATEMENTS_PER_DIMENSION",
    "empty_profile",
    "merge_findings_into_profile",
    "read_profile_record",
    "read_profile_text",
    "render_profile",
    "write_profile_record",
]
