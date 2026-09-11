"""Nodes of the psycho-analysis graph.

``select_target_documents`` decides what the dimensions read, every dimension node
reads it concurrently, ``consolidate_psychological_profile`` folds the findings
into the avatar's accumulated profile, and ``seed_current_emotional_state`` sets
the emotional baseline the live state decays back toward.

Everything here is best effort in the same sense ``observe_user`` is: one
dimension that fails logs and leaves its section empty rather than failing the
upload the person is waiting on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document

from src.anubis.utils.psycho.current_emotion import (
    PLUTCHIK_PRIMARY_EMOTIONS,
    build_current_emotion_record,
    read_current_emotion_record,
    write_current_emotion_record,
)
from src.anubis.utils.psycho.profile import (
    merge_findings_into_profile,
    read_profile_record,
    write_profile_record,
)
from src.subgraphs.psycho_analysis_graph.utils.dimensions import (
    EMOTIONAL_BASELINE_DIMENSION,
    PsychologicalDimension,
)

logger = logging.getLogger(__name__)

# Documents shorter than this say too little about a person to be worth a dozen
# structured-output calls.
MIN_DOCUMENT_CHARACTERS = 200
# How many narrative findings one dimension may contribute from a single upload.
# The profile keeps accumulating across uploads; this only stops one long
# transcript from crowding every other source out of the section.
MAX_STATEMENTS_PER_DIMENSION_PER_UPLOAD = 6


def _situational_context_from_document(document: Document) -> str | None:
    """The scene summary and preceding turn the media pipeline already attached."""
    metadata = document.metadata or {}
    parts = [
        str(metadata.get("scene_summary") or "").strip(),
        str(metadata.get("user_context") or "").strip(),
    ]
    context = "\n".join(part for part in parts if part)
    return context or None


def _document_is_about_the_target(document: Document) -> bool:
    """Whether this document carries the TARGET's own words or actions.

    The psychological reading describes one person, so a document that does not
    identify its target — or that was classified as somebody else's reference
    material rather than the target's own speech — must not reach the dimensions.
    The media pipeline has already done the attribution work (see
    ``target_attribution.py`` and ``text_dialogue_segmentation.py``); this only
    reads its conclusions.
    """
    metadata = document.metadata or {}
    if not str(metadata.get("target_name") or "").strip():
        return False
    if metadata.get("classified_situation") == "proprietary_content":
        return False
    # ``is_target`` is written on per-speaker quote documents by
    # ``_build_target_quote_documents_from_dialogue`` and
    # ``build_all_speakers_quote_documents``; when the key is absent the document
    # is not per-speaker and the target check above is what governs. When it is
    # present and false, the document holds SOMEONE ELSE's words and must never
    # reach a dimension: another speaker's values are not the target's values.
    is_target = metadata.get("is_target")
    if is_target is not None and not is_target:
        return False
    return True


def _select_dialogue_documents(state: dict, maximum: int) -> list[Document]:
    """The whole conversation as one document, both speakers intact.

    A trigger is a PAIR — something happens, the target reacts — so it is only
    visible in a stretch of conversation long enough to contain both halves. The
    analysis queue holds the target's answers cut into separate chunks of a few
    hundred characters each, and an isolated answer almost never shows a pair:
    measured on one interview, the chunked queue yielded a single trigger while
    the same interview as one document yielded eight.

    The adapter queue holds the role-converted conversation as ONE document with
    every speaker's turns in order, which fixes both halves of the problem — the
    stimulus is present because the other speaker is, and the arc is present
    because it was never chunked. Trait dimensions keep reading the target-only
    chunks, where the isolation is a feature: it stops another speaker's values
    being attributed to the target.
    """
    documents = [
        document
        for document in (state.get("dialogue_documents") or [])
        if (document.page_content or "").strip()
        and str((document.metadata or {}).get("target_name") or "").strip()
    ]
    documents.sort(key=lambda document: len(document.page_content or ""), reverse=True)
    return documents[: max(1, maximum)]


def select_target_documents(state: dict) -> dict:
    """Keep the documents richest in the target's own words, bounded by the cap."""
    documents = [
        document
        for document in (state.get("documents") or [])
        if _document_is_about_the_target(document)
        and len((document.page_content or "").strip()) >= MIN_DOCUMENT_CHARACTERS
    ]
    if not documents:
        logger.info(
            "psycho analysis: no target-scoped documents in this upload; skipping"
        )
        return {"selected_documents": [], "selected_dialogue_documents": []}
    # A trait is a stable thing, so reading every chunk of a long transcript pays
    # many calls to reach the same conclusion. Prefer the longest documents, which
    # carry the most of the target per call.
    maximum = int(state.get("max_documents") or 24)
    documents.sort(key=lambda document: len(document.page_content or ""), reverse=True)
    selected = documents[: max(1, maximum)]
    dialogue = _select_dialogue_documents(state, maximum)
    logger.info(
        "psycho analysis: %d of %d documents selected for analysis, "
        "%d full-dialogue documents for the trigger dimensions",
        len(selected),
        len(state.get("documents") or []),
        len(dialogue),
    )
    return {"selected_documents": selected, "selected_dialogue_documents": dialogue}


def _merge_dimension_findings(
    dimension_name: str, kind: str, findings: list[dict]
) -> dict:
    """Combine one dimension's per-document findings into a single finding."""
    if not findings:
        return {}
    if kind == "graded":
        traits: dict[str, Any] = {}
        summaries: list[str] = []
        for finding in findings:
            for trait_name, reading in (finding.get("traits") or {}).items():
                previous = traits.get(trait_name)
                # Within one upload, keep the reading the model was most confident
                # about; across uploads the profile does the weighted averaging.
                if previous is None or float(reading.get("confidence") or 0.0) > float(
                    previous.get("confidence") or 0.0
                ):
                    traits[trait_name] = reading
            summary = (finding.get("summary") or "").strip()
            if summary:
                summaries.append(summary)
        return {
            "dimension": dimension_name,
            "kind": "graded",
            "traits": traits,
            "summary": summaries[0] if summaries else "",
        }
    statements: list[dict] = []
    seen: set[str] = set()
    for finding in findings:
        for entry in finding.get("statements") or []:
            statement = (entry.get("statement") or "").strip()
            if not statement:
                continue
            # Several chunks of one transcript describe the same habit in slightly
            # different words. Keep the first wording of each and drop the rest, so
            # one dimension cannot fill the profile with paraphrases of one finding.
            key = " ".join(statement.lower().split())[:120]
            if key in seen:
                continue
            seen.add(key)
            statements.append(entry)
    return {
        "dimension": dimension_name,
        "kind": "narrative",
        "statements": statements[:MAX_STATEMENTS_PER_DIMENSION_PER_UPLOAD],
    }


def build_dimension_node(dimension: PsychologicalDimension):
    """Build the graph node that reads one dimension across the selected documents.

    Every dimension is its own node so the graph fans them out concurrently and each
    one is separately visible while an upload runs.
    """

    async def dimension_node(state: dict, runtime: Any = None) -> dict:
        if dimension.reads_full_dialogue:
            # Fall back to the target-only documents when an upload carried no
            # dialogue at all (a monologue, a series of tweets): the dimension
            # then recovers only what the target self-reports, which is still
            # better than reading nothing.
            documents = list(state.get("selected_dialogue_documents") or []) or list(
                state.get("selected_documents") or []
            )
        else:
            documents = list(state.get("selected_documents") or [])
        if not documents:
            return {}
        context = getattr(runtime, "context", None)
        concurrency = int(
            getattr(context, "psychological_analysis_concurrency", 6) or 6
        )
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def analyze_one(document: Document):
            async with semaphore:
                return await dimension.analyze(
                    document,
                    target_name=(document.metadata or {}).get("target_name"),
                    source_metadata=_source_metadata(document),
                    situational_context=_situational_context_from_document(document),
                )

        results = await asyncio.gather(
            *(analyze_one(document) for document in documents),
            return_exceptions=True,
        )
        produced_documents: list[Document] = []
        findings: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "psycho analysis: dimension %r failed on one document: %s; continuing",
                    dimension.name,
                    result,
                )
                continue
            dimension_documents, finding = result
            produced_documents.extend(dimension_documents)
            if finding:
                findings.append(finding)
        merged = _merge_dimension_findings(dimension.name, dimension.kind, findings)
        logger.info(
            "psycho analysis: dimension %r produced %d findings",
            dimension.name,
            len(produced_documents),
        )
        return {
            "psychological_documents": produced_documents,
            "dimension_findings": [merged] if merged else [],
        }

    dimension_node.__name__ = f"analyze_{dimension.name}"
    return dimension_node


# Routing metadata keys that describe the analysis QUEUE rather than the source, so
# they must not be copied onto an analysis output (it would be re-queued).
_QUEUE_METADATA_KEYS = frozenset(
    {
        "analysis_scaffolds",
        "analysis_job_kind",
        "analysis_acceptable",
        "vectorstore_acceptable",
        "adapter_acceptable",
        "namespace",
    }
)


def _source_metadata(document: Document) -> dict[str, Any]:
    """Carry the source Document's identifying metadata onto the findings."""
    return {
        key: value
        for key, value in (document.metadata or {}).items()
        if key not in _QUEUE_METADATA_KEYS
    }


async def consolidate_psychological_profile(state: dict, runtime: Any = None) -> dict:
    """Fold this upload's findings into the avatar's accumulated profile.

    Accumulates rather than replaces: a second upload reinforces or moves what the
    first one learned, and never erases it. See
    :func:`src.anubis.utils.psycho.profile.merge_findings_into_profile`.
    """
    findings = [
        finding for finding in (state.get("dimension_findings") or []) if finding
    ]
    if not findings:
        return {}
    store = getattr(runtime, "store", None) or state.get("store")
    creator_id = state.get("creator_id")
    assistant_id = state.get("assistant_id")
    if store is None or not creator_id or not assistant_id:
        logger.warning(
            "psycho analysis: no store or avatar identity; the profile was not saved"
        )
        return {"psychological_profile": None}

    existing = await read_profile_record(store, creator_id, assistant_id)
    context = getattr(runtime, "context", None)
    profile = merge_findings_into_profile(existing, findings)
    max_characters = int(
        getattr(context, "psychological_profile_max_characters", 6000) or 6000
    )
    from src.anubis.utils.psycho.profile import render_profile

    profile["value"] = render_profile(profile, max_characters)
    await write_profile_record(store, creator_id, assistant_id, profile)
    logger.info(
        "psycho analysis: consolidated %d dimensions into the profile of avatar %s",
        len(findings),
        assistant_id,
    )
    return {"psychological_profile": profile}


async def seed_current_emotional_state(state: dict, runtime: Any = None) -> dict:
    """Set the emotional baseline the avatar's live state decays back toward.

    The baseline is the target's temperament, so it comes from the media rather
    than from a conversation. The live state is seeded to the baseline the first
    time; afterwards only the baseline is refreshed, because overwriting the live
    state would erase whatever the current conversation had built up.
    """
    profile = state.get("psychological_profile") or {}
    baseline_traits = (
        (profile.get("dimensions") or {}).get(EMOTIONAL_BASELINE_DIMENSION) or {}
    ).get("traits") or {}
    if not baseline_traits:
        return {}
    baseline_wheel = {
        emotion: float((baseline_traits.get(emotion) or {}).get("score") or 0.0)
        for emotion in PLUTCHIK_PRIMARY_EMOTIONS
    }
    store = getattr(runtime, "store", None) or state.get("store")
    creator_id = state.get("creator_id")
    assistant_id = state.get("assistant_id")
    if store is None or not creator_id or not assistant_id:
        return {}
    existing = await read_current_emotion_record(store, creator_id, assistant_id)
    live_wheel = (existing or {}).get("wheel") or baseline_wheel
    record = build_current_emotion_record(live_wheel, baseline_wheel)
    await write_current_emotion_record(store, creator_id, assistant_id, record)
    logger.info(
        "psycho analysis: emotional baseline seeded for avatar %s", assistant_id
    )
    return {"current_emotion": record}


__all__ = [
    "MIN_DOCUMENT_CHARACTERS",
    "build_dimension_node",
    "consolidate_psychological_profile",
    "seed_current_emotional_state",
    "select_target_documents",
]
