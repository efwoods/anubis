"""Stage two of signature key-phrase discovery: the precision judge.

:mod:`src.anubis.utils.dataset.key_phrase_candidates` produces a shortlist of
everything the target says far more often than a reference speaker would. That
shortlist is high recall and mixed precision, and the mixing is not a tuning
problem — it is a limit of the statistic. Measured on a real corpus, the stance
marker "concerning" and the subject word "civilization" landed within a
hundredth of each other on every number available, and the genuine idiolect
"gets me every time" sat beside the advertising fragment "sign up via web". No
threshold separates those pairs, because the thing that separates them is
meaning.

So this module asks a model. One batched structured-output call classifies each
candidate as the speaker's style, the speaker's subject matter, ordinary
informal English, or reproduced boilerplate, and only the first survives.

Three properties make the call safe to run inside the upload calibration:

* **It fails open.** Any batch that raises, times out, or comes back empty
  yields its candidates as low-confidence style, ranked below every
  model-confirmed phrase. A classifier outage degrades the list; it never
  empties it.
* **It is cached per phrase.** A phrase judged once is never judged again.
  ``init_model`` runs at a low but non-zero temperature, so without the cache
  the phrase set would drift between calibrations, and every drift forces the
  caller to recompute the stylometric feature row of every document in the
  corpus. The cache is a correctness mechanism, not an optimisation.
* **It is bounded.** Batch size, concurrency and a total timeout are all
  configured, and the timeout lives here rather than being left to the caller's
  much larger calibration budget.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel

try:  # Literal is in typing on Python 3.11; keep the import defensive.
    from typing import Literal
except ImportError:  # pragma: no cover - Python < 3.8 not supported here
    from typing_extensions import Literal  # type: ignore

from src.anubis.utils.dataset.key_phrase_candidates import (
    KeyPhraseCandidate,
    KeyPhraseDiscoveryConfiguration,
    discover_key_phrase_candidates,
)

logger = logging.getLogger(__name__)

# The classification that keeps a phrase. Every other classification discards.
SIGNATURE_STYLE_CLASSIFICATION = "signature_style"

# Confidences that count as a model confirmation. A "low" style judgement is
# treated as a rejection: the prompt tells the model to choose "low" whenever it
# is unsure, and an unsure inclusion is the expensive kind of mistake here.
CONFIRMING_CONFIDENCES = frozenset({"high", "medium"})

# Marker written onto a judgement the model never actually produced, because a
# batch failed. Such records are kept in the RESULT (so an outage degrades the
# list rather than emptying it) but deliberately never written to the CACHE, so
# the next calibration retries them.
STAGE_ONE_FALLBACK_SOURCE = "stage_one_fallback"
MODEL_JUDGEMENT_SOURCE = "model"

# How many times the judge re-asks about candidates a previous round left out.
# Structured output does not guarantee one record per input, and a phrase the
# model simply skipped is indistinguishable from one it refused — so without a
# retry those phrases fall open and never reach the profile.
JUDGEMENT_ROUND_LIMIT = 3


class SignaturePhraseJudgement(BaseModel):
    """One candidate phrase, classified as style, subject, generic, or copy."""

    phrase: str
    classification: Literal[
        "signature_style", "topic_or_content", "generic_english", "boilerplate"
    ]
    confidence: Literal["high", "medium", "low"]
    reason: str


class SignaturePhraseJudgementResponse(BaseModel):
    """One judgement per candidate in the batch."""

    judgements: List[SignaturePhraseJudgement]


@dataclass
class KeyPhraseProfile:
    """The finished phrase set plus everything needed to explain and reuse it.

    ``phrases`` is the rank-ordered bare list the rest of the system consumes.
    ``entries`` and ``judgement_cache`` ride in the sibling detail record: the
    entries are the observability that catches a judge which has started
    admitting subject matter, and the cache is what keeps the set stable across
    calibrations.
    """

    phrases: List[str] = field(default_factory=list)
    entries: List[Dict[str, Any]] = field(default_factory=list)
    judgement_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    classification_histogram: Dict[str, int] = field(default_factory=dict)


def _fallback_judgement(reason: str) -> Dict[str, Any]:
    """The record used when the model could not be consulted for a phrase."""
    return {
        "classification": SIGNATURE_STYLE_CLASSIFICATION,
        "confidence": "low",
        "reason": reason,
        "judgement_source": STAGE_ONE_FALLBACK_SOURCE,
    }



def normalize_phrase_key(phrase: str) -> str:
    """Whitespace- and case-insensitive key for matching a returned phrase.

    The judge is asked to copy each phrase verbatim and mostly does, but it
    routinely appends a trailing space — it returns ``"want you to "`` for the
    candidate ``"want you to"``. An exact-equality guard therefore threw away
    every verdict in a batch and reported the whole batch as unjudged, which is
    what made a real avatar fall back to its raw statistical shortlist while the
    model had in fact classified all forty-five of its candidates correctly.

    Normalising collapses that difference without weakening the anti-hallucination
    guarantee: a returned phrase is still only accepted when it maps onto one the
    batch actually asked about, and the CANDIDATE's spelling is what gets stored.
    """
    return " ".join((phrase or "").split()).lower()


def render_candidate_for_judge(candidate: KeyPhraseCandidate) -> str:
    """One candidate as the judge sees it: the phrase, its numbers, its uses.

    The usage lines are the load-bearing part. The same word can be a verbal
    habit for one speaker and a subject for another, and only seeing the phrase
    at work distinguishes them — which is exactly the call the statistics could
    not make.
    """
    header = (
        f"{candidate.phrase}\n"
        f"    [words={candidate.ngram_size}, "
        f"occurrences={candidate.occurrence_count}, "
        f"places={candidate.dispersion_unit_count}, "
        f"log_ratio={candidate.log_ratio_over_reference:.2f}]"
    )
    usage_lines = [f"    … {line} …" for line in candidate.concordance_lines]
    return "\n".join([header, *usage_lines])


def _build_judge_human_message(
    candidates: Sequence[KeyPhraseCandidate], speaker_name: Optional[str]
) -> str:
    """The human turn for one batch."""
    return "\n\n".join(
        [
            f"Speaker: {speaker_name or 'unknown'}",
            (
                "Classify every candidate below exactly once. Each candidate is "
                "shown with its statistics and up to three lines of real use, "
                "with the candidate marked in square brackets."
            ),
            "\n\n".join(
                render_candidate_for_judge(candidate) for candidate in candidates
            ),
        ]
    )


async def _judge_one_batch(
    candidates: Sequence[KeyPhraseCandidate], *, speaker_name: Optional[str]
) -> Dict[str, Dict[str, Any]]:
    """Classify one batch of candidates. Raises on any model failure."""
    # Lazy import so a cold start never pulls the model SDK for this module.
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.signature_phrase_judge_prompt import (
        SIGNATURE_PHRASE_JUDGE_SYSTEM_PROMPT,
    )

    model = init_model(
        model_without_tools=False,
        response_format=SignaturePhraseJudgementResponse,
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=SIGNATURE_PHRASE_JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=_build_judge_human_message(candidates, speaker_name)
            ),
        ]
    )

    judgements = getattr(response, "judgements", None) or []
    requested_phrase_by_key = {
        normalize_phrase_key(candidate.phrase): candidate.phrase
        for candidate in candidates
    }
    judged: Dict[str, Dict[str, Any]] = {}
    for judgement in judgements:
        returned_phrase = getattr(judgement, "phrase", None)
        # Never trust a phrase the batch did not ask about; a hallucinated
        # phrase would otherwise be written straight into the voice profile.
        # Matching is whitespace- and case-insensitive, and the key stored is
        # always the CANDIDATE's own spelling, never the model's echo of it.
        phrase = requested_phrase_by_key.get(normalize_phrase_key(returned_phrase))
        if phrase is None:
            continue
        judged[phrase] = {
            "classification": getattr(
                judgement, "classification", "generic_english"
            ),
            "confidence": getattr(judgement, "confidence", "low"),
            "reason": getattr(judgement, "reason", "") or "",
            "judgement_source": MODEL_JUDGEMENT_SOURCE,
        }
    return judged


async def judge_key_phrase_candidates(
    candidates: Sequence[KeyPhraseCandidate],
    *,
    speaker_name: Optional[str],
    previously_judged: Optional[Mapping[str, Mapping[str, Any]]] = None,
    batch_size: int = 20,
    concurrency: int = 4,
    timeout_seconds: float = 120.0,
) -> Dict[str, Dict[str, Any]]:
    """Classify every candidate, consulting the model only for unjudged ones.

    Returns a judgement for EVERY candidate. A batch that fails contributes
    fail-open records rather than nothing, and those records are marked so the
    caller can rank them below model-confirmed phrases and keep them out of the
    cache.
    """
    previously_judged = previously_judged or {}
    judged: Dict[str, Dict[str, Any]] = {
        candidate.phrase: dict(previously_judged[candidate.phrase])
        for candidate in candidates
        if candidate.phrase in previously_judged
    }
    unjudged = [
        candidate for candidate in candidates if candidate.phrase not in judged
    ]
    if not unjudged:
        return judged

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_batch(
        batch: Sequence[KeyPhraseCandidate],
    ) -> Dict[str, Dict[str, Any]]:
        async with semaphore:
            return await _judge_one_batch(batch, speaker_name=speaker_name)

    async def judge_all() -> Dict[str, Dict[str, Any]]:
        """Judge every candidate, re-asking about whatever comes back missing.

        A structured-output call does NOT reliably return one record per input.
        Measured against a real shortlist, a batch of forty came back covering
        only part of itself — the model simply stopped early — and because a
        missing record is indistinguishable from a refusal, those phrases were
        silently falling open and never reaching the profile. So each round
        re-asks about only what is still unjudged, with the batch halved, which
        both shortens the answer the model has to produce and gives the
        remainder a fresh, smaller context.
        """
        collected: Dict[str, Dict[str, Any]] = {}
        remaining = list(unjudged)
        current_batch_size = max(1, batch_size)

        for round_number in range(JUDGEMENT_ROUND_LIMIT):
            if not remaining:
                break
            batches = [
                remaining[start : start + current_batch_size]
                for start in range(0, len(remaining), current_batch_size)
            ]
            results = await asyncio.gather(
                *(run_batch(batch) for batch in batches), return_exceptions=True
            )
            for batch, result in zip(batches, results):
                if isinstance(result, BaseException) or not isinstance(result, dict):
                    logger.warning(
                        "Signature phrase judging failed for one batch of %d (%s)",
                        len(batch),
                        result,
                    )
                    continue
                collected.update(result)

            still_missing = [
                candidate
                for candidate in remaining
                if candidate.phrase not in collected
            ]
            if len(still_missing) == len(remaining):
                # A round that judged nothing at all will not do better next
                # time; stop rather than spend more calls on the same failure.
                break
            remaining = still_missing
            current_batch_size = max(1, current_batch_size // 2)

        if remaining:
            logger.warning(
                "Signature phrase judging returned no verdict for %d of %d "
                "candidates after %d rounds; those fall back to the "
                "statistical shortlist",
                len(remaining),
                len(unjudged),
                JUDGEMENT_ROUND_LIMIT,
            )
        return collected

    try:
        collected = await asyncio.wait_for(judge_all(), timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - includes asyncio.TimeoutError
        # The whole fan-out failed or ran out of time. Every unjudged candidate
        # falls open; none of them is cached, so the next calibration retries.
        logger.warning(
            "Signature phrase judging failed for the whole batch set (%s); "
            "falling back to the statistical shortlist for %d candidates",
            exc,
            len(unjudged),
        )
        collected = {}

    for candidate in unjudged:
        judged[candidate.phrase] = collected.get(
            candidate.phrase,
            _fallback_judgement("no judgement returned for this phrase"),
        )
    return judged


def _resolve_judgement_settings() -> Dict[str, Any]:
    """Read the judge's tunables from the runtime context.

    Read here rather than at import so a test or a script can flip the gate
    without reimporting the module, matching how the rest of the codebase reads
    ``GlobalContext``.
    """
    from src.anubis.utils.context import GlobalContext

    context = GlobalContext()

    def _integer(field_name: str, fallback: int) -> int:
        raw = getattr(context, field_name, None)
        try:
            return int(raw) if raw is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def _number(field_name: str, fallback: float) -> float:
        raw = getattr(context, field_name, None)
        try:
            return float(raw) if raw is not None else fallback
        except (TypeError, ValueError):
            return fallback

    enabled_raw = getattr(context, "key_phrase_judgement_enabled", None)
    enabled = str(enabled_raw if enabled_raw is not None else "TRUE").upper() == "TRUE"

    return {
        "enabled": enabled,
        "batch_size": _integer("key_phrase_judgement_batch_size", 20),
        "concurrency": _integer("key_phrase_judgement_concurrency", 4),
        "timeout_seconds": _number("key_phrase_judgement_timeout_seconds", 120.0),
        "shortlist_size": _integer("key_phrase_candidate_shortlist_size", 200),
        "maximum_phrases": _integer("key_phrase_profile_maximum_phrases", 25),
        "incumbency_bonus": _number("key_phrase_incumbency_score_bonus", 0.25),
    }


def _previous_phrases_from_detail(
    previous_profile_detail: Optional[Mapping[str, Any]],
) -> List[str]:
    """Phrases stored for this speaker by an earlier calibration."""
    if not previous_profile_detail:
        return []
    phrases = previous_profile_detail.get("phrases")
    if isinstance(phrases, list):
        return [phrase for phrase in phrases if isinstance(phrase, str)]
    return []


def _judgement_cache_from_detail(
    previous_profile_detail: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """The per-phrase judgements an earlier calibration already paid for."""
    if not previous_profile_detail:
        return {}
    cache = previous_profile_detail.get("judgement_cache")
    if not isinstance(cache, dict):
        return {}
    return {
        phrase: dict(record)
        for phrase, record in cache.items()
        if isinstance(phrase, str) and isinstance(record, dict)
    }


async def build_signature_key_phrase_profile(
    documents: Sequence[str],
    *,
    speaker_name: Optional[str] = None,
    previous_profile_detail: Optional[Mapping[str, Any]] = None,
    configuration: Optional[KeyPhraseDiscoveryConfiguration] = None,
) -> KeyPhraseProfile:
    """Discover, judge, rank and cap this speaker's signature phrases.

    The whole pipeline in one call, so the caller never has to know that there
    are two stages or which of them produced a given phrase.

    Previously-stored phrases are NOT carried forward automatically. They are
    re-measured against the current corpus alongside every new candidate and
    keep their place only if the evidence still supports them, which is what
    stops the stored set from growing by roughly forty phrases on every upload
    the way a blind union did. Incumbents get two advantages and no more: they
    survive pool truncation, and they carry a small score bonus that stops the
    list oscillating when two phrases are separated by a hair.
    """
    settings = _resolve_judgement_settings()
    previous_phrases = _previous_phrases_from_detail(previous_profile_detail)
    judgement_cache = _judgement_cache_from_detail(previous_profile_detail)

    configuration = configuration or KeyPhraseDiscoveryConfiguration()
    if configuration.shortlist_size != settings["shortlist_size"]:
        configuration = replace(
            configuration, shortlist_size=settings["shortlist_size"]
        )

    candidates = await asyncio.to_thread(
        discover_key_phrase_candidates,
        list(documents),
        configuration=configuration,
        protected_phrases=frozenset(previous_phrases),
    )

    if not candidates:
        # Nothing survived the statistics. Never destroy a working profile on a
        # run that produced nothing — an unreadable corpus looks exactly like a
        # speaker with no signature phrases, and only one of those should cost
        # the speaker their voice profile.
        if previous_phrases:
            logger.info(
                "Signature phrase discovery produced no candidates; keeping the "
                "%d previously stored phrases",
                len(previous_phrases),
            )
            return KeyPhraseProfile(
                phrases=list(previous_phrases),
                entries=list(
                    (previous_profile_detail or {}).get("entries") or []
                ),
                judgement_cache=judgement_cache,
            )
        return KeyPhraseProfile(judgement_cache=judgement_cache)

    if settings["enabled"]:
        judgements = await judge_key_phrase_candidates(
            candidates,
            speaker_name=speaker_name,
            previously_judged=judgement_cache,
            batch_size=settings["batch_size"],
            concurrency=settings["concurrency"],
            timeout_seconds=settings["timeout_seconds"],
        )
    else:
        # The gate is off: keep the statistical shortlist as-is, marked so the
        # records never masquerade as model confirmations in the stored detail.
        judgements = {
            candidate.phrase: _fallback_judgement("judging disabled")
            for candidate in candidates
        }

    incumbent_phrases = set(previous_phrases)
    incumbency_bonus = settings["incumbency_bonus"]

    kept: List[Dict[str, Any]] = []
    histogram: Counter[str] = Counter()
    for candidate in candidates:
        judgement = judgements.get(candidate.phrase) or _fallback_judgement(
            "no judgement available"
        )
        classification = judgement.get("classification")
        confidence = judgement.get("confidence", "low")
        source = judgement.get("judgement_source", MODEL_JUDGEMENT_SOURCE)
        histogram[str(classification)] += 1

        if classification != SIGNATURE_STYLE_CLASSIFICATION:
            continue
        model_confirmed = (
            source == MODEL_JUDGEMENT_SOURCE and confidence in CONFIRMING_CONFIDENCES
        )
        if source == MODEL_JUDGEMENT_SOURCE and not model_confirmed:
            # A low-confidence style call is a rejection: the prompt asks the
            # model to choose "low" when unsure, and admitting an unsure phrase
            # into the voice profile is the costly direction of error.
            continue

        ranking_score = candidate.log_ratio_over_reference + (
            incumbency_bonus if candidate.phrase in incumbent_phrases else 0.0
        )
        entry = candidate.as_storable_dictionary()
        entry.update(
            {
                "classification": classification,
                "confidence": confidence,
                "reason": judgement.get("reason", ""),
                "judgement_source": source,
                "model_confirmed": model_confirmed,
                "ranking_score": ranking_score,
            }
        )
        kept.append(entry)

    # Model-confirmed phrases outrank fail-open ones, so a classifier outage
    # degrades the tail of the list rather than displacing confirmed phrases.
    kept.sort(
        key=lambda entry: (
            not entry["model_confirmed"],
            -entry["ranking_score"],
            -entry["occurrence_count"],
            entry["phrase"],
        )
    )
    kept = kept[: max(1, settings["maximum_phrases"])]

    # Only real model judgements are cached. A fail-open record must be retried
    # on the next calibration, not remembered as if the model had spoken.
    refreshed_cache = dict(judgement_cache)
    for phrase, judgement in judgements.items():
        if judgement.get("judgement_source") == MODEL_JUDGEMENT_SOURCE:
            refreshed_cache[phrase] = dict(judgement)

    # NOTE: there is deliberately NO "keep the previous phrases when nothing
    # survived judging" branch here. Reaching this point means the corpus DID
    # produce candidates and the judge DID return verdicts on them — it simply
    # rejected them all, which is a real answer and must be stored as one.
    #
    # An earlier version preserved the previous phrases here, and on live data
    # that pinned exactly the worst profiles to exactly their worst phrases: an
    # avatar whose stored list was film-credit fragments, and another whose list
    # was "and he" / "and i" / "and she", were both re-judged, had every
    # candidate correctly rejected, and then had their junk written straight
    # back. The profiles that most needed clearing were the only ones that could
    # never be cleared.
    #
    # An empty phrase set is a supported state everywhere downstream: the
    # SIGNATURE PHRASES section says so in its own text, key_phrase_occurrence_rate
    # returns its neutral 0.0, and _publishable_avatar_key_phrase_rate reports
    # None rather than a fake zero. The transient failures this guard was written
    # for are handled above, where an unreadable or candidate-less corpus keeps
    # the previous profile, and by the caller's own exception handler.

    logger.info(
        "Signature phrase discovery: %d candidates -> %d phrases (%s)",
        len(candidates),
        len(kept),
        ", ".join(
            f"{name}={count}" for name, count in sorted(histogram.items())
        )
        or "no judgements",
    )

    return KeyPhraseProfile(
        phrases=[entry["phrase"] for entry in kept],
        entries=kept,
        judgement_cache=refreshed_cache,
        classification_histogram=dict(histogram),
    )
