"""Find and acquire the reference image and reference audio an avatar lacks.

Creating an avatar used to leave an empty shell: no portrait, no voice, nothing
learned, until the creator uploaded something. Deep research already answers
"what is true about this person"; this module answers the two questions that
made the avatar feel absent — "what does this person look like" and "what does
this person sound like" — and acquires the answers on the creator's behalf.

The shape here is deliberately NOT the tool-calling researcher loop the fact
research follows. Finding one usable portrait is a bounded selection problem,
not open-ended research: the candidate set is enumerable, each candidate costs a
download and a vision call, and the answer is "the first one that clears every
bar". So this gathers candidates from the pages the research already read, tops
the set up with a direct search when those pages did not offer enough, and then
judges candidates in rank order until one is accepted. Every step is bounded by
a configured ceiling rather than by a model deciding when to stop, because this
runs for every avatar on every tier and an unbounded loop here spends real money
per created avatar.

Order matters. The portrait is acquired first, and the acquired portrait is then
used to judge video candidates: knowing what the subject looks like is the most
reliable way to tell whether a given recording is really of that person rather
than about them.

Three rules hold throughout:

* **The creator always wins.** The portrait is written with ``replace=False``
  and the reference clip through ``store_reference_audio(replace=False)``, and
  the acquisition re-reads what the avatar holds immediately before handing
  anything to the media pipeline. A picture the creator uploaded while the
  research was running is never replaced.
* **Once per avatar.** ``claim_bootstrap`` writes a claim row before any work,
  so re-running research on an avatar acquires nothing a second time. That is
  the cost bound that makes running this for free-tier accounts safe.
* **Uncertainty means acquiring nothing.** A subject the research cannot
  confidently identify — a private individual, a name with no public presence,
  a namesake — gets no portrait and no voice, and the reason is recorded so the
  settings screen can ask the creator to upload one instead. Attaching a
  stranger's face or voice to an avatar is worse than leaving it blank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from src.anubis.utils.research.web_search import (
    ImageCandidate,
    SearchResult,
    search_images,
    search_web,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_NAMESPACE_CATEGORY = "research_bootstrap"

EventSink = Callable[[dict[str, Any]], None]

# Origins ranked by how likely the picture is to be a portrait of the page's
# subject. A page's own social-card image is the page's answer to "which
# picture is this page about"; a Tavily image result matched the query but
# nothing has vouched for what it depicts.
_IMAGE_ORIGIN_RANK = {
    "wikipedia": 0,
    "og:image": 1,
    "twitter:image": 2,
    "image_src": 3,
    "json-ld": 4,
    "tavily": 5,
}


def _is_enabled(context: Any) -> bool:
    value = str(getattr(context, "deep_research_bootstrap_enabled", "true") or "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def bootstrap_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the store namespace recording this avatar's acquisition attempt."""
    return (creator_id, assistant_id, BOOTSTRAP_NAMESPACE_CATEGORY)


# ── what the API layer lends the pipeline ───────────────────────────────────


class BootstrapGateway:
    """The API-side operations the acquisition needs, handed in rather than imported.

    The research pipeline lives under ``src/anubis/utils`` and must never import
    ``src/api/webapp.py``: the API imports the pipeline, and the reverse would be
    a cycle. The four operations here are the whole of what acquisition needs
    from the API — download an image, start a reference-image media job, start a
    reference-audio media job, and wait for one of those jobs — so the API
    constructs this object and passes it in, exactly as it already passes ``emit``
    and ``is_cancelled``.
    """

    def __init__(
        self,
        *,
        fetch_image_bytes: Callable[[str], Awaitable[tuple[str, bytes, str]]],
        start_reference_image_job: Callable[..., Awaitable[dict[str, Any]]],
        start_reference_audio_job: Callable[..., Awaitable[dict[str, Any]]],
        await_media_job: Callable[[str, float], Awaitable[str]],
    ) -> None:
        """Hold the four API-side operations the acquisition is allowed to call."""
        self.fetch_image_bytes = fetch_image_bytes
        self.start_reference_image_job = start_reference_image_job
        self.start_reference_audio_job = start_reference_audio_job
        self.await_media_job = await_media_job


# ── the two vetting schemas ─────────────────────────────────────────────────


class PortraitVerdict(BaseModel):
    """Whether one candidate picture may become this avatar's face."""

    depicts_named_subject: bool = Field(
        description="Whether the picture shows the named subject rather than someone else, a namesake, or a group in which the subject cannot be picked out."
    )
    single_person: bool = Field(description="Whether exactly one person is shown.")
    face_clearly_visible: bool = Field(
        description="Whether the person's face is visible, in focus, and large enough to read."
    )
    obstructed: bool = Field(
        description="Whether the face is blocked by a hand, a microphone, sunglasses, heavy shadow, or a watermark across it."
    )
    is_photograph: bool = Field(
        description="Whether this is a photograph of a real person, rather than a drawing, a painting, a rendered or generated image, a collage, a screenshot of text, or a logo."
    )
    moderation_risk: Literal["low", "high"] = Field(
        description="Whether an image vendor's content moderation would refuse to work from this picture."
    )
    confidence: float = Field(
        description="How sure you are, from 0.0 to 1.0, that this picture shows the named subject."
    )
    reasoning: str = Field(
        description="One or two sentences saying what the picture shows and why it does or does not qualify."
    )


class VideoSubjectPresence(BaseModel):
    """Whether the person in the reference photograph appears in a video's thumbnail."""

    same_person_present: bool = Field(
        description="Whether the person shown in the first image also appears in the second image."
    )
    is_the_focus: bool = Field(
        description="Whether that person is the subject of the second image rather than someone incidental in it."
    )
    confidence: float = Field(
        description="How sure you are, from 0.0 to 1.0, that the same person appears."
    )
    reasoning: str = Field(description="One or two sentences explaining the judgement.")


class SpeakingVideoScore(BaseModel):
    """How well one candidate recording would teach this avatar's voice."""

    subject_speaks_the_most: bool = Field(
        description="Whether the named subject is likely to speak more than anyone else across this recording."
    )
    format: Literal[
        "interview", "talk", "podcast", "monologue", "panel", "documentary", "other"
    ] = Field(
        description="What kind of recording this is, judged from its title and description."
    )
    is_about_the_subject: bool = Field(
        description="Whether the recording features the named subject themselves, rather than other people discussing or reporting on them."
    )
    score: float = Field(
        description="From 0.0 to 1.0, how good this recording is as the single source for learning the subject's voice."
    )
    reasoning: str = Field(description="One or two sentences explaining the score.")


# ── reading and claiming what the avatar already holds ──────────────────────


@dataclass
class BootstrapNeeds:
    """Which reference assets this avatar is still missing, and why."""

    needs_portrait: bool
    needs_voice: bool
    portrait_reason: str | None = None
    voice_reason: str | None = None

    @property
    def needs_anything(self) -> bool:
        """Whether there is anything at all for the acquisition to do."""
        return self.needs_portrait or self.needs_voice


async def assess_missing_assets(
    store: Any, *, creator_id: str, assistant_id: str, context: Any
) -> BootstrapNeeds:
    """Read what this avatar holds now and report which reference assets are absent.

    Called twice: once to decide whether to start, and again immediately before
    handing an acquired asset to the media pipeline, because the creator may
    have uploaded their own in between.
    """
    from src.anubis.utils.media_generation.reference_image import read_reference_image
    from src.anubis.utils.voice.reference_audio import read_usable_reference_audio

    portrait = await read_reference_image(store, creator_id, assistant_id)
    reference_clip, clip_problem = await read_usable_reference_audio(
        store, creator_id, assistant_id, context=context
    )
    return BootstrapNeeds(
        needs_portrait=portrait is None,
        needs_voice=reference_clip is None,
        portrait_reason=None
        if portrait is None
        else "This avatar already has a portrait.",
        voice_reason=clip_problem,
    )


async def claim_bootstrap(store: Any, *, creator_id: str, assistant_id: str) -> bool:
    """Record that acquisition has started, and report whether this call won the claim.

    Returns ``False`` when a claim already exists. That is what limits the whole
    feature to one attempt per avatar: pressing "Research & verify facts" again
    re-researches the facts, as it always has, and acquires no further media.
    """
    namespace = bootstrap_namespace(creator_id, assistant_id)
    try:
        existing_claim = await store.aget(namespace, assistant_id)
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Bootstrap claim lookup failed (continuing): %s", read_error)
        existing_claim = None
    if existing_claim is not None:
        return False
    await store.aput(
        namespace,
        key=assistant_id,
        value={
            "status": "running",
            "claimed_at": datetime.now(UTC).isoformat(),
        },
    )
    return True


async def record_bootstrap_outcome(
    store: Any, *, creator_id: str, assistant_id: str, outcome: dict[str, Any]
) -> None:
    """Persist what acquisition ended up doing, so a screen can report it later.

    The progress stream is gone by the time the creator opens the settings page,
    so "no photograph of this subject could be found" has to live somewhere
    durable for the portrait tile and the voice panel to read.
    """
    namespace = bootstrap_namespace(creator_id, assistant_id)
    try:
        await store.aput(
            namespace,
            key=assistant_id,
            value={
                "status": "finished",
                "finished_at": datetime.now(UTC).isoformat(),
                **outcome,
            },
        )
    except Exception as write_error:  # noqa: BLE001 - reporting must never fail the run
        logger.warning(
            "Could not record the acquisition outcome for %s: %s",
            assistant_id,
            write_error,
        )


async def read_bootstrap_outcome(
    store: Any, *, creator_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Return the recorded acquisition outcome for this avatar, if there is one."""
    try:
        item = await store.aget(
            bootstrap_namespace(creator_id, assistant_id), assistant_id
        )
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Bootstrap outcome lookup failed (continuing): %s", read_error)
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    return dict(value or {}) or None


# ── portrait ────────────────────────────────────────────────────────────────


def _normalized_candidate_key(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def portrait_candidates_from_sources(
    sources: list[SearchResult], *, limit: int
) -> list[ImageCandidate]:
    """Rank the pictures the research's own pages declared, best portrait bet first.

    These cost nothing: the pages were fetched to read their facts, and the
    pictures came out of the same HTML. A page that supported more of the
    research's queries is a page more about this subject, so its picture is
    ranked above one from a page that matched a single query.
    """
    ranked: list[tuple[int, int, ImageCandidate]] = []
    seen: set[str] = set()
    for source in sources:
        query_support = -len(set(source.queries or []))
        for candidate in source.images or []:
            key = _normalized_candidate_key(candidate.url)
            if not key or key in seen:
                continue
            seen.add(key)
            origin_rank = _IMAGE_ORIGIN_RANK.get(candidate.origin, 9)
            ranked.append((origin_rank, query_support, candidate))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [candidate for _, _, candidate in ranked[:limit]]


async def search_portrait_candidates(
    subject_name: str, *, context: Any, limit: int
) -> list[ImageCandidate]:
    """Search the web for photographs of the subject.

    Two queries rather than one: "portrait photo" tends to return posed
    headshots and "photo" returns press pictures, and a subject well covered by
    one is often thin in the other. Deployments with no Tavily key get nothing
    here and fall back on the pictures the read pages declared.
    """
    queries = [f"{subject_name} portrait photo", f"{subject_name} headshot photo"]
    candidates: list[ImageCandidate] = []
    seen: set[str] = set()
    for query in queries:
        if len(candidates) >= limit:
            break
        for candidate in await search_images(query, limit=limit, context=context):
            key = _normalized_candidate_key(candidate.url)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
    return candidates


async def vet_portrait_candidate(
    image_data_uri: str, *, subject_name: str, subject_summary: str
) -> PortraitVerdict | None:
    """Ask the vision model whether this picture may become the avatar's face.

    Returns ``None`` when the model could not answer, which the caller treats as
    a rejection: an unanswered question about whose face this is must never
    resolve in favour of installing it.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_image_description_model
        from src.anubis.utils.prompts.system_prompts import (
            BOOTSTRAP_PORTRAIT_VETTING_PROMPT,
        )

        model = init_image_description_model().with_structured_output(
            schema=PortraitVerdict
        )
        subject_block = f"Name: {subject_name}"
        if subject_summary:
            subject_block += f"\nWhat is known about {subject_name}: {subject_summary}"
        response = await model.ainvoke(
            [
                SystemMessage(content=BOOTSTRAP_PORTRAIT_VETTING_PROMPT),
                HumanMessage(
                    content=[
                        {"type": "text", "text": subject_block},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ]
                ),
            ]
        )
        if isinstance(response, PortraitVerdict):
            return response
        return PortraitVerdict.model_validate(response)
    except Exception as vetting_error:  # noqa: BLE001 - an unanswerable question is a rejection
        logger.info("Could not vet a portrait candidate: %s", vetting_error)
        return None


def portrait_verdict_accepts(
    verdict: PortraitVerdict, *, minimum_confidence: float
) -> bool:
    """Whether a verdict clears every bar a picture must clear to become the face."""
    return bool(
        verdict.depicts_named_subject
        and verdict.single_person
        and verdict.face_clearly_visible
        and not verdict.obstructed
        and verdict.is_photograph
        and verdict.moderation_risk == "low"
        and float(verdict.confidence or 0.0) >= minimum_confidence
    )


async def acquire_portrait(
    store: Any,
    context: Any,
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    subject_summary: str,
    sources: list[SearchResult],
    gateway: BootstrapGateway,
    emit: EventSink,
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Find, vet, and install a photograph of the subject as the avatar's portrait.

    Returns a result dictionary describing what happened; it never raises, so a
    failure here can only cost the portrait, never the research run.
    """
    maximum_candidates = max(
        1, int(getattr(context, "deep_research_bootstrap_max_image_candidates", 8) or 8)
    )
    minimum_confidence = float(
        getattr(context, "deep_research_bootstrap_min_portrait_confidence", 0.7) or 0.7
    )

    candidates = portrait_candidates_from_sources(sources, limit=maximum_candidates)
    if len(candidates) < maximum_candidates:
        # Reflect-then-search-again: the pages the research read did not offer
        # enough portrait candidates on their own, so go looking for pictures.
        already_seen = {_normalized_candidate_key(entry.url) for entry in candidates}
        for extra in await search_portrait_candidates(
            subject_name, context=context, limit=maximum_candidates
        ):
            if _normalized_candidate_key(extra.url) in already_seen:
                continue
            candidates.append(extra)
            if len(candidates) >= maximum_candidates:
                break
    candidates = candidates[:maximum_candidates]

    emit(
        {
            "type": "research_progress",
            "stage": "finding_portrait",
            "candidates": len(candidates),
        }
    )
    if not candidates:
        emit(
            {
                "type": "research_progress",
                "stage": "portrait_not_found",
                "reason": "no candidate pictures were found for this subject",
            }
        )
        return {
            "acquired": False,
            "reason": "No pictures of this subject could be found on the web.",
        }

    rejections: list[str] = []
    for candidate in candidates:
        if is_cancelled():
            return {"acquired": False, "reason": "cancelled", "cancelled": True}
        try:
            mime_type, body, image_data_uri = await gateway.fetch_image_bytes(
                candidate.url
            )
        except Exception as download_error:  # noqa: BLE001 - try the next candidate
            logger.info("Could not download %s: %s", candidate.url, download_error)
            rejections.append(f"{candidate.url}: could not be downloaded")
            continue
        emit(
            {
                "type": "research_progress",
                "stage": "vetting_portrait",
                "url": candidate.url,
                "origin": candidate.origin,
            }
        )
        verdict = await vet_portrait_candidate(
            image_data_uri, subject_name=subject_name, subject_summary=subject_summary
        )
        if verdict is None:
            rejections.append(f"{candidate.url}: could not be assessed")
            continue
        if not portrait_verdict_accepts(verdict, minimum_confidence=minimum_confidence):
            rejections.append(f"{candidate.url}: {verdict.reasoning}")
            continue

        # The creator may have uploaded their own portrait while this ran. Read
        # again before handing anything over; the media pipeline's replace=False
        # is the second guard, and this one keeps the work from starting at all.
        needs = await assess_missing_assets(
            store, creator_id=creator_id, assistant_id=assistant_id, context=context
        )
        if not needs.needs_portrait:
            return {
                "acquired": False,
                "reason": "The creator's own portrait arrived first; it was kept.",
            }

        emit(
            {
                "type": "research_progress",
                "stage": "portrait_found",
                "url": candidate.url,
                "reasoning": verdict.reasoning,
            }
        )
        try:
            batch = await gateway.start_reference_image_job(
                filename=candidate.url,
                mime_type=mime_type,
                content=body,
                source_url=candidate.url,
            )
        except Exception as job_error:  # noqa: BLE001 - the research still succeeded
            logger.warning("Could not start the portrait media job: %s", job_error)
            return {
                "acquired": False,
                "reason": f"The portrait could not be stored: {job_error}",
                "source_url": candidate.url,
            }
        if batch.get("status") != "started":
            return {
                "acquired": False,
                "reason": batch.get("detail") or "The portrait media job was refused.",
                "source_url": candidate.url,
            }

        stored = await _await_asset(
            store,
            context,
            gateway=gateway,
            job_id=batch.get("job_id"),
            creator_id=creator_id,
            assistant_id=assistant_id,
            asset="portrait",
        )
        emit(
            {
                "type": "research_progress",
                "stage": "portrait_stored" if stored else "portrait_pending",
                "url": candidate.url,
            }
        )
        return {
            "acquired": True,
            "stored": stored,
            "source_url": candidate.url,
            "image_data_uri": image_data_uri,
            "job_id": batch.get("job_id"),
            "reasoning": verdict.reasoning,
        }

    emit(
        {
            "type": "research_progress",
            "stage": "portrait_not_found",
            "reason": "no candidate could be confirmed as this subject",
            "candidates_examined": len(candidates),
        }
    )
    return {
        "acquired": False,
        "reason": (
            f"{len(candidates)} picture(s) were examined and none could be confirmed as "
            f"{subject_name}."
        ),
        "rejections": rejections[:10],
    }


# ── voice ───────────────────────────────────────────────────────────────────


def youtube_candidates_from_sources(sources: list[SearchResult]) -> list[str]:
    """Return the single-video YouTube links already among the research's sources."""
    return _filter_speaking_video_urls(source.url for source in sources)


def _filter_speaking_video_urls(urls: Any) -> list[str]:
    """Keep only URLs that can become one avatar's reference recording.

    A playlist or a channel is rejected twice over — ``_classify_url`` labels
    both ``youtube_playlist``, and ``reference_source_rejection`` refuses
    enumerated links outright — because "the video where this person speaks
    most" has no meaning for a link that stands for many videos, and the
    reference clip would be cut from whichever one the enumerator happened to
    reach first.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url
    from src.anubis.utils.voice.reference_eligibility import reference_source_rejection

    kept: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = (url or "").strip()
        key = _normalized_candidate_key(cleaned)
        if not cleaned or key in seen:
            continue
        try:
            if _classify_url(cleaned) != "youtube":
                continue
        except Exception:  # noqa: BLE001 - an unclassifiable URL is not a candidate
            continue
        if (
            reference_source_rejection(
                filename=cleaned, url_kind=None, media_type="video"
            )
            is not None
        ):
            continue
        seen.add(key)
        kept.append(cleaned)
    return kept


async def search_speaking_videos(
    subject_name: str, *, context: Any, limit: int
) -> list[str]:
    """Search for recordings in which the subject is the one talking.

    The queries name the formats where one person does most of the speaking —
    an interview, a podcast appearance, a talk — rather than asking for videos
    "about" the subject, which returns commentary by other people.
    """
    queries = [
        f"{subject_name} interview video",
        f"{subject_name} podcast episode",
        f"{subject_name} talk",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if len(found) >= limit:
            break
        try:
            results = await search_web(query, limit=limit, context=context)
        except Exception as search_error:  # noqa: BLE001 - one failed query is not fatal
            logger.info("Video search failed for %r: %s", query, search_error)
            continue
        for url in _filter_speaking_video_urls(result.url for result in results):
            key = _normalized_candidate_key(url)
            if key in seen:
                continue
            seen.add(key)
            found.append(url)
            if len(found) >= limit:
                break
    return found


async def probe_video_candidates(
    urls: list[str], *, context: Any
) -> list[dict[str, Any]]:
    """Read each candidate's own metadata and keep the ones worth transcribing.

    Nothing is downloaded here. The duration bounds are the real cost control on
    this whole feature: transcription and diarization are priced by length, and
    this is what stops one automatically created avatar from ingesting a
    four-hour stream.
    """
    from src.anubis.utils.utility import get_remote_video_metadata

    minimum_seconds = float(
        getattr(context, "deep_research_bootstrap_min_video_seconds", 120.0) or 0.0
    )
    maximum_seconds = float(
        getattr(context, "deep_research_bootstrap_max_video_seconds", 2400.0) or 0.0
    )
    surviving: list[dict[str, Any]] = []
    for url in urls:
        try:
            metadata = await get_remote_video_metadata(url)
        except Exception as probe_error:  # noqa: BLE001 - skip an unreadable candidate
            logger.info("Could not probe %s: %s", url, probe_error)
            continue
        duration = float(metadata.get("duration") or 0.0)
        if metadata.get("live_status") in ("is_live", "is_upcoming"):
            continue
        if int(metadata.get("age_limit") or 0) > 0:
            continue
        if duration < minimum_seconds:
            continue
        if maximum_seconds > 0 and duration > maximum_seconds:
            continue
        surviving.append({"url": url, **metadata})
    return surviving


async def confirm_subject_in_thumbnail(
    reference_image_data_uri: str, thumbnail_url: str
) -> VideoSubjectPresence | None:
    """Ask whether the person in the portrait appears in this video's thumbnail.

    This is what the acquired portrait is for. A title can say "interview with
    <name>" when the video is an interview *about* them, and a channel can
    reuse a person's name; the picture is the check that the recording is of the
    person whose voice the avatar is trying to learn.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_image_description_model
        from src.anubis.utils.prompts.system_prompts import (
            BOOTSTRAP_VIDEO_SUBJECT_PRESENCE_PROMPT,
        )

        model = init_image_description_model().with_structured_output(
            schema=VideoSubjectPresence
        )
        response = await model.ainvoke(
            [
                SystemMessage(content=BOOTSTRAP_VIDEO_SUBJECT_PRESENCE_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "The first image is the reference photograph. The second image is the video thumbnail.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": reference_image_data_uri},
                        },
                        {"type": "image_url", "image_url": {"url": thumbnail_url}},
                    ]
                ),
            ]
        )
        if isinstance(response, VideoSubjectPresence):
            return response
        return VideoSubjectPresence.model_validate(response)
    except Exception as presence_error:  # noqa: BLE001 - fall back on the text scorer
        logger.info(
            "Could not compare a thumbnail with the portrait: %s", presence_error
        )
        return None


async def rank_speaking_videos(
    candidates: list[dict[str, Any]], *, subject_name: str, subject_summary: str
) -> list[dict[str, Any]]:
    """Score each surviving candidate and return them best first.

    One structured call per candidate over its own title, channel and
    description. Candidates the model will not vouch for — someone else talking
    about the subject, or a recording where the subject is one voice among many
    — are dropped rather than ranked low, because the reference clip is cut from
    whoever speaks most and a panel would teach the wrong voice.
    """
    from src.anubis.utils.prompts.system_prompts import (
        BOOTSTRAP_SPEAKING_VIDEO_SCORING_PROMPT,
    )
    from src.anubis.utils.research.deep_research import invoke_structured

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        described = (
            f"Subject: {subject_name}\n"
            f"What is known about {subject_name}: {subject_summary or 'nothing yet'}\n\n"
            f"Video title: {candidate.get('title') or 'unknown'}\n"
            f"Channel: {candidate.get('uploader') or 'unknown'}\n"
            f"Duration: {round(float(candidate.get('duration') or 0.0) / 60.0)} minutes\n"
            f"Description: {(candidate.get('description') or '')[:1500]}"
        )
        try:
            score = await invoke_structured(
                SpeakingVideoScore, BOOTSTRAP_SPEAKING_VIDEO_SCORING_PROMPT, described
            )
        except Exception as scoring_error:  # noqa: BLE001 - skip an unscorable candidate
            logger.info("Could not score %s: %s", candidate.get("url"), scoring_error)
            continue
        if not (score.subject_speaks_the_most and score.is_about_the_subject):
            continue
        scored.append(
            {
                **candidate,
                "score": float(score.score or 0.0),
                "reasoning": score.reasoning,
            }
        )
    scored.sort(key=lambda entry: -entry["score"])
    return scored


async def acquire_voice(
    store: Any,
    context: Any,
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    subject_summary: str,
    sources: list[SearchResult],
    reference_image_data_uri: str | None,
    gateway: BootstrapGateway,
    emit: EventSink,
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Find one recording of the subject speaking and make it the reference audio.

    Returns a result dictionary; it never raises, so a failure here can only
    cost the voice, never the research run.
    """
    maximum_candidates = max(
        1, int(getattr(context, "deep_research_bootstrap_max_video_candidates", 6) or 6)
    )

    candidate_urls = youtube_candidates_from_sources(sources)
    if len(candidate_urls) < maximum_candidates:
        already_seen = {_normalized_candidate_key(url) for url in candidate_urls}
        for extra in await search_speaking_videos(
            subject_name, context=context, limit=maximum_candidates
        ):
            if _normalized_candidate_key(extra) in already_seen:
                continue
            candidate_urls.append(extra)
            if len(candidate_urls) >= maximum_candidates:
                break
    candidate_urls = candidate_urls[:maximum_candidates]

    emit(
        {
            "type": "research_progress",
            "stage": "finding_voice",
            "candidates": len(candidate_urls),
        }
    )
    if is_cancelled():
        return {"acquired": False, "reason": "cancelled", "cancelled": True}

    probed = await probe_video_candidates(candidate_urls, context=context)
    if not probed:
        emit(
            {
                "type": "research_progress",
                "stage": "voice_not_found",
                "reason": "no recording of a usable length was found",
            }
        )
        return {
            "acquired": False,
            "reason": "No recording of this subject speaking, of a usable length, could be found.",
        }

    # The portrait is the strongest signal available for "is this really them",
    # so use it to drop candidates before spending a scoring call on each.
    if reference_image_data_uri:
        confirmed: list[dict[str, Any]] = []
        for candidate in probed:
            thumbnail = candidate.get("thumbnail") or ""
            if not thumbnail:
                confirmed.append(candidate)
                continue
            presence = await confirm_subject_in_thumbnail(
                reference_image_data_uri, thumbnail
            )
            if presence is None:
                # The comparison could not run; let the text scorer decide
                # rather than dropping a candidate for a model failure.
                confirmed.append(candidate)
                continue
            if presence.same_person_present and presence.is_the_focus:
                confirmed.append(
                    {**candidate, "thumbnail_reasoning": presence.reasoning}
                )
        if confirmed:
            probed = confirmed

    ranked = await rank_speaking_videos(
        probed, subject_name=subject_name, subject_summary=subject_summary
    )
    if not ranked:
        emit(
            {
                "type": "research_progress",
                "stage": "voice_not_found",
                "reason": "no recording could be confirmed as this subject speaking",
            }
        )
        return {
            "acquired": False,
            "reason": (
                f"{len(probed)} recording(s) were examined and none could be confirmed as "
                f"{subject_name} speaking."
            ),
        }

    winner = ranked[0]
    needs = await assess_missing_assets(
        store, creator_id=creator_id, assistant_id=assistant_id, context=context
    )
    if not needs.needs_voice:
        return {
            "acquired": False,
            "reason": "The creator's own recording arrived first; it was kept.",
        }

    emit(
        {
            "type": "research_progress",
            "stage": "voice_found",
            "url": winner["url"],
            "title": winner.get("title"),
            "duration_seconds": winner.get("duration"),
            "reasoning": winner.get("reasoning"),
        }
    )
    try:
        batch = await gateway.start_reference_audio_job(url=winner["url"])
    except Exception as job_error:  # noqa: BLE001 - the research still succeeded
        logger.warning("Could not start the voice media job: %s", job_error)
        return {
            "acquired": False,
            "reason": f"The recording could not be ingested: {job_error}",
            "source_url": winner["url"],
        }
    if batch.get("status") != "started":
        return {
            "acquired": False,
            "reason": batch.get("detail") or "The voice media job was refused.",
            "source_url": winner["url"],
        }

    stored = await _await_asset(
        store,
        context,
        gateway=gateway,
        job_id=batch.get("job_id"),
        creator_id=creator_id,
        assistant_id=assistant_id,
        asset="voice",
    )
    emit(
        {
            "type": "research_progress",
            "stage": "voice_stored" if stored else "voice_pending",
            "url": winner["url"],
            "title": winner.get("title"),
        }
    )
    return {
        "acquired": True,
        "stored": stored,
        "source_url": winner["url"],
        "title": winner.get("title"),
        "duration_seconds": winner.get("duration"),
        "job_id": batch.get("job_id"),
    }


# ── shared waiting ──────────────────────────────────────────────────────────


async def _await_asset(
    store: Any,
    context: Any,
    *,
    gateway: BootstrapGateway,
    job_id: str | None,
    creator_id: str,
    assistant_id: str,
    asset: str,
) -> bool:
    """Wait for a media job, then report whether the asset really landed.

    A finished job is not proof the asset was stored — the media pipeline can
    finish having declined to overwrite a portrait the creator uploaded — so the
    store is what is read, not the job's status.
    """
    if not job_id:
        return False
    wait_seconds = float(
        getattr(context, "deep_research_bootstrap_media_wait_seconds", 900.0) or 900.0
    )
    try:
        await gateway.await_media_job(job_id, wait_seconds)
    except Exception as wait_error:  # noqa: BLE001 - the job carries on without us
        logger.info("Stopped waiting for media job %s: %s", job_id, wait_error)
        return False
    needs = await assess_missing_assets(
        store, creator_id=creator_id, assistant_id=assistant_id, context=context
    )
    return not (needs.needs_portrait if asset == "portrait" else needs.needs_voice)


# ── the stage ───────────────────────────────────────────────────────────────


async def run_asset_bootstrap(
    store: Any,
    context: Any,
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    subject_summary: str,
    sources: list[SearchResult],
    gateway: BootstrapGateway,
    emit: EventSink,
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Acquire whichever reference assets this avatar is missing.

    The portrait is acquired first because the voice half uses it. Both halves
    are best-effort: whatever cannot be found is reported with a reason, and the
    reason is persisted so the settings screen can ask the creator for it.
    """
    if not _is_enabled(context):
        return {"skipped": "disabled"}

    needs = await assess_missing_assets(
        store, creator_id=creator_id, assistant_id=assistant_id, context=context
    )
    if not needs.needs_anything:
        return {"skipped": "nothing_missing"}
    if not await claim_bootstrap(
        store, creator_id=creator_id, assistant_id=assistant_id
    ):
        return {"skipped": "already_attempted"}

    emit(
        {
            "type": "research_progress",
            "stage": "bootstrap_scoping",
            "needs_portrait": needs.needs_portrait,
            "needs_voice": needs.needs_voice,
        }
    )

    outcome: dict[str, Any] = {
        "subject_name": subject_name,
        "portrait": None,
        "voice": None,
        "media_urls": [],
    }

    reference_image_data_uri: str | None = None
    if needs.needs_portrait and not is_cancelled():
        try:
            portrait_result = await acquire_portrait(
                store,
                context,
                creator_id=creator_id,
                assistant_id=assistant_id,
                subject_name=subject_name,
                subject_summary=subject_summary,
                sources=sources,
                gateway=gateway,
                emit=emit,
                is_cancelled=is_cancelled,
            )
        except Exception as portrait_error:  # noqa: BLE001 - never fail the research run
            logger.exception("Portrait acquisition failed for %s", assistant_id)
            portrait_result = {"acquired": False, "reason": str(portrait_error)}
        reference_image_data_uri = portrait_result.pop("image_data_uri", None)
        outcome["portrait"] = portrait_result
        if portrait_result.get("source_url"):
            outcome["media_urls"].append(portrait_result["source_url"])
    if reference_image_data_uri is None:
        # Either the portrait was already there or acquisition just stored one;
        # either way the stored picture is what the voice half should compare
        # thumbnails against.
        from src.anubis.utils.media_generation.reference_image import (
            read_reference_image,
        )

        stored_portrait = await read_reference_image(store, creator_id, assistant_id)
        if stored_portrait:
            reference_image_data_uri = stored_portrait.get("reference_image_data")

    if needs.needs_voice and not is_cancelled():
        try:
            voice_result = await acquire_voice(
                store,
                context,
                creator_id=creator_id,
                assistant_id=assistant_id,
                subject_name=subject_name,
                subject_summary=subject_summary,
                sources=sources,
                reference_image_data_uri=reference_image_data_uri,
                gateway=gateway,
                emit=emit,
                is_cancelled=is_cancelled,
            )
        except Exception as voice_error:  # noqa: BLE001 - never fail the research run
            logger.exception("Voice acquisition failed for %s", assistant_id)
            voice_result = {"acquired": False, "reason": str(voice_error)}
        outcome["voice"] = voice_result
        if voice_result.get("source_url"):
            outcome["media_urls"].append(voice_result["source_url"])

    await record_bootstrap_outcome(
        store, creator_id=creator_id, assistant_id=assistant_id, outcome=outcome
    )
    emit(
        {
            "type": "research_progress",
            "stage": "bootstrap_done",
            **_outcome_summary(outcome),
        }
    )
    return outcome


def _outcome_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    """Return the two booleans a progress line needs, without the whole outcome."""
    portrait = outcome.get("portrait") or {}
    voice = outcome.get("voice") or {}
    return {
        "portrait_acquired": bool(portrait.get("acquired")),
        "voice_acquired": bool(voice.get("acquired")),
    }


__all__ = [
    "BOOTSTRAP_NAMESPACE_CATEGORY",
    "BootstrapGateway",
    "BootstrapNeeds",
    "PortraitVerdict",
    "SpeakingVideoScore",
    "VideoSubjectPresence",
    "acquire_portrait",
    "acquire_voice",
    "assess_missing_assets",
    "bootstrap_namespace",
    "claim_bootstrap",
    "portrait_candidates_from_sources",
    "portrait_verdict_accepts",
    "probe_video_candidates",
    "rank_speaking_videos",
    "read_bootstrap_outcome",
    "record_bootstrap_outcome",
    "run_asset_bootstrap",
    "search_portrait_candidates",
    "search_speaking_videos",
    "youtube_candidates_from_sources",
]
