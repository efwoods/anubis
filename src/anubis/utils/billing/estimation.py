# src/anubis/utils/billing/estimation.py

"""Manual pre-request token estimation for every model-consuming request.

Every request that will reach a model (a message turn, a media upload's
extraction and identity-analysis passes) is estimated BEFORE the model is
called, so allotment gating can refuse the request while nothing has been
spent yet. Estimation is deliberately manual — no tokenizer library and no
token-counting API endpoint — because manual arithmetic over known quantities
(word counts, image dimensions, media durations, fixed prompt overheads) is
the fastest computation available on the request path.

Derivations, per the OpenAI token-counting and vision guides and the product
owner's audio cost model:

* Text: English averages three quarters of a word per token, so
  ``tokens = words × 4/3``. Where words cannot be counted (binary or unknown
  encodings), fall back to ``tokens = characters × 0.25`` (four characters
  per token).
* Images (patch-based vision models — the gpt-5-nano / gpt-5.4-nano class):
  ``input tokens = min(ceil(width/32) × ceil(height/32), 1536) × 2.46``,
  plus a fixed expected description-output allowance, because every uploaded
  or attached image is run through a vision-description pass.
* Audio (diarization/transcription): spoken dialogue averages 150 words per
  minute; at 4/3 tokens per word that is 200 transcript-output tokens per
  minute, alongside roughly 1,600 audio-input tokens per minute of encoded
  audio (≈ $0.006 per minute at $2.50 per million input tokens).
* Identity analysis is a MODULAR add-on: analysis passes re-read the
  extracted content (transcript or text), so the add-on is
  ``extracted content tokens × analysis pass count`` and is included only
  when the pipeline will actually analyze that item. Dropping analysis from
  the pipeline later automatically changes the pre-call estimate by passing
  ``include_analysis=False`` (or an analysis pass count of zero).

Estimation is FAIL-CLOSED: invalid inputs raise ``TokenEstimationError`` and
the request must be refused (HTTP 422) — nothing unestimated may proceed to a
model. The only sanctioned fallback is an unknown audio/video duration after
the media type is confirmed, for which callers substitute
``ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS`` (a conservative overestimate)
before calling into this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

# Text: three quarters of a word per token ⇒ 4/3 tokens per word.
ESTIMATED_TOKENS_PER_WORD = 4.0 / 3.0
# Fallback where words cannot be counted: four characters per token.
ESTIMATED_TOKENS_PER_TEXT_CHARACTER = 0.25

# Vision, patch-based models (gpt-5-nano / gpt-5.4-nano class): the image is
# cut into 32x32-pixel patches, capped at 1,536 patches (larger images are
# proportionally downscaled by the provider), and each patch costs the model
# multiplier in tokens (2.46 for the nano class).
VISION_PATCH_SIDE_PIXELS = 32
VISION_MAXIMUM_PATCHES_PER_IMAGE = 1536
VISION_TOKEN_MULTIPLIER = 2.46
# Every image also incurs a vision-description pass whose output feeds the
# conversation or the identity pipeline; allow a fixed output budget for it.
ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS = 300

# Audio: ~1,600 audio-input tokens per minute of encoded audio, plus 200
# transcript-output tokens per minute (150 spoken words/min × 4/3 tokens/word).
ESTIMATED_AUDIO_INPUT_TOKENS_PER_MINUTE = 1600
ESTIMATED_TRANSCRIPT_OUTPUT_TOKENS_PER_MINUTE = 200
# Sanctioned fallback duration when a confirmed audio/video item's length
# cannot be probed: assume ten minutes (conservative overestimate).
ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS = 600.0

MediaItemKind = Literal["text", "image", "audio", "video"]


class TokenEstimationError(ValueError):
    """An estimate could not be computed — the request must not reach a model.

    Estimation is fail-closed: the caller turns this into an HTTP 422 naming
    the item that could not be estimated, rather than proceeding unmetered.
    """


@dataclass(frozen=True)
class TokenEstimateBreakdown:
    """One request's pre-call token estimate, split into input versus output.

    The split matters because gating and billing treat the two sides
    differently:

    * ``input_tokens`` — everything the model will READ (the measured system
      prompt, fixed per-turn prompt overhead, the user's message and attached
      file text, image vision patches, audio-input tokens, analysis passes
      re-reading extracted content). The allotment gate refuses a request
      whose estimated INPUT tokens cannot fit under the remaining allotment.
    * ``output_tokens`` — everything the model is expected to WRITE (the
      reply budget, image-description output, transcript output). Output is
      never gated in advance: a request whose input fits may overshoot the
      allotment through its output exactly once, because the recorded actual
      total then blocks the next request.
    * ``total_tokens`` — input plus output; this is what counts against the
      allotment once the actual usage is recorded, and what the token rate
      limit projects with.
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def count_words(text: str | None) -> int:
    """Return the whitespace-delimited word count of ``text`` (0 for empty)."""
    if not text:
        return 0
    return len(text.split())


def estimate_text_tokens_from_words(word_count: int) -> int:
    """Estimate tokens for ``word_count`` words at 4/3 tokens per word."""
    if word_count < 0:
        raise TokenEstimationError(f"Negative word count: {word_count}")
    return math.ceil(word_count * ESTIMATED_TOKENS_PER_WORD)


def estimate_text_tokens_from_characters(character_count: int) -> int:
    """Estimate tokens for ``character_count`` characters at 4 characters per token.

    Fallback for content whose words cannot be counted (unknown encodings).
    """
    if character_count < 0:
        raise TokenEstimationError(f"Negative character count: {character_count}")
    return math.ceil(character_count * ESTIMATED_TOKENS_PER_TEXT_CHARACTER)


def estimate_image_input_tokens(width_pixels: int, height_pixels: int) -> int:
    """Estimate one image's vision INPUT tokens from its pixel dimensions.

    ``min(ceil(width/32) × ceil(height/32), 1536) × 2.46`` — the patch cost
    the model reads. The description-output allowance is a separate OUTPUT
    quantity (see ``estimate_image_tokens`` for the composed total). Image
    dimensions are always known at estimation time (the bytes are in hand),
    so non-positive dimensions are an estimation error, not a fallback case.
    """
    if width_pixels <= 0 or height_pixels <= 0:
        raise TokenEstimationError(
            f"Invalid image dimensions: {width_pixels}x{height_pixels}"
        )
    patch_count = math.ceil(width_pixels / VISION_PATCH_SIDE_PIXELS) * math.ceil(
        height_pixels / VISION_PATCH_SIDE_PIXELS
    )
    capped_patch_count = min(patch_count, VISION_MAXIMUM_PATCHES_PER_IMAGE)
    return int(capped_patch_count * VISION_TOKEN_MULTIPLIER)


def estimate_image_tokens(width_pixels: int, height_pixels: int) -> int:
    """Estimate one image's TOTAL vision tokens (input patches + description output).

    Composition of ``estimate_image_input_tokens`` plus the fixed
    description-output allowance, because every uploaded or attached image is
    run through a vision-description pass whose output feeds the conversation
    or the identity pipeline.
    """
    return (
        estimate_image_input_tokens(width_pixels, height_pixels)
        + ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )


def estimate_audio_diarization_tokens(duration_seconds: float) -> int:
    """Estimate diarization/transcription tokens for ``duration_seconds`` of audio.

    ``minutes × (1,600 audio-input + 200 transcript-output)`` = 1,800 tokens
    per minute. Callers that could not probe a confirmed audio/video item's
    duration substitute ``ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS`` before
    calling; a non-positive duration here is therefore an estimation error.
    """
    if duration_seconds <= 0:
        raise TokenEstimationError(f"Invalid audio duration: {duration_seconds}")
    duration_minutes = duration_seconds / 60.0
    return math.ceil(
        duration_minutes
        * (
            ESTIMATED_AUDIO_INPUT_TOKENS_PER_MINUTE
            + ESTIMATED_TRANSCRIPT_OUTPUT_TOKENS_PER_MINUTE
        )
    )


def estimate_audio_input_tokens(duration_seconds: float) -> int:
    """Estimate the audio-INPUT tokens for ``duration_seconds`` of encoded audio.

    ``minutes × 1,600`` — the encoded-audio tokens the model reads. The
    expected transcript is the OUTPUT side
    (``estimated_transcript_tokens_for_duration``);
    ``estimate_audio_diarization_tokens`` composes the two into a total.
    """
    if duration_seconds <= 0:
        raise TokenEstimationError(f"Invalid audio duration: {duration_seconds}")
    return math.ceil(
        (duration_seconds / 60.0) * ESTIMATED_AUDIO_INPUT_TOKENS_PER_MINUTE
    )


def estimated_transcript_tokens_for_duration(duration_seconds: float) -> int:
    """Return the expected transcript-output tokens for a stretch of audio.

    The transcript is what the identity-analysis passes re-read, so this is
    the content-token input to ``estimate_analysis_tokens`` for audio/video.
    """
    if duration_seconds <= 0:
        raise TokenEstimationError(f"Invalid audio duration: {duration_seconds}")
    return math.ceil(
        (duration_seconds / 60.0) * ESTIMATED_TRANSCRIPT_OUTPUT_TOKENS_PER_MINUTE
    )


def estimate_analysis_tokens(
    extracted_content_tokens: int, analysis_passes: int
) -> int:
    """Estimate the identity-analysis add-on for one item's extracted content.

    Each analysis pass (classification, identity-dimension extraction, question
    generation, ...) re-reads the extracted content, so the add-on is simply
    ``extracted content tokens × pass count``. A pass count of zero models a
    pipeline with analysis dropped and costs nothing — the modularity hook.
    """
    if extracted_content_tokens < 0:
        raise TokenEstimationError(
            f"Negative extracted content tokens: {extracted_content_tokens}"
        )
    if analysis_passes < 0:
        raise TokenEstimationError(f"Negative analysis pass count: {analysis_passes}")
    return extracted_content_tokens * analysis_passes


def estimate_media_item_token_breakdown(
    kind: MediaItemKind,
    *,
    word_count: int | None = None,
    width_pixels: int | None = None,
    height_pixels: int | None = None,
    duration_seconds: float | None = None,
    include_analysis: bool,
    analysis_passes: int,
) -> TokenEstimateBreakdown:
    """Estimate one media item's model tokens, split into input versus output.

    The modular composite behind the upload endpoint's per-item estimates:

    * ``text``  — requires ``word_count`` (of the ALREADY-extracted text);
      the text itself is model INPUT (word-ratio token count), and the
      analysis add-on re-reads those same tokens as further input.
    * ``image`` — requires ``width_pixels``/``height_pixels``; the vision
      patches are INPUT, the expected description is OUTPUT, and the analysis
      add-on re-reads the description as further input.
    * ``audio``/``video`` — requires ``duration_seconds`` (video is processed
      as its audio track); the encoded audio is INPUT, the expected
      transcript is OUTPUT, and the analysis add-on re-reads the transcript
      as further input.

    Analysis passes are always INPUT tokens: each pass re-reads the extracted
    content, so the add-on counts toward the gated input estimate.
    ``include_analysis`` reflects whether the pipeline will actually analyze
    this item (reference-image/reference-audio uploads skip identity
    analysis); when the analysis stage is dropped for an item, the estimate
    automatically drops with it.
    """
    if kind == "text":
        if word_count is None:
            raise TokenEstimationError("Text estimation requires a word count.")
        extraction_input_tokens = estimate_text_tokens_from_words(word_count)
        extraction_output_tokens = 0
        extracted_content_tokens = extraction_input_tokens
    elif kind == "image":
        if width_pixels is None or height_pixels is None:
            raise TokenEstimationError("Image estimation requires pixel dimensions.")
        extraction_input_tokens = estimate_image_input_tokens(
            width_pixels, height_pixels
        )
        extraction_output_tokens = ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
        extracted_content_tokens = ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    elif kind in ("audio", "video"):
        if duration_seconds is None:
            raise TokenEstimationError(
                f"{kind.capitalize()} estimation requires a duration."
            )
        extraction_input_tokens = estimate_audio_input_tokens(duration_seconds)
        extraction_output_tokens = estimated_transcript_tokens_for_duration(
            duration_seconds
        )
        extracted_content_tokens = extraction_output_tokens
    else:
        raise TokenEstimationError(f"Unknown media item kind: {kind!r}")

    analysis_input_tokens = 0
    if include_analysis:
        analysis_input_tokens = estimate_analysis_tokens(
            extracted_content_tokens, analysis_passes
        )
    return TokenEstimateBreakdown(
        input_tokens=extraction_input_tokens + analysis_input_tokens,
        output_tokens=extraction_output_tokens,
    )


def estimate_media_item_tokens(
    kind: MediaItemKind,
    *,
    word_count: int | None = None,
    width_pixels: int | None = None,
    height_pixels: int | None = None,
    duration_seconds: float | None = None,
    include_analysis: bool,
    analysis_passes: int,
) -> int:
    """Estimate one media item's TOTAL model tokens: extraction + optional analysis.

    Thin total-only view over ``estimate_media_item_token_breakdown`` for
    callers that do not need the input/output split.
    """
    return estimate_media_item_token_breakdown(
        kind,
        word_count=word_count,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        duration_seconds=duration_seconds,
        include_analysis=include_analysis,
        analysis_passes=analysis_passes,
    ).total_tokens


def estimate_message_request_token_breakdown(
    message_word_count: int,
    image_dimensions: Sequence[tuple[int, int]],
    system_prompt_tokens: int,
    tool_schema_tokens: int,
    expected_output_tokens: int,
) -> TokenEstimateBreakdown:
    """Estimate one message turn's tokens, split into input versus output.

    ``system_prompt_tokens`` is the MEASURED word-ratio token estimate of the
    actual, fully built system prompt for this (user, avatar) pair — the
    system prompt is built (or read from the estimate cache populated when
    the prompt was last built) BEFORE this estimate runs, so the estimate
    reflects real prompt size rather than a guessed constant.
    ``tool_schema_tokens`` is the MEASURED estimate of every tool schema the
    deep agent binds (see ``tool_schema_estimate_cache.py``) — the provider
    serializes the bound tool definitions into the prompt of the initial
    model call and bills them as input tokens, so they belong in the gated
    input estimate; there is NO fixed or guessed input overhead. Variable
    user content is the word-ratio estimate of the message plus any
    attached-file text, and each attached image adds its vision-input
    patches; every attached image's expected description joins the reply
    budget on the OUTPUT side.

    The allotment gate consumes ``input_tokens`` (total input may not exceed
    the remaining allotment); ``output_tokens`` is returned to the caller and
    may overshoot the allotment exactly once, because recorded actual totals
    then block the next request.
    """
    if (
        system_prompt_tokens < 0
        or tool_schema_tokens < 0
        or expected_output_tokens < 0
    ):
        raise TokenEstimationError(
            "Negative prompt/output token constants are not valid."
        )
    input_tokens = (
        system_prompt_tokens
        + tool_schema_tokens
        + estimate_text_tokens_from_words(message_word_count)
    )
    output_tokens = expected_output_tokens
    for width_pixels, height_pixels in image_dimensions:
        input_tokens += estimate_image_input_tokens(width_pixels, height_pixels)
        output_tokens += ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    return TokenEstimateBreakdown(
        input_tokens=input_tokens, output_tokens=output_tokens
    )


def estimate_message_request_tokens(
    message_word_count: int,
    image_dimensions: Sequence[tuple[int, int]],
    static_prompt_tokens: int,
    expected_output_tokens: int,
) -> int:
    """Estimate one message turn's TOTAL tokens before the model is called.

    Thin total-only view over ``estimate_message_request_token_breakdown``
    with ``static_prompt_tokens`` standing in for the whole input-side prompt
    overhead (measured system prompt plus measured tool schemas), for callers
    that do not need the input/output split.
    """
    return estimate_message_request_token_breakdown(
        message_word_count,
        image_dimensions,
        system_prompt_tokens=static_prompt_tokens,
        tool_schema_tokens=0,
        expected_output_tokens=expected_output_tokens,
    ).total_tokens
