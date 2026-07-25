"""Unit tests for manual pre-request token estimation and the admin bypass.

Covers the manual estimation formulas (word ratio, vision patch math, audio
diarization minutes, the modular analysis add-on), the message-request
composition (static prompt constant + variable words + expected output +
images), fail-closed behavior on invalid inputs, the estimate-aware allotment
block decision, and the admin testing-account metering bypass.
"""

import json
import math

import pytest

from src.anubis.utils.billing.estimation import (
    ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS,
    ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS,
    VISION_MAXIMUM_PATCHES_PER_IMAGE,
    VISION_TOKEN_MULTIPLIER,
    TokenEstimateBreakdown,
    TokenEstimationError,
    count_words,
    estimate_analysis_tokens,
    estimate_audio_diarization_tokens,
    estimate_audio_input_tokens,
    estimate_image_input_tokens,
    estimate_image_tokens,
    estimate_media_item_token_breakdown,
    estimate_media_item_tokens,
    estimate_message_request_token_breakdown,
    estimate_message_request_tokens,
    estimate_text_tokens_from_characters,
    estimate_text_tokens_from_words,
    estimated_transcript_tokens_for_duration,
)
from src.anubis.utils.billing.system_prompt_estimate_cache import (
    fetch_system_prompt_token_estimate,
    invalidate_system_prompt_token_estimate,
    record_system_prompt_token_estimate,
)
from src.anubis.utils.billing.gating import (
    exhausted_allotment_block_reason,
    MeteringBypass,
    is_admin_metering_bypass,
    is_dev_metered_enforcement_bypass,
    parse_metering_bypass_identifiers,
    resolve_metering_bypass,
)
from src.anubis.utils.billing.tiers import (
    SubscriptionTier,
    UsageMeter,
    tier_allotment_for_meter,
)

# ---------------------------------------------------------------------------
# Word-ratio and character text estimates
# ---------------------------------------------------------------------------


def test_word_ratio_is_four_thirds_tokens_per_word():
    assert estimate_text_tokens_from_words(300) == 400
    assert estimate_text_tokens_from_words(0) == 0
    # ceil: 1 word -> ceil(4/3) = 2 tokens
    assert estimate_text_tokens_from_words(1) == 2


def test_character_fallback_is_four_characters_per_token():
    assert estimate_text_tokens_from_characters(4000) == 1000
    assert estimate_text_tokens_from_characters(0) == 0


def test_negative_text_inputs_raise():
    with pytest.raises(TokenEstimationError):
        estimate_text_tokens_from_words(-1)
    with pytest.raises(TokenEstimationError):
        estimate_text_tokens_from_characters(-1)


def test_count_words_splits_on_whitespace():
    assert count_words("hello there  world\n new line") == 5
    assert count_words("") == 0
    assert count_words(None) == 0


# ---------------------------------------------------------------------------
# Vision patch math
# ---------------------------------------------------------------------------


def test_small_image_patch_math():
    # 64x64 -> ceil(64/32)^2 = 4 patches -> int(4 * 2.46) input tokens.
    assert estimate_image_tokens(64, 64) == (
        int(4 * VISION_TOKEN_MULTIPLIER) + ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )
    # 33x33 rounds each side up to 2 patches.
    assert estimate_image_tokens(33, 33) == (
        int(4 * VISION_TOKEN_MULTIPLIER) + ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )


def test_huge_image_hits_the_patch_cap():
    assert estimate_image_tokens(100_000, 100_000) == (
        int(VISION_MAXIMUM_PATCHES_PER_IMAGE * VISION_TOKEN_MULTIPLIER)
        + ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )


@pytest.mark.parametrize("width, height", [(0, 10), (10, 0), (-1, 5)])
def test_invalid_image_dimensions_raise(width, height):
    # Dimensions are always knowable from real image bytes, so bad values are
    # an estimation error (fail-closed), never a fallback.
    with pytest.raises(TokenEstimationError):
        estimate_image_tokens(width, height)


# ---------------------------------------------------------------------------
# Audio / video diarization estimates
# ---------------------------------------------------------------------------


def test_audio_minute_formula():
    # minutes * (1600 audio-input + 200 transcript-output) = 1800 per minute.
    assert estimate_audio_diarization_tokens(60) == 1800
    assert estimate_audio_diarization_tokens(90) == 2700
    assert estimated_transcript_tokens_for_duration(60) == 200


def test_audio_fallback_duration_is_ten_minutes():
    assert (
        estimate_audio_diarization_tokens(ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS)
        == 18_000
    )


def test_non_positive_durations_raise():
    with pytest.raises(TokenEstimationError):
        estimate_audio_diarization_tokens(0)
    with pytest.raises(TokenEstimationError):
        estimated_transcript_tokens_for_duration(-5)


# ---------------------------------------------------------------------------
# Modular analysis add-on
# ---------------------------------------------------------------------------


def test_analysis_add_on_multiplies_content_tokens_by_passes():
    assert estimate_analysis_tokens(200, 2) == 400
    assert estimate_analysis_tokens(200, 0) == 0
    with pytest.raises(TokenEstimationError):
        estimate_analysis_tokens(-1, 2)
    with pytest.raises(TokenEstimationError):
        estimate_analysis_tokens(10, -1)


def test_audio_item_with_and_without_analysis():
    # One minute of audio: 1800 extraction; analysis re-reads the 200-token
    # transcript per pass -> 2 passes adds 400 => 2200 total.
    without_analysis = estimate_media_item_tokens(
        "audio", duration_seconds=60, include_analysis=False, analysis_passes=2
    )
    with_analysis = estimate_media_item_tokens(
        "audio", duration_seconds=60, include_analysis=True, analysis_passes=2
    )
    assert without_analysis == 1800
    assert with_analysis == 2200
    # Zero passes models a pipeline with analysis dropped: extraction only.
    assert (
        estimate_media_item_tokens(
            "audio", duration_seconds=60, include_analysis=True, analysis_passes=0
        )
        == 1800
    )


def test_video_item_delegates_to_the_audio_formula():
    for duration_seconds in (30, 60, 617):
        assert estimate_media_item_tokens(
            "video",
            duration_seconds=duration_seconds,
            include_analysis=True,
            analysis_passes=2,
        ) == estimate_media_item_tokens(
            "audio",
            duration_seconds=duration_seconds,
            include_analysis=True,
            analysis_passes=2,
        )


def test_text_item_analysis_rereads_the_extracted_words():
    # 300 words -> 400 extraction tokens; 2 analysis passes re-read them.
    assert (
        estimate_media_item_tokens(
            "text", word_count=300, include_analysis=True, analysis_passes=2
        )
        == 400 + 800
    )
    assert (
        estimate_media_item_tokens(
            "text", word_count=300, include_analysis=False, analysis_passes=2
        )
        == 400
    )


def test_image_item_analysis_rereads_the_description_output():
    base = estimate_image_tokens(64, 64)
    assert estimate_media_item_tokens(
        "image",
        width_pixels=64,
        height_pixels=64,
        include_analysis=True,
        analysis_passes=2,
    ) == base + 2 * ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS


def test_media_item_kind_requires_its_inputs():
    with pytest.raises(TokenEstimationError):
        estimate_media_item_tokens("text", include_analysis=True, analysis_passes=1)
    with pytest.raises(TokenEstimationError):
        estimate_media_item_tokens("image", include_analysis=True, analysis_passes=1)
    with pytest.raises(TokenEstimationError):
        estimate_media_item_tokens("audio", include_analysis=True, analysis_passes=1)
    with pytest.raises(TokenEstimationError):
        estimate_media_item_tokens(
            "unknown",  # type: ignore[arg-type]
            include_analysis=True,
            analysis_passes=1,
        )


# ---------------------------------------------------------------------------
# Message-request composition (manual: constants + word ratio + images)
# ---------------------------------------------------------------------------


def test_message_estimate_composition_is_exact():
    static_prompt_tokens = 4096
    expected_output_tokens = 512
    message_word_count = 3
    total = estimate_message_request_tokens(
        message_word_count,
        [(64, 64)],
        static_prompt_tokens,
        expected_output_tokens,
    )
    assert total == (
        static_prompt_tokens
        + math.ceil(message_word_count * 4 / 3)
        + expected_output_tokens
        + estimate_image_tokens(64, 64)
    )


def test_message_estimate_without_images_or_text():
    assert estimate_message_request_tokens(0, [], 4096, 512) == 4608


def test_message_estimate_rejects_negative_constants():
    with pytest.raises(TokenEstimationError):
        estimate_message_request_tokens(10, [], -1, 512)


# ---------------------------------------------------------------------------
# Estimate-aware allotment blocking
# ---------------------------------------------------------------------------


def test_estimate_crossing_the_allotment_blocks_and_names_the_estimate():
    allotment = tier_allotment_for_meter(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
    )
    usage_just_under = allotment.monthly_allotment - 10
    reason = exhausted_allotment_block_reason(
        SubscriptionTier.PRO,
        UsageMeter.MESSAGING_TOKENS,
        usage_just_under,
        False,
        estimated_request_tokens=20,
    )
    assert reason is not None
    assert "estimated at 20" in reason


def test_estimate_crossing_with_pay_per_use_is_allowed():
    allotment = tier_allotment_for_meter(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
    )
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.PRO,
            UsageMeter.MESSAGING_TOKENS,
            allotment.monthly_allotment - 10,
            True,
            estimated_request_tokens=20,
        )
        is None
    )


def test_estimate_fitting_the_remaining_allotment_is_allowed():
    allotment = tier_allotment_for_meter(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
    )
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.PRO,
            UsageMeter.MESSAGING_TOKENS,
            allotment.monthly_allotment - 30,
            False,
            estimated_request_tokens=20,
        )
        is None
    )


def test_zero_estimate_reproduces_the_plain_usage_matrix():
    # Regression guard: with no estimate the decision is exactly the original
    # usage-only check for every tier and pay-per-use setting.
    for tier in SubscriptionTier:
        allotment = tier_allotment_for_meter(tier, UsageMeter.MESSAGING_TOKENS)
        assert (
            exhausted_allotment_block_reason(
                tier,
                UsageMeter.MESSAGING_TOKENS,
                allotment.monthly_allotment - 1,
                False,
                estimated_request_tokens=0,
            )
            is None
        )
        assert (
            exhausted_allotment_block_reason(
                tier,
                UsageMeter.MESSAGING_TOKENS,
                allotment.monthly_allotment,
                False,
                estimated_request_tokens=0,
            )
            is not None
        )


# ---------------------------------------------------------------------------
# Input/output breakdowns (the allotment gate consumes INPUT only)
# ---------------------------------------------------------------------------


def test_breakdown_total_is_input_plus_output():
    breakdown = TokenEstimateBreakdown(input_tokens=100, output_tokens=25)
    assert breakdown.total_tokens == 125


def test_message_breakdown_splits_input_and_output():
    system_prompt_tokens = 15_000
    tool_schema_tokens = 13_472
    expected_output_tokens = 512
    message_word_count = 3
    breakdown = estimate_message_request_token_breakdown(
        message_word_count,
        [(64, 64)],
        system_prompt_tokens=system_prompt_tokens,
        tool_schema_tokens=tool_schema_tokens,
        expected_output_tokens=expected_output_tokens,
    )
    assert breakdown.input_tokens == (
        system_prompt_tokens
        + tool_schema_tokens
        + math.ceil(message_word_count * 4 / 3)
        + estimate_image_input_tokens(64, 64)
    )
    assert breakdown.output_tokens == (
        expected_output_tokens + ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )


def test_usage_estimate_client_shape_exposes_input_tokens_only():
    """Messaging SSE/JSON reports a single pre-call ``input_tokens`` field.

    Estimated totals and output budgets stay server-side for gating/rate limits;
    clients must not receive ``estimated_request_tokens`` /
    ``estimated_input_tokens`` / ``estimated_output_tokens``.
    """
    breakdown = TokenEstimateBreakdown(input_tokens=9_541, output_tokens=512)
    usage_estimate_event = {
        "type": "usage_estimate",
        "input_tokens": breakdown.input_tokens,
    }
    assert usage_estimate_event == {
        "type": "usage_estimate",
        "input_tokens": 9_541,
    }
    assert "estimated_request_tokens" not in usage_estimate_event
    assert "estimated_input_tokens" not in usage_estimate_event
    assert "estimated_output_tokens" not in usage_estimate_event
    # Gate / rate-limit still have access to the split via the breakdown.
    assert breakdown.total_tokens == 9_541 + 512


def test_message_total_delegate_matches_the_breakdown():
    # The total-only helper is a thin view over the breakdown.
    assert estimate_message_request_tokens(3, [(64, 64)], 4096, 512) == (
        estimate_message_request_token_breakdown(
            3,
            [(64, 64)],
            system_prompt_tokens=4096,
            tool_schema_tokens=0,
            expected_output_tokens=512,
        ).total_tokens
    )


def test_audio_breakdown_reads_audio_and_writes_transcript():
    # One minute: 1,600 audio-input read; 200 transcript written; 2 analysis
    # passes re-read the transcript (2 × 200 = 400 further INPUT).
    breakdown = estimate_media_item_token_breakdown(
        "audio", duration_seconds=60, include_analysis=True, analysis_passes=2
    )
    assert breakdown.input_tokens == 1600 + 400
    assert breakdown.output_tokens == 200
    assert breakdown.total_tokens == 2200


def test_audio_input_side_formula():
    assert estimate_audio_input_tokens(60) == 1600
    with pytest.raises(TokenEstimationError):
        estimate_audio_input_tokens(0)


def test_image_breakdown_reads_patches_and_writes_description():
    breakdown = estimate_media_item_token_breakdown(
        "image",
        width_pixels=64,
        height_pixels=64,
        include_analysis=True,
        analysis_passes=2,
    )
    assert breakdown.input_tokens == (
        estimate_image_input_tokens(64, 64)
        + 2 * ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS
    )
    assert breakdown.output_tokens == ESTIMATED_IMAGE_DESCRIPTION_OUTPUT_TOKENS


def test_text_breakdown_is_all_input():
    # Text extraction and analysis are both reads: analysis passes are
    # quantified as INPUT tokens.
    breakdown = estimate_media_item_token_breakdown(
        "text", word_count=300, include_analysis=True, analysis_passes=2
    )
    assert breakdown.input_tokens == 400 + 800
    assert breakdown.output_tokens == 0


def test_media_total_delegate_matches_the_breakdown():
    assert estimate_media_item_tokens(
        "video", duration_seconds=90, include_analysis=True, analysis_passes=2
    ) == (
        estimate_media_item_token_breakdown(
            "video", duration_seconds=90, include_analysis=True, analysis_passes=2
        ).total_tokens
    )


# ---------------------------------------------------------------------------
# Output overshoot is allowed exactly once (gate consumes input only)
# ---------------------------------------------------------------------------


def test_output_overshoot_is_allowed_exactly_once():
    """Total input may not exceed the allotment; output may cross it ONCE.

    Sequence: a request whose INPUT estimate fits under the remaining
    allotment is allowed even though the eventual output pushes recorded
    total usage past the allotment; the next request is then blocked because
    recorded usage (which counts total tokens) is at/past the allotment.
    """
    allotment = tier_allotment_for_meter(
        SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
    )
    usage_before_final_request = allotment.monthly_allotment - 100

    # Request N: estimated INPUT of 50 fits under the remaining 100 → allowed.
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            usage_before_final_request,
            False,
            estimated_request_tokens=50,
        )
        is None
    )

    # The model then writes a large reply: recorded TOTAL usage (input +
    # output) lands past the allotment. That overshoot was allowed.
    recorded_usage_after_overshoot = allotment.monthly_allotment + 5_000

    # Request N+1: blocked — the single sanctioned overshoot already happened.
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            recorded_usage_after_overshoot,
            False,
            estimated_request_tokens=0,
        )
        is not None
    )


def test_input_estimate_exceeding_the_allotment_is_refused_up_front():
    # Total input may not exceed the total allotment: even at zero usage, an
    # input estimate at/over the whole allotment is refused without
    # pay-per-use.
    allotment = tier_allotment_for_meter(
        SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
    )
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            0,
            False,
            estimated_request_tokens=allotment.monthly_allotment,
        )
        is not None
    )


# ---------------------------------------------------------------------------
# System-prompt token-estimate cache (measured when the prompt is built)
# ---------------------------------------------------------------------------


def test_system_prompt_estimate_round_trips_through_the_cache():
    prompt_text = "word " * 300  # 300 words -> 400 tokens at 4/3 per word
    recorded = record_system_prompt_token_estimate(
        "user-a", "assistant-a", prompt_text
    )
    assert recorded == 400
    assert fetch_system_prompt_token_estimate("user-a", "assistant-a") == 400
    invalidate_system_prompt_token_estimate("user-a", "assistant-a")
    assert fetch_system_prompt_token_estimate("user-a", "assistant-a") is None


def test_system_prompt_estimate_is_scoped_per_user_and_assistant():
    record_system_prompt_token_estimate("user-b", "assistant-b", "word " * 75)
    assert fetch_system_prompt_token_estimate("user-b", "assistant-other") is None
    assert fetch_system_prompt_token_estimate("user-other", "assistant-b") is None
    invalidate_system_prompt_token_estimate("user-b", "assistant-b")


def test_stale_system_prompt_estimate_is_a_miss():
    record_system_prompt_token_estimate("user-c", "assistant-c", "word " * 30)
    # A zero maximum age makes every entry stale immediately: the caller must
    # rebuild the prompt rather than trust an outdated measurement.
    assert (
        fetch_system_prompt_token_estimate(
            "user-c", "assistant-c", max_age_seconds=0.0
        )
        is None
    )
    invalidate_system_prompt_token_estimate("user-c", "assistant-c")


def test_rerecording_overwrites_the_previous_measurement():
    record_system_prompt_token_estimate("user-d", "assistant-d", "word " * 75)
    record_system_prompt_token_estimate("user-d", "assistant-d", "word " * 150)
    assert fetch_system_prompt_token_estimate("user-d", "assistant-d") == 200
    invalidate_system_prompt_token_estimate("user-d", "assistant-d")


# ---------------------------------------------------------------------------
# Deep-agent tool-schema token-estimate cache (measured from the bound tools)
# ---------------------------------------------------------------------------


def test_tool_schema_measurement_matches_serialized_characters_over_four():
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from src.anubis.utils.billing.tool_schema_estimate_cache import (
        estimate_tool_schema_tokens_for_tools,
    )

    @tool
    def example_tool(example_argument: str) -> str:
        """An example tool description that costs input tokens on every call."""
        return example_argument

    serialized_characters = len(json.dumps(convert_to_openai_tool(example_tool)))
    assert estimate_tool_schema_tokens_for_tools([example_tool]) == math.ceil(
        serialized_characters * 0.25
    )
    # Two copies of the same schema cost exactly twice the characters.
    assert estimate_tool_schema_tokens_for_tools(
        [example_tool, example_tool]
    ) == math.ceil(serialized_characters * 2 * 0.25)


def test_tool_schema_estimate_is_measured_once_then_cached(monkeypatch):
    from src.anubis.utils.billing import tool_schema_estimate_cache

    tool_schema_estimate_cache.invalidate_deep_agent_tool_schema_token_estimate()
    measurement_call_count = 0

    def _fake_measurement() -> int:
        nonlocal measurement_call_count
        measurement_call_count += 1
        return 13_472

    monkeypatch.setattr(
        tool_schema_estimate_cache,
        "measure_deep_agent_tool_schema_token_estimate",
        _fake_measurement,
    )
    first_fetch = (
        tool_schema_estimate_cache
        .fetch_or_measure_deep_agent_tool_schema_token_estimate()
    )
    second_fetch = (
        tool_schema_estimate_cache
        .fetch_or_measure_deep_agent_tool_schema_token_estimate()
    )
    assert first_fetch == second_fetch == 13_472
    assert measurement_call_count == 1

    tool_schema_estimate_cache.invalidate_deep_agent_tool_schema_token_estimate()
    assert (
        tool_schema_estimate_cache
        .fetch_or_measure_deep_agent_tool_schema_token_estimate()
        == 13_472
    )
    assert measurement_call_count == 2
    tool_schema_estimate_cache.invalidate_deep_agent_tool_schema_token_estimate()


# ---------------------------------------------------------------------------
# Admin testing-account metering bypass
# ---------------------------------------------------------------------------

_ADMIN_USER_ID = "69e5e49980b783d7dff3012b"


def test_admin_bypass_matches_the_configured_id():
    assert is_admin_metering_bypass({"user_id": _ADMIN_USER_ID}, _ADMIN_USER_ID)


def test_admin_bypass_matches_the_identities_id():
    anonymous_shaped_admin = {"identities": [{"user_id": _ADMIN_USER_ID}]}
    assert is_admin_metering_bypass(anonymous_shaped_admin, _ADMIN_USER_ID)


def test_non_admin_users_never_bypass():
    assert not is_admin_metering_bypass({"user_id": "someone-else"}, _ADMIN_USER_ID)
    anonymous_user = {"identities": [{"user_id": "hashed-ip-identifier"}]}
    assert not is_admin_metering_bypass(anonymous_user, _ADMIN_USER_ID)


def test_missing_admin_id_never_bypasses():
    assert not is_admin_metering_bypass({"user_id": _ADMIN_USER_ID}, None)
    assert not is_admin_metering_bypass({"user_id": _ADMIN_USER_ID}, "")
    assert not is_admin_metering_bypass(None, _ADMIN_USER_ID)


# ---------------------------------------------------------------------------
# Configured bypass identifiers (anonymous hashed-IP testing accounts)
# ---------------------------------------------------------------------------

# sha256("172.18.0.1") — the docker bridge gateway the dev container sees.
_HASHED_IP_DEV_GATEWAY = (
    "245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e"
)
_HASHED_IP_VPN_SIMULATED = (
    "2a1201bb6c0061be63fc4ce58a048136fa91d3afea9e21f62ae7988a20cc09f1"
)
_BYPASS_IDENTIFIERS = f"{_HASHED_IP_DEV_GATEWAY},{_HASHED_IP_VPN_SIMULATED}"


def _anonymous_user(hashed_ip: str) -> dict:
    """Shape an anonymous user the way the auth layer stamps one.

    ``get_anonymous_user_with_anonymous_api_key`` puts the hashed IP in
    ``identities[0].user_id`` and leaves no top-level ``user_id``, which is the
    only place ``resolve_metering_user_id`` can find an anonymous identity.
    """
    return {"identities": [{"user_id": hashed_ip}]}


def test_listed_anonymous_hashed_ip_bypasses_metering():
    assert is_admin_metering_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY), None, _BYPASS_IDENTIFIERS
    )
    assert is_admin_metering_bypass(
        _anonymous_user(_HASHED_IP_VPN_SIMULATED), _ADMIN_USER_ID, _BYPASS_IDENTIFIERS
    )


def test_unlisted_anonymous_hashed_ip_stays_metered():
    unlisted = "72aefc13eebd36bf5ec1cbfa1f2e930117a62e07f600dc618c18725f3d52be15"
    assert not is_admin_metering_bypass(
        _anonymous_user(unlisted), _ADMIN_USER_ID, _BYPASS_IDENTIFIERS
    )


def test_admin_user_id_still_bypasses_alongside_the_identifier_list():
    assert is_admin_metering_bypass(
        {"user_id": _ADMIN_USER_ID}, _ADMIN_USER_ID, _BYPASS_IDENTIFIERS
    )


def test_empty_identifier_list_leaves_everyone_metered():
    # The production posture: no ADMIN_USER_ID and no configured identifiers
    # must not degrade into "the empty string matches", which would exempt a
    # user whose metering id could not be resolved.
    for empty_configuration in (None, "", "  ", ",,", "\n"):
        assert not is_admin_metering_bypass(
            _anonymous_user(_HASHED_IP_DEV_GATEWAY), None, empty_configuration
        )
        assert not is_admin_metering_bypass({}, None, empty_configuration)


def test_identifier_parsing_tolerates_whitespace_case_and_comments():
    assert is_admin_metering_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY),
        None,
        f"  {_HASHED_IP_DEV_GATEWAY.upper()} , # a comment\n",
    )
    assert parse_metering_bypass_identifiers(
        f"{_HASHED_IP_DEV_GATEWAY}\n{_HASHED_IP_VPN_SIMULATED},"
    ) == frozenset({_HASHED_IP_DEV_GATEWAY, _HASHED_IP_VPN_SIMULATED})
    assert parse_metering_bypass_identifiers(None) == frozenset()


def test_identifier_list_accepts_an_already_split_iterable():
    assert is_admin_metering_bypass(
        _anonymous_user(_HASHED_IP_VPN_SIMULATED),
        None,
        [_HASHED_IP_DEV_GATEWAY, _HASHED_IP_VPN_SIMULATED],
    )


# ---------------------------------------------------------------------------
# Development enforcement-only bypass (still metered)
# ---------------------------------------------------------------------------


class _BypassContext:
    """The four GlobalContext fields resolve_metering_bypass reads."""

    def __init__(
        self,
        dev="TRUE",
        admin_user_id=None,
        admin_metering_bypass_identifiers=None,
        dev_metered_enforcement_bypass_identifiers=None,
    ):
        self.dev = dev
        self.admin_user_id = admin_user_id
        self.admin_metering_bypass_identifiers = admin_metering_bypass_identifiers
        self.dev_metered_enforcement_bypass_identifiers = (
            dev_metered_enforcement_bypass_identifiers
        )


def test_dev_metered_bypass_skips_enforcement_but_keeps_metering():
    # The whole point of the mode: the anonymous tester keeps messaging past the
    # free-tier allotment, and every turn still reaches Stripe and api_metrics
    # so the portal, /verify_subscription_status and the SSE frames agree.
    bypass = resolve_metering_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY),
        _BypassContext(
            dev_metered_enforcement_bypass_identifiers=_BYPASS_IDENTIFIERS
        ),
    )
    assert bypass.skips_enforcement
    assert not bypass.skips_metering_writes
    assert bypass.usage_response_fields() == {"admin_enforcement_bypass": True}


def test_dev_metered_bypass_is_inert_outside_development():
    # A hashed IP left in a copied environment file must not become an
    # unenforced production requester.
    for production_dev_flag in ("FALSE", "", None, "false-ish"):
        bypass = resolve_metering_bypass(
            _anonymous_user(_HASHED_IP_DEV_GATEWAY),
            _BypassContext(
                dev=production_dev_flag,
                dev_metered_enforcement_bypass_identifiers=_BYPASS_IDENTIFIERS,
            ),
        )
        assert bypass == MeteringBypass()
        assert bypass.usage_response_fields() == {}


def test_dev_metered_bypass_honors_the_dev_flag_case_and_whitespace():
    assert is_dev_metered_enforcement_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY), _BYPASS_IDENTIFIERS, " true "
    )
    assert not is_dev_metered_enforcement_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY), None, "TRUE"
    )
    assert not is_dev_metered_enforcement_bypass(
        _anonymous_user("an-unlisted-hashed-ip"), _BYPASS_IDENTIFIERS, "TRUE"
    )
    assert not is_dev_metered_enforcement_bypass(None, _BYPASS_IDENTIFIERS, "TRUE")


def test_full_admin_bypass_wins_over_the_dev_metered_list():
    # Listed on both: the broader treatment applies rather than one that depends
    # on which list is consulted first.
    bypass = resolve_metering_bypass(
        _anonymous_user(_HASHED_IP_DEV_GATEWAY),
        _BypassContext(
            admin_metering_bypass_identifiers=_BYPASS_IDENTIFIERS,
            dev_metered_enforcement_bypass_identifiers=_BYPASS_IDENTIFIERS,
        ),
    )
    assert bypass.skips_enforcement
    assert bypass.skips_metering_writes
    assert bypass.usage_response_fields() == {"admin_metering_bypass": True}


def test_ordinary_requester_is_enforced_and_metered():
    bypass = resolve_metering_bypass(
        _anonymous_user("an-unlisted-hashed-ip"),
        _BypassContext(
            admin_user_id=_ADMIN_USER_ID,
            dev_metered_enforcement_bypass_identifiers=_BYPASS_IDENTIFIERS,
        ),
    )
    assert bypass == MeteringBypass(
        skips_enforcement=False, skips_metering_writes=False
    )
    assert bypass.usage_response_fields() == {}
