"""An upload's recorded cost and its billed usage match what really happened.

A media upload used to record no cost at all and to bill a pre-upload estimate:
the ``document_upload`` row was written before the graph ran, with no
``cost_usd``, and the meter event carried a guess. Everything the upload pays
for — diarizing or transcribing the audio, describing each image, classifying,
segmenting, rewriting, extracting, the psycho-analysis dimensions, moderation —
ran afterwards and was priced nowhere. On 2026-09-17 one account's uploads cost
about $9 at the vendors while ``api_metrics`` recorded $0.38, and diarization
alone was $5.79 of that.

These tests pin the three pieces that close the gap: the price table, the model
call recorder, and the per-job usage read the Stripe meter is reported from.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.anubis.utils.billing.metering import (
    media_processing_inference_type,
    price_model_token_usage,
    read_recorded_media_job_usage,
    resolve_model_token_prices,
)


@dataclass
class PricingContextStub:
    """The pricing fields the price table reads on ``GlobalContext``."""

    model: str = "gpt-5.6-luna"
    model_prompt_cost: float = 0.0000002
    model_completion_cost: float = 0.00000125
    model_cached_prompt_cost: float = 0.00000002
    model_cache_write_cost: float = 0.00000025
    classification_model: str = "gpt-5.4-nano"
    classification_model_prompt_cost: float = 0.0000002
    classification_model_completion_cost: float = 0.00000125
    classification_model_cached_prompt_cost: float = 0.00000002
    classification_model_cache_write_cost: float = 0.00000025
    image_model: str = "gpt-5-nano"
    image_model_prompt_cost: float = 0.0000001
    image_model_completion_cost: float = 0.00000125
    image_model_cached_prompt_cost: float = 0.00000001
    llama_model: str = "llama-3.2-11b"
    llama_model_prompt_cost: float = 0.00000027
    llama_model_completion_cost: float = 0.00000085


def test_a_dated_model_name_resolves_to_the_configured_prices() -> None:
    """A provider reports a dated name for a model configured without the date."""
    prompt, completion, cached, cache_write = resolve_model_token_prices(
        "gpt-5.4-nano-2026-03-17", PricingContextStub()
    )
    assert (prompt, completion, cached, cache_write) == (
        0.0000002,
        0.00000125,
        0.00000002,
        0.00000025,
    )


def test_an_unknown_model_falls_back_to_the_inference_model_prices() -> None:
    """An unrecognized model still costs something: a silent zero hides spend."""
    prompt, completion, _cached, _cache_write = resolve_model_token_prices(
        "some-unreleased-model", PricingContextStub()
    )
    assert (prompt, completion) == (0.0000002, 0.00000125)


def test_cached_and_written_prompt_tokens_are_priced_at_their_own_rates() -> None:
    """Prompt caching is not a rounding error.

    On 2026-09-17 cache writes were the largest single line of the inference
    model's invoice, so the uncached rate must apply only to the tokens that were
    neither read from nor written to the cache.
    """
    cost = price_model_token_usage(
        "gpt-5.6-luna",
        PricingContextStub(),
        prompt_tokens=1_000_000,
        completion_tokens=10_000,
        cached_prompt_tokens=400_000,
        cache_write_tokens=500_000,
    )
    assert cost == (
        100_000 * 0.0000002
        + 400_000 * 0.00000002
        + 500_000 * 0.00000025
        + 10_000 * 0.00000125
    )


def test_pricing_never_charges_the_uncached_rate_twice() -> None:
    """Cached tokens reported as the whole prompt leave nothing at full price."""
    assert (
        price_model_token_usage(
            "gpt-5.6-luna",
            PricingContextStub(),
            prompt_tokens=1_000,
            cached_prompt_tokens=1_000,
        )
        == 1_000 * 0.00000002
    )


def test_a_media_node_is_recorded_under_the_work_it_did() -> None:
    """The row says which stage of an upload spent the money."""
    assert media_processing_inference_type("analyze_schwartz_values") == (
        "psychological_analysis"
    )
    assert media_processing_inference_type("deep_judge") == "moderation"
    assert media_processing_inference_type("convert_media_list_to_text_document") == (
        "media_conversion"
    )
    # An unmapped node keeps its own name rather than vanishing into a catch-all.
    assert media_processing_inference_type("some_new_node") == "some_new_node"
    assert media_processing_inference_type(None) == "media_processing"


class _CursorStub:
    def __init__(self, row: Any) -> None:
        self._row = row
        self.executed: list[tuple[str, Any]] = []

    async def execute(self, statement: str, parameters: Any = None) -> None:
        self.executed.append((statement, parameters))

    async def fetchone(self) -> Any:
        return self._row

    async def __aenter__(self) -> "_CursorStub":
        return self

    async def __aexit__(self, *exception_details: Any) -> None:
        return None


class _ConnectionStub:
    def __init__(self, cursor: _CursorStub) -> None:
        self._cursor = cursor

    def cursor(self) -> _CursorStub:
        return self._cursor

    async def __aenter__(self) -> "_ConnectionStub":
        return self

    async def __aexit__(self, *exception_details: Any) -> None:
        return None


class _PoolStub:
    def __init__(self, row: Any) -> None:
        self.cursor_stub = _CursorStub(row)

    def connection(self) -> _ConnectionStub:
        return _ConnectionStub(self.cursor_stub)


@pytest.mark.asyncio
async def test_a_media_job_s_usage_is_summed_from_the_rows_its_calls_wrote() -> None:
    """What the upload really consumed is what the Stripe meter is reported from."""
    pool = _PoolStub((1_284_551, 42_110, 1_326_661, 5.7912, 631))
    usage = await read_recorded_media_job_usage(pool, ["job-a", "job-b"])
    assert usage == {
        "prompt_tokens": 1_284_551,
        "completion_tokens": 42_110,
        "total_tokens": 1_326_661,
        "cost_usd": 5.7912,
        "call_count": 631,
    }
    _statement, parameters = pool.cursor_stub.executed[0]
    assert parameters == (["job-a", "job-b"],)


@pytest.mark.asyncio
async def test_usage_for_no_jobs_reads_zero_without_touching_the_database() -> None:
    """A batch that expanded into nothing bills nothing, and asks nothing."""
    pool = _PoolStub((1, 1, 1, 1.0, 1))
    assert await read_recorded_media_job_usage(pool, []) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "call_count": 0,
    }
    assert pool.cursor_stub.executed == []


@pytest.mark.asyncio
async def test_an_unreadable_sum_reports_zero_rather_than_raising() -> None:
    """Accounting must never fail an upload that already succeeded."""

    class _FailingPool:
        def connection(self) -> Any:
            raise RuntimeError("the pool is closed")

    usage = await read_recorded_media_job_usage(_FailingPool(), ["job-a"])
    assert usage["total_tokens"] == 0
    assert usage["cost_usd"] == 0.0


@dataclass
class DiarizationPricingContextStub:
    """The diarization pricing fields ``_diarize_token_cost`` reads.

    ``gpt-4o-transcribe-diarize`` is billed at $2.50 per million audio input
    tokens and $10.00 per million text output tokens.
    """

    audio_diarization_price_per_million_tokens_input: float = 0.0000025
    audio_diarization_price_per_million_tokens_output: float = 0.00001


def test_diarization_is_priced_on_the_tokens_it_was_billed_for() -> None:
    """Diarizing the audio is the largest charge a media upload makes.

    The cost was always computed correctly here and then thrown away before it
    reached the database, which is why one account's $5.79 of diarization was
    recorded as $0.
    """
    from src.anubis.utils.utility import _diarize_token_cost

    usage = {"type": "tokens", "input_tokens": 1_197_200, "output_tokens": 271_100}
    assert _diarize_token_cost(usage, DiarizationPricingContextStub()) == pytest.approx(
        1_197_200 * 0.0000025 + 271_100 * 0.00001
    )


def test_diarization_without_reported_usage_costs_nothing_here() -> None:
    """A response with no token usage falls back to the caller's estimate."""
    from src.anubis.utils.utility import _diarize_token_cost

    assert _diarize_token_cost({}, DiarizationPricingContextStub()) == 0.0
