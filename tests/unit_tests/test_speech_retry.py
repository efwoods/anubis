"""Unit tests for durable retry of OpenAI speech (transcription/diarization) calls.

Covers ``_speech_call_with_retry`` — exponential backoff on transient errors,
fail-fast on permanent ones (insufficient_quota) — and that the speech client
delegates retries to the wrapper (``max_retries=0``).

All sleeps are stubbed so the tests stay instant and offline.
"""

from types import SimpleNamespace

import openai
import pytest

import src.anubis.utils.utility as util
from src.anubis.utils.utility import (
    _openai_client_for_speech,
    _speech_call_with_retry,
)


def _ctx(max_retries: int = 3, base: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        openai_speech_max_retries=max_retries,
        openai_speech_retry_base_seconds=base,
    )


def _rate_limit(code: str) -> openai.RateLimitError:
    # Build the error without the SDK's httpx-response __init__ requirements; the
    # wrapper only inspects the type and ``.code``.
    exc = openai.RateLimitError.__new__(openai.RateLimitError)
    exc.code = code
    return exc


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(util.time, "sleep", lambda d: sleeps.append(d))
    return sleeps


def test_retries_transient_rate_limit_then_succeeds(no_sleep):
    calls = {"n": 0}

    def make_call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limit("rate_limit_exceeded")
        return "ok"

    result = _speech_call_with_retry(
        make_call, _ctx(max_retries=3, base=0.5), description="t"
    )
    assert result == "ok"
    assert calls["n"] == 3
    # Two backoff sleeps before the third (successful) attempt.
    assert len(no_sleep) == 2


def test_insufficient_quota_fails_fast(no_sleep):
    """A permanent quota error must surface on the FIRST try (no retries)."""
    calls = {"n": 0}

    def make_call():
        calls["n"] += 1
        raise _rate_limit("insufficient_quota")

    with pytest.raises(openai.RateLimitError):
        _speech_call_with_retry(make_call, _ctx(max_retries=5), description="t")
    assert calls["n"] == 1
    assert no_sleep == []


def test_transient_retries_exhausted_then_raises(no_sleep):
    calls = {"n": 0}

    def make_call():
        calls["n"] += 1
        raise _rate_limit("rate_limit_exceeded")

    with pytest.raises(openai.RateLimitError):
        _speech_call_with_retry(make_call, _ctx(max_retries=2), description="t")
    # 1 initial + 2 retries.
    assert calls["n"] == 3


def test_non_5xx_status_error_not_retried(no_sleep):
    exc = openai.APIStatusError.__new__(openai.APIStatusError)
    exc.status_code = 400
    calls = {"n": 0}

    def make_call():
        calls["n"] += 1
        raise exc

    with pytest.raises(openai.APIStatusError):
        _speech_call_with_retry(make_call, _ctx(max_retries=5), description="t")
    assert calls["n"] == 1  # 4xx is a caller bug, fail fast


def test_speech_client_disables_sdk_retries():
    """The client delegates retry control to _speech_call_with_retry."""
    ctx = SimpleNamespace(openai_api_key="sk-test", llm_provider_api_key=None)
    client = _openai_client_for_speech(ctx)
    assert client.max_retries == 0
