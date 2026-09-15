"""POST /transcribe must not call a refused vendor key 'out of funds'."""

from types import SimpleNamespace

import pytest

from src.api import webapp as webapp_module


class _Audio:
    filename = "utterance.webm"
    content_type = "audio/webm"

    async def read(self):
        return b"not-empty"


def _install_transcribe(monkeypatch, error):
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=SimpleNamespace(), pool=None),
    )

    async def failing_transcribe(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "src.anubis.utils.utility.transcribe_audio", failing_transcribe
    )


@pytest.mark.asyncio
async def test_transcribe_reports_a_refused_vendor_key_not_empty_funds(monkeypatch):
    _install_transcribe(
        monkeypatch,
        RuntimeError("Error code: 401 - invalid_api_key: Incorrect API key provided"),
    )
    response = await webapp_module.transcribe_recording(
        assistant_id="assistant-1",
        audio=_Audio(),
        current_user={"API_KEY": "k", "identities": [{"user_id": "u1"}]},
    )
    assert response.status_code == 503
    body = __import__("json").loads(response.body)
    assert body["error"] == "vendor_key_refused"
    assert "allotment" in body["detail"].lower()


@pytest.mark.asyncio
async def test_transcribe_reports_empty_vendor_credits_as_credit_exhausted(
    monkeypatch,
):
    _install_transcribe(
        monkeypatch,
        RuntimeError("Error code: 429 - insufficient_quota: You exceeded your current quota"),
    )
    response = await webapp_module.transcribe_recording(
        assistant_id="assistant-1",
        audio=_Audio(),
        current_user={"API_KEY": "k", "identities": [{"user_id": "u1"}]},
    )
    assert response.status_code == 503
    body = __import__("json").loads(response.body)
    assert body["error"] == "model_provider_credit_exhausted"
