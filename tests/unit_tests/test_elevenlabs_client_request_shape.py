"""The ElevenLabs SDK calls the voice pipeline makes.

* ``labels`` is always sent as a dictionary: SDK 2.65.0 serializes the argument
  with ``json.dumps`` before its omit filter runs, so leaving the argument out
  puts ``labels=null`` on the wire and the API answers 400 ``invalid_labels``.
* A vendor refusal is reduced to the API's own ``detail`` message instead of the
  SDK's header-and-body dump.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.voice import elevenlabs_client


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return _Recorder._Node(self, [name])

    class _Node:
        def __init__(self, recorder, path):
            self._recorder = recorder
            self._path = path

        def __getattr__(self, name):
            return _Recorder._Node(self._recorder, [*self._path, name])

        def __call__(self, *args, **kwargs):
            self._recorder.calls.append((".".join(self._path), args, kwargs))
            return SimpleNamespace(voice_id="voice-1")


@pytest.fixture
def recorded_client(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(elevenlabs_client, "_client", lambda context: recorder)
    return recorder


@pytest.mark.asyncio
async def test_instant_clone_sends_labels_as_a_dictionary(recorded_client):
    voice_id = await elevenlabs_client.create_instant_voice(
        SimpleNamespace(),
        name="Liv (instant)",
        clips=[("clip_000.mp3", b"bytes", "audio/mpeg")],
        description="Neural Nexus instant voice clone",
    )
    assert voice_id == "voice-1"
    path, _args, kwargs = recorded_client.calls[0]
    assert path == "voices.ivc.create"
    assert kwargs["labels"] == {}
    assert kwargs["files"] == [("clip_000.mp3", b"bytes", "audio/mpeg")]


@pytest.mark.asyncio
async def test_professional_clone_sends_labels_as_a_dictionary(recorded_client):
    await elevenlabs_client.create_professional_voice(
        SimpleNamespace(), name="Liv", language="en", description=""
    )
    path, _args, kwargs = recorded_client.calls[0]
    assert path == "voices.pvc.create"
    assert kwargs["labels"] == {}
    assert kwargs["description"] is None


def test_vendor_error_is_reduced_to_the_api_detail():
    vendor_error = Exception("headers: {...}, status_code: 400, body: {...}")
    vendor_error.status_code = 400
    vendor_error.body = {
        "detail": {
            "status": "invalid_labels",
            "message": "Labels must be serialized dictionary object.",
        }
    }
    assert (
        elevenlabs_client._describe_vendor_error(vendor_error)
        == "ElevenLabs rejected the request (400 invalid_labels: Labels must be serialized dictionary object.)"
    )

    string_detail = Exception("x")
    string_detail.status_code = 422
    string_detail.body = {"detail": "Not enough audio."}
    assert (
        elevenlabs_client._describe_vendor_error(string_detail)
        == "ElevenLabs rejected the request (422: Not enough audio.)"
    )

    plain = RuntimeError("connection reset")
    assert elevenlabs_client._describe_vendor_error(plain) == "connection reset"


def _vendor_error(status_code, detail):
    vendor_error = Exception("sdk dump")
    vendor_error.status_code = status_code
    vendor_error.body = {"detail": detail}
    return vendor_error


def test_a_401_invalid_api_key_is_a_refused_key_not_empty_credits():
    vendor_error = _vendor_error(
        401,
        {
            "type": "authentication_error",
            "code": "unauthorized",
            "message": "Invalid API key",
            "status": "invalid_api_key",
        },
    )
    classified = elevenlabs_client.classify_elevenlabs_vendor_error(vendor_error)
    assert isinstance(classified, elevenlabs_client.ElevenLabsKeyRefusedError)
    assert not isinstance(
        classified, elevenlabs_client.ElevenLabsCreditsExhaustedError
    )
    assert "invalid_api_key" in str(classified)


def test_quota_exceeded_is_empty_credits():
    vendor_error = _vendor_error(
        400,
        {
            "status": "quota_exceeded",
            "message": "You have insufficient quota to complete the request.",
        },
    )
    classified = elevenlabs_client.classify_elevenlabs_vendor_error(vendor_error)
    assert isinstance(classified, elevenlabs_client.ElevenLabsCreditsExhaustedError)


def test_a_blocked_voice_is_still_the_ban():
    vendor_error = _vendor_error(
        403,
        {
            "status": "detected_blocked_voice",
            "message": "This voice is blocked",
        },
    )
    classified = elevenlabs_client.classify_elevenlabs_vendor_error(vendor_error)
    assert isinstance(classified, elevenlabs_client.ElevenLabsVoiceBlockedError)
