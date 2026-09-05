"""The ``ambient=true`` gate on ``POST /message/{assistant_id}``.

An ambient observation is a webcam / screen snapshot the browser sends on a
timer through the one message endpoint. The gate refuses what a typed turn
never has to worry about: a switched-off deployment, an anonymous visitor
(who would watch through the owner's allotment), a turn with no snapshot, an
oversized snapshot, and a client faster than the per-thread floor.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import webapp as webapp_module
from src.anubis.utils.ambient import observations as observations_module


def _user(anonymous=False):
    return {"API_KEY": "k", "identities": [{"user_id": "u1"}], "anonymous": anonymous}


def _file(name="webcam.jpg", size=1000):
    return SimpleNamespace(filename=name, size=size, content_type="image/jpeg")


@pytest.fixture(autouse=True)
def _fresh_throttle_and_flags(monkeypatch):
    monkeypatch.setattr(
        observations_module, "ambient_throttle", observations_module.AmbientThrottle()
    )
    monkeypatch.setattr(
        webapp_module, "is_anonymous_user", lambda user: bool(user.get("anonymous"))
    )
    for name in (
        "AMBIENT_CAPTURE_ENABLED",
        "AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS",
        "AMBIENT_CAPTURE_MAX_IMAGE_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def _call(user=None, files=None, thread_id="thread-1", **overrides):
    return webapp_module.enforce_ambient_request(
        user or _user(),
        files=files
        if files is not None
        else [_file("webcam.jpg"), _file("screen.jpg")],
        image_filenames=overrides.pop("image_filenames", ["webcam.jpg", "screen.jpg"]),
        sources=overrides.pop("sources", None),
        captured_at=overrides.pop("captured_at", "2026-09-04T15:00:00Z"),
        voice_mode=overrides.pop("voice_mode", False),
        thread_id=thread_id,
    )


def test_a_valid_observation_yields_the_hidden_message_tag():
    kwargs = _call(voice_mode=True)
    assert kwargs["hidden"] is True
    assert kwargs["kind"] == "ambient_observation"
    assert kwargs["image_filenames"] == ["webcam.jpg", "screen.jpg"]
    assert kwargs["ambient"]["sources"] == ["webcam", "screen"]
    assert kwargs["ambient"]["captured_at"] == "2026-09-04T15:00:00Z"
    assert kwargs["ambient"]["voice_mode"] is True
    assert kwargs["ambient"]["observation_id"]


def test_explicit_sources_win_over_filenames():
    kwargs = _call(sources='["screen", "webcam"]')
    assert kwargs["ambient"]["sources"] == ["screen", "webcam"]


def test_a_switched_off_deployment_refuses_with_404(monkeypatch):
    monkeypatch.setenv("AMBIENT_CAPTURE_ENABLED", "false")
    with pytest.raises(HTTPException) as error:
        _call()
    assert error.value.status_code == 404


def test_an_anonymous_visitor_is_refused():
    with pytest.raises(HTTPException) as error:
        _call(user=_user(anonymous=True))
    assert error.value.status_code == 403


def test_an_observation_without_a_snapshot_is_refused():
    with pytest.raises(HTTPException) as error:
        _call(files=[], image_filenames=[])
    assert error.value.status_code == 422


def test_an_oversized_snapshot_is_refused(monkeypatch):
    monkeypatch.setenv("AMBIENT_CAPTURE_MAX_IMAGE_BYTES", "500")
    with pytest.raises(HTTPException) as error:
        _call(files=[_file("screen.jpg", size=501)], image_filenames=["screen.jpg"])
    assert error.value.status_code == 413


def test_a_second_observation_inside_the_floor_is_throttled(monkeypatch):
    monkeypatch.setenv("AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS", "60")
    _call(thread_id="busy-thread")
    with pytest.raises(HTTPException) as error:
        _call(thread_id="busy-thread")
    assert error.value.status_code == 429
    assert int(error.value.headers["Retry-After"]) >= 1
    # A different conversation, and a brand-new one, are not held.
    _call(thread_id="quiet-thread")
    _call(thread_id=None)
