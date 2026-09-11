"""The ambient-vision settings are read from the environment like every other knob."""

from src.anubis.utils.context import GlobalContext


def test_ambient_settings_have_defaults_and_read_the_environment(monkeypatch):
    for name in (
        "AMBIENT_CAPTURE_ENABLED",
        "AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS",
        "AMBIENT_CAPTURE_MAX_IMAGE_BYTES",
        "AMBIENT_PREFERENCE_RECALL_LIMIT",
        "AMBIENT_RESPOND_SALIENCE_FLOOR",
        "AMBIENT_NOTIFY_SALIENCE_FLOOR",
        "AMBIENT_RESPOND_COOLDOWN_SECONDS",
        "AMBIENT_RESPOND_COOLDOWN_OVERRIDE_SALIENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = GlobalContext()
    assert defaults.ambient_capture_enabled == "true"
    assert defaults.ambient_capture_min_interval_seconds == 10.0
    assert defaults.ambient_capture_max_image_bytes == 2_000_000
    assert defaults.ambient_preference_recall_limit == 8
    assert defaults.ambient_respond_salience_floor == 0.55
    assert defaults.ambient_notify_salience_floor == 0.70
    assert defaults.ambient_respond_cooldown_seconds == 300.0
    assert defaults.ambient_respond_cooldown_override_salience == 0.90

    monkeypatch.setenv("AMBIENT_CAPTURE_ENABLED", "false")
    monkeypatch.setenv("AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("AMBIENT_CAPTURE_MAX_IMAGE_BYTES", "500000")
    monkeypatch.setenv("AMBIENT_PREFERENCE_RECALL_LIMIT", "3")
    monkeypatch.setenv("AMBIENT_RESPOND_SALIENCE_FLOOR", "0.8")
    monkeypatch.setenv("AMBIENT_NOTIFY_SALIENCE_FLOOR", "0.9")
    monkeypatch.setenv("AMBIENT_RESPOND_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("AMBIENT_RESPOND_COOLDOWN_OVERRIDE_SALIENCE", "0.99")
    configured = GlobalContext()
    assert configured.ambient_capture_enabled == "false"
    assert configured.ambient_capture_min_interval_seconds == 5.0
    assert configured.ambient_capture_max_image_bytes == 500000
    assert configured.ambient_preference_recall_limit == 3
    assert configured.ambient_respond_salience_floor == 0.8
    assert configured.ambient_notify_salience_floor == 0.9
    assert configured.ambient_respond_cooldown_seconds == 60.0
    assert configured.ambient_respond_cooldown_override_salience == 0.99
