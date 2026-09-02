"""Unit tests for the direct-quote backfill script's decision helpers.

The script itself only orchestrates: it enumerates avatars, resolves each one's
owner, and hands the pair to the shared calibration entry point (covered by
``test_calibrate_ground_truth.py``). What is worth pinning here is the reasoning
that decides WHICH avatar gets calibrated and UNDER WHOSE namespace — a wrong
owner id writes a cloud into a namespace the message path never reads, and the
avatar stays silently uncalibrated exactly as before the backfill ran.

Follows ``test_retrain_baseline.py``: exercise the pure helpers, with a hand
written fake in place of the database rather than a mocking framework.
"""

import pytest

import scripts.backfill_ground_truth_calibration as backfill
from scripts.backfill_ground_truth_calibration import (
    CALIBRATION_ARTIFACT_KEYS,
    FITTED_MODEL_KEY,
    parse_arguments,
    redact_credentials,
    resolve_owner_user_id,
    select_uncalibrated_candidates,
    unreachable_store_message,
)


class _FakeCursor:
    """Returns a queued result per execute, in call order."""

    def __init__(self, results):
        self._results = list(results)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, statement, parameters=None):
        self.executed.append((statement, parameters))
        self._current = self._results.pop(0) if self._results else []

    def fetchone(self):
        return self._current[0] if self._current else None

    def fetchall(self):
        return list(self._current)


class _FakeConnection:
    def __init__(self, results):
        self.cursor_object = _FakeCursor(results)
        self.closed = False

    def cursor(self):
        return self.cursor_object

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Owner resolution
# ---------------------------------------------------------------------------


def test_owner_comes_from_the_creator_id_marker_first():
    """The marker is what the API itself treats as the owner, so it wins."""
    connection = _FakeConnection([[("6a64d1ef4e063740350632ae",)]])
    assert (
        resolve_owner_user_id(connection, "0f92b031")
        == "6a64d1ef4e063740350632ae"
    )


def test_owner_falls_back_to_the_quote_namespace_owner():
    """Without a marker, whoever the existing quotes are stored under is the owner.

    The calibration has to land in the same namespace as the quotes it was fitted
    from, or the message path will not find it.
    """
    connection = _FakeConnection(
        [
            [],  # no creator_id marker
            [("6a64d1ef4e063740350632ae",)],
        ]
    )
    assert (
        resolve_owner_user_id(connection, "0f92b031")
        == "6a64d1ef4e063740350632ae"
    )


def test_owner_resolution_tolerates_stored_whitespace():
    """Some recorded identifiers carry trailing whitespace; matching must survive."""
    connection = _FakeConnection([[("  6a64d1ef4e063740350632ae  ",)]])
    assert (
        resolve_owner_user_id(connection, "0f92b031")
        == "6a64d1ef4e063740350632ae"
    )


def test_ambiguous_owner_raises_rather_than_guessing():
    connection = _FakeConnection([[], [("owner-a",), ("owner-b",)]])
    with pytest.raises(ValueError, match="more than one owner"):
        resolve_owner_user_id(connection, "0f92b031")


def test_unknown_owner_raises_rather_than_guessing():
    connection = _FakeConnection([[], []])
    with pytest.raises(ValueError, match="No creator_id marker"):
        resolve_owner_user_id(connection, "0f92b031")


# ---------------------------------------------------------------------------
# Operator-facing messages
# ---------------------------------------------------------------------------


def test_credentials_are_redacted_before_printing():
    redacted = redact_credentials(
        "postgresql://postgres:hunter2@host.docker.internal:5432/postgres"
    )
    assert "hunter2" not in redacted
    assert "host.docker.internal:5432/postgres" in redacted


def test_unreachable_store_message_names_the_usual_cause_and_the_remedies():
    """The committed URI fails on the host for a reason worth stating outright."""
    message = unreachable_store_message(
        "postgresql://postgres:hunter2@host.docker.internal:5432/postgres",
        OSError("Name or service not known"),
    )
    assert "hunter2" not in message
    assert "host.docker.internal" in message
    assert "docker compose exec" in message
    assert "localhost:5432" in message


# ---------------------------------------------------------------------------
# Artifact bookkeeping
# ---------------------------------------------------------------------------


def test_fitted_model_is_the_key_that_marks_an_avatar_calibrated():
    """The per-document dict is written even below the floor, so it cannot be the
    marker: keying on it would re-select avatars that are correctly deferred."""
    assert FITTED_MODEL_KEY == "ground_truth_text_features_model_b64_pkl"
    assert FITTED_MODEL_KEY in CALIBRATION_ARTIFACT_KEYS


def test_reported_artifacts_match_what_calibration_actually_writes():
    """Guards against the summary drifting away from the writer."""
    assert set(CALIBRATION_ARTIFACT_KEYS) == {
        "ground_truth_text_features_by_doc_id_dict_str",
        "ground_truth_text_empirical_threshold_list_str",
        "ground_truth_text_features_model_b64_pkl",
        "key_phrase_profile",
        "style_profile",
    }


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_assistant_id_is_repeatable():
    arguments = parse_arguments(["--assistant-id", "a1", "--assistant-id", "a2"])
    assert arguments.assistant_id == ["a1", "a2"]
    assert arguments.all is False


def test_dry_run_and_all_are_parsed():
    arguments = parse_arguments(["--all", "--dry-run", "--min-quote-rows", "25"])
    assert arguments.all is True
    assert arguments.dry_run is True
    assert arguments.min_quote_rows == 25


# ---------------------------------------------------------------------------
# Candidate selection and the dry run
# ---------------------------------------------------------------------------


class _StubContext:
    """Stands in for GlobalContext so the tests need no configured environment."""

    async_postgres_store_uri = "postgresql://backfill:secret@localhost:5432/postgres"


def test_candidate_selection_applies_the_minimum_row_floor():
    """The floor must reach the query, and already-fitted avatars must be excluded.

    Both are parameters of the SQL rather than a Python filter, so a wrong or
    dropped parameter would silently widen the sweep: with the floor missing,
    --all would refit every avatar holding a single stray quote, each one paying a
    whole-corpus quadratic fit.
    """
    connection = _FakeConnection([[("  owner-1  ", "  assistant-1  ", 42)]])

    candidates = select_uncalibrated_candidates(connection, 25)

    # Stored ids can carry surrounding whitespace; a namespace built from an
    # untrimmed id is not the namespace the message path reads.
    assert candidates == [("owner-1", "assistant-1", 42)]
    statement, parameters = connection.cursor_object.executed[0]
    assert parameters == (25, FITTED_MODEL_KEY)
    assert "quote_row_count >= %s" in statement
    assert "NOT EXISTS" in statement


def test_dry_run_reports_targets_without_calibrating_anything(monkeypatch, capsys):
    """--dry-run is the safe way to preview a fleet-wide sweep.

    It must reach candidate selection (otherwise it previews nothing useful) and
    stop strictly before any fit, since calibration overwrites the avatar's stored
    threshold and model.
    """
    connection = _FakeConnection([[("owner-1", "assistant-1", 42)]])
    monkeypatch.setattr(backfill, "GlobalContext", _StubContext)
    monkeypatch.setattr(
        backfill, "open_enumeration_connection", lambda store_uri: connection
    )

    def _refuse_to_write(*args, **kwargs):
        raise AssertionError("--dry-run must not calibrate anything")

    monkeypatch.setattr(backfill, "run_backfill", _refuse_to_write)

    exit_code = backfill.main(["--all", "--dry-run"])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "assistant-1" in printed
    assert "nothing written" in printed
    # The password must not reach the console even on the happy path.
    assert "secret" not in printed
    assert connection.closed is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
