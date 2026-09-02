"""Unit tests for the two per-reply authenticity-metadata decisions in ``graph.py``.

Both are small, pure, and guard a failure that is invisible at runtime:

* ``_publishable_avatar_key_phrase_rate`` — publishing 0.0 for an avatar that has
  no calibrated phrase profile would be read by a client as "this reply reuses
  none of the target's phrasing" when the truth is that nothing has been measured
  yet. The two cases must stay distinguishable.
* the ground-truth SHAP explainer cache — serving an explainer fitted against a
  superseded model produces attributions that look plausible and are wrong, with
  nothing in the response to indicate staleness.

The explainer objects here are opaque sentinels: the cache is keyed and validated
entirely on the serialized model, so it never inspects what it stores.
"""

import math

import pytest

import src.anubis.graph as graph_module
from src.anubis.graph import (
    _cached_ground_truth_explainer,
    _publishable_avatar_key_phrase_rate,
    _remember_ground_truth_explainer,
)

FEATURE_WIDTH = 28


@pytest.fixture(autouse=True)
def _clear_explainer_cache():
    """The cache is module state; isolate every test from its neighbours."""
    graph_module._ground_truth_explainer_cache.clear()
    yield
    graph_module._ground_truth_explainer_cache.clear()


# ---------------------------------------------------------------------------
# Avatar-referenced key-phrase rate
# ---------------------------------------------------------------------------


def test_rate_is_published_when_the_avatar_has_signature_phrases():
    rate = _publishable_avatar_key_phrase_rate(
        ["born with cystic fibrosis"], {"key_phrase_rate": 0.093}
    )
    assert rate == 0.093


def test_a_genuine_zero_is_published_rather_than_suppressed():
    """A calibrated avatar whose reply reuses no phrase really did score zero."""
    rate = _publishable_avatar_key_phrase_rate(
        ["born with cystic fibrosis"], {"key_phrase_rate": 0.0}
    )
    assert rate == 0.0


@pytest.mark.parametrize("empty_phrase_set", [None, [], ()])
def test_an_uncalibrated_avatar_reports_unknown_not_zero(empty_phrase_set):
    """Without a phrase profile the rate is unmeasured, and 0.0 would lie.

    ``extract_style_features`` returns 0.0 for an empty phrase set, so the caller
    cannot distinguish the two on the value alone — this is where they separate.
    """
    assert (
        _publishable_avatar_key_phrase_rate(empty_phrase_set, {"key_phrase_rate": 0.0})
        is None
    )


def test_nan_is_not_published():
    """The metadata copy must stay strict JSON; a bare NaN token breaks the frame."""
    assert (
        _publishable_avatar_key_phrase_rate(["a phrase"], {"key_phrase_rate": math.nan})
        is None
    )


def test_a_missing_rate_is_not_published():
    assert _publishable_avatar_key_phrase_rate(["a phrase"], {}) is None


# ---------------------------------------------------------------------------
# Ground-truth SHAP explainer cache
# ---------------------------------------------------------------------------


def test_cache_misses_before_anything_is_remembered():
    assert _cached_ground_truth_explainer("owner", "avatar", "model", FEATURE_WIDTH) is None


def test_a_remembered_explainer_is_reused_for_the_same_model():
    explainer = object()
    _remember_ground_truth_explainer("owner", "avatar", "model", FEATURE_WIDTH, explainer)

    assert (
        _cached_ground_truth_explainer("owner", "avatar", "model", FEATURE_WIDTH)
        is explainer
    )


def test_recalibration_invalidates_the_cached_explainer():
    """Calibration rewrites the serialized model, which is the cache's validator.

    This is why no explicit invalidation hook is needed anywhere in the
    calibration path: a refit changes the blob, and the changed blob misses.
    """
    _remember_ground_truth_explainer(
        "owner", "avatar", "model-before-refit", FEATURE_WIDTH, object()
    )

    assert (
        _cached_ground_truth_explainer(
            "owner", "avatar", "model-after-refit", FEATURE_WIDTH
        )
        is None
    )


def test_a_feature_width_change_invalidates_the_cached_explainer():
    """The explainer embeds a background matrix of a fixed width.

    Scoring a current-width candidate against a previous-width background raises,
    so a vector-version bump must miss rather than reuse.
    """
    _remember_ground_truth_explainer("owner", "avatar", "model", 27, object())

    assert _cached_ground_truth_explainer("owner", "avatar", "model", 28) is None


def test_avatars_do_not_share_a_cache_entry():
    first, second = object(), object()
    _remember_ground_truth_explainer("owner", "avatar-one", "model", FEATURE_WIDTH, first)
    _remember_ground_truth_explainer("owner", "avatar-two", "model", FEATURE_WIDTH, second)

    assert (
        _cached_ground_truth_explainer("owner", "avatar-one", "model", FEATURE_WIDTH)
        is first
    )
    assert (
        _cached_ground_truth_explainer("owner", "avatar-two", "model", FEATURE_WIDTH)
        is second
    )


def test_the_cache_is_bounded_and_evicts_least_recently_used():
    """Each entry holds a background matrix, so an unbounded cache leaks memory."""
    limit = graph_module._GROUND_TRUTH_EXPLAINER_CACHE_MAX_ENTRIES
    for index in range(limit):
        _remember_ground_truth_explainer(
            "owner", f"avatar-{index}", "model", FEATURE_WIDTH, object()
        )

    # Touch the oldest so it is no longer the eviction candidate.
    assert (
        _cached_ground_truth_explainer("owner", "avatar-0", "model", FEATURE_WIDTH)
        is not None
    )
    _remember_ground_truth_explainer(
        "owner", "avatar-overflow", "model", FEATURE_WIDTH, object()
    )

    assert len(graph_module._ground_truth_explainer_cache) == limit
    # avatar-0 was refreshed, so avatar-1 is the least recently used.
    assert (
        _cached_ground_truth_explainer("owner", "avatar-0", "model", FEATURE_WIDTH)
        is not None
    )
    assert (
        _cached_ground_truth_explainer("owner", "avatar-1", "model", FEATURE_WIDTH)
        is None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
