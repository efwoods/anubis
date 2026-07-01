"""Unit tests for the ground-truth corpus persistence helpers.

These cover the deterministic, offline serialization/reconstruction/recalibration
helpers in :mod:`src.anubis.utils.dataset.style_features` that the "direct quote"
corpus round-trips through the LangGraph store with:

* ``serialize_features_by_doc_id`` / ``deserialize_features_by_doc_id`` round-trip,
* ``features_by_doc_id_to_arr`` reconstruction (and the empty-corpus shape), and
* ``recompute_ground_truth_artifacts`` producing a parseable threshold + an
  unpicklable IsolationForest.
"""

import base64
import json
import pickle

import numpy as np
import pytest

from src.anubis.utils.dataset.style_features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    FEATURE_NAMES_HUMAN_LEGIBLE,
    STYLE_FEATURE_VECTOR_VERSION,
    baseline_feature_array_is_current,
    deserialize_features_by_doc_id,
    extract_style_features,
    features_by_doc_id_to_arr,
    recompute_ground_truth_artifacts,
    serialize_features_by_doc_id,
)

N_FEATURES = len(FEATURE_NAMES)  # current vector width (STYLE_FEATURE_VECTOR_VERSION)


def _make_corpus(n_docs: int, seed: int = 0) -> dict:
    """A {document_id: 1-D current-width feature row} dict of pseudo-random rows."""
    rng = np.random.default_rng(seed)
    return {f"doc-{i}": rng.standard_normal(N_FEATURES) for i in range(n_docs)}


def test_serialize_deserialize_to_arr_round_trip():
    corpus = _make_corpus(5)

    rebuilt = deserialize_features_by_doc_id(serialize_features_by_doc_id(corpus))

    # Same keys preserved; each row faithfully reconstructed.
    assert set(rebuilt) == set(corpus)
    for doc_id, row in corpus.items():
        np.testing.assert_allclose(rebuilt[doc_id], row)

    # Reconstructed matrix matches a manual vstack (insertion order preserved).
    arr = features_by_doc_id_to_arr(rebuilt)
    assert arr.shape == (5, N_FEATURES)
    np.testing.assert_allclose(arr, np.vstack(list(corpus.values())))


def test_deserialize_falsy_inputs_yield_empty_dict():
    for falsy in (None, "", "{}"):
        assert deserialize_features_by_doc_id(falsy) == {}


def test_features_by_doc_id_to_arr_empty_shape():
    arr = features_by_doc_id_to_arr({})
    assert arr.shape == (0, N_FEATURES)


def test_serialize_is_json_object_keyed_by_doc_id():
    corpus = _make_corpus(3, seed=7)
    raw = json.loads(serialize_features_by_doc_id(corpus))
    assert set(raw) == set(corpus)
    # Each value is a plain JSON list of N_FEATURES floats (store-safe).
    for row in raw.values():
        assert isinstance(row, list)
        assert len(row) == N_FEATURES


def test_recompute_ground_truth_artifacts_round_trips():
    # Enough rows for the leave-one-out empirical distribution to be well-posed.
    arr = features_by_doc_id_to_arr(_make_corpus(40, seed=3))

    threshold_str, model_b64 = recompute_ground_truth_artifacts(arr)

    # Threshold is a JSON-parseable finite scalar.
    threshold = json.loads(threshold_str)
    assert np.isfinite(float(threshold))

    # Model is a base64-encoded pickle of a fitted IsolationForest.
    model = pickle.loads(base64.b64decode(model_b64))
    preds = model.predict(arr[:1])
    assert preds.shape == (1,)
    assert preds[0] in (-1, 1)


""" ------------------------------------------------------------------ """
""" Feature-vector composition (v2: word & vocabulary shape scalars)    """
""" ------------------------------------------------------------------ """


def test_feature_metadata_dicts_cover_every_feature():
    # The legible-title and description dicts must stay 1:1 with FEATURE_NAMES so
    # build_style_profile_str never KeyErrors on a feature.
    assert set(FEATURE_NAMES_HUMAN_LEGIBLE) == set(FEATURE_NAMES)
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)
    assert STYLE_FEATURE_VECTOR_VERSION >= 2


def test_new_word_and_vocabulary_scalars_present_and_correct():
    features = extract_style_features("Big cats run. Big cats run fast today.")
    for name in (
        "average_word_length_characters",
        "vocabulary_size_unique_words",
        "total_word_count",
    ):
        assert name in features

    # "big cats run big cats run fast today" -> 8 tokens, 5 unique types
    # (big, cats, run, fast, today).
    assert features["total_word_count"] == 8.0
    assert features["vocabulary_size_unique_words"] == 5.0
    # Average characters per word is finite and in a sane band for short words.
    assert 3.0 <= features["average_word_length_characters"] <= 5.0


def test_extract_returns_exactly_the_declared_vector_width():
    features = extract_style_features("A short but real sentence to measure.")
    assert list(features.keys()) == list(FEATURE_NAMES)
    assert len(features) == N_FEATURES


def test_all_punctuation_input_yields_nan_word_length_not_crash():
    # No word tokens: average word length is NaN (imputed downstream), and the
    # vector width is still exactly N_FEATURES.
    features = extract_style_features("!!! ??? ...")
    assert np.isnan(features["average_word_length_characters"])
    assert features["total_word_count"] == 0.0
    assert len(features) == N_FEATURES


""" ------------------------------------------------------------------ """
""" Feature-version migration of the persisted corpus                  """
""" ------------------------------------------------------------------ """


def test_deserialize_drops_stale_width_rows_keeps_current():
    # A corpus persisted under an older vector width must be pruned on read so it
    # cannot be stacked with (or scored against) current-width rows.
    stale_width = N_FEATURES - 3
    mixed = {
        "old-a": list(range(stale_width)),
        "old-b": list(range(stale_width)),
        "new": list(range(N_FEATURES)),
    }
    rebuilt = deserialize_features_by_doc_id(json.dumps(mixed))

    assert set(rebuilt) == {"new"}
    assert features_by_doc_id_to_arr(rebuilt).shape == (1, N_FEATURES)


def test_baseline_feature_array_is_current():
    assert baseline_feature_array_is_current(np.zeros((5, N_FEATURES)))
    assert not baseline_feature_array_is_current(np.zeros((5, N_FEATURES - 3)))
    # A 1-D array (not a matrix) is never a valid baseline cloud.
    assert not baseline_feature_array_is_current(np.zeros((N_FEATURES,)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
