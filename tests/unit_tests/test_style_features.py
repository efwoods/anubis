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
    feature_row_is_all_nan,
    features_by_doc_id_to_arr,
    recompute_ground_truth_artifacts,
    sanitize_ground_truth_feature_matrix,
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
""" Feature-vector composition (v3: pruned collinear set + key phrases)  """
""" ------------------------------------------------------------------ """


def test_feature_metadata_dicts_cover_every_feature():
    # The legible-title and description dicts must stay 1:1 with FEATURE_NAMES so
    # build_style_profile_str never KeyErrors on a feature.
    assert set(FEATURE_NAMES_HUMAN_LEGIBLE) == set(FEATURE_NAMES)
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)
    assert STYLE_FEATURE_VECTOR_VERSION >= 3


def test_v3_removed_features_are_absent():
    # The nine multicollinear features pruned in vector version 3 must no longer
    # appear in the vector (nor in the metadata dicts).
    features = extract_style_features("A short but real sentence to measure.")
    for name in (
        "type_token_ratio",
        "maas_lexical_diversity",
        "yule_characteristic_k",
        "pos_sequence_compressibility",
        "flesch_kincaid_grade",
        "gunning_fog_index",
        "smog_index",
        "vocabulary_size_unique_words",
        "total_word_count",
    ):
        assert name not in features
        assert name not in FEATURE_NAMES


def test_average_word_length_present_and_correct():
    features = extract_style_features("Big cats run. Big cats run fast today.")
    assert "average_word_length_characters" in features
    # Average characters per word is finite and in a sane band for short words.
    assert 3.0 <= features["average_word_length_characters"] <= 5.0


def test_key_phrase_rate_positive_with_matching_phrases_zero_without():
    text = "you know I got it, you know what I mean."
    # The avatar's signature phrase appears, so the rate is positive...
    with_phrases = extract_style_features(text, key_phrases=["you know"])
    assert with_phrases["key_phrase_rate"] > 0.0
    # ...and it is 0.0 when there is no phrase set (or an empty one).
    assert extract_style_features(text)["key_phrase_rate"] == 0.0
    assert extract_style_features(text, key_phrases=[])["key_phrase_rate"] == 0.0


def test_update_key_phrases_only_swaps_only_the_rate():
    text = "you know I got it, you know what I mean."
    baseline_scored = extract_style_features(text, key_phrases=["completely absent"])
    assert baseline_scored["key_phrase_rate"] == 0.0

    # Re-measure ONLY key_phrase_rate against a matching phrase set...
    updated = extract_style_features(
        text,
        key_phrases=["you know"],
        update_key_phrases_only=True,
        features_dict=baseline_scored,
    )
    assert updated["key_phrase_rate"] > 0.0
    # ...every other feature carries over, and the input dict is not mutated.
    for name in FEATURE_NAMES:
        if name != "key_phrase_rate":
            assert updated[name] == baseline_scored[name]
    assert baseline_scored["key_phrase_rate"] == 0.0


def test_update_key_phrases_only_requires_features_dict():
    with pytest.raises(ValueError):
        extract_style_features(
            "some text", key_phrases=["some text"], update_key_phrases_only=True
        )


def test_extract_returns_exactly_the_declared_vector_width():
    features = extract_style_features("A short but real sentence to measure.")
    assert list(features.keys()) == list(FEATURE_NAMES)
    assert len(features) == N_FEATURES


def test_all_punctuation_input_yields_nan_word_length_not_crash():
    # No word tokens: average word length is NaN (imputed downstream), and the
    # vector width is still exactly N_FEATURES.
    features = extract_style_features("!!! ??? ...")
    assert np.isnan(features["average_word_length_characters"])
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


""" ------------------------------------------------------------------ """
""" NaN sanitation of the ground-truth corpus                          """
""" ------------------------------------------------------------------ """


def test_extract_style_features_degenerate_texts_yield_nan_rows():
    # Empty / URL-only / mention-only quote lines (common in tweet corpora)
    # clean down to nothing and come back as an all-NaN row (dropped by the
    # sanitizer).
    for degenerate_text in ("", "https://example.com/x", "@BillyM2k"):
        features = extract_style_features(degenerate_text)
        row = np.array([features[name] for name in FEATURE_NAMES])
        assert feature_row_is_all_nan(row), degenerate_text

    # An emoji-only line keeps its emoji tokens, so only the alphabetic-token
    # metrics are NaN — a PARTIAL NaN row (imputed by the sanitizer, not dropped).
    emoji_features = extract_style_features("😀 👍")
    emoji_row = np.array([emoji_features[name] for name in FEATURE_NAMES])
    assert not feature_row_is_all_nan(emoji_row)
    assert np.isnan(emoji_row).any()


def test_feature_row_is_all_nan_predicate():
    assert feature_row_is_all_nan(np.full(N_FEATURES, np.nan))
    partial = np.full(N_FEATURES, np.nan)
    partial[0] = 1.0
    assert not feature_row_is_all_nan(partial)
    assert not feature_row_is_all_nan(np.zeros(N_FEATURES))


def test_sanitize_drops_all_nan_rows_and_imputes_partial_cells():
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((6, N_FEATURES))
    matrix[0, :] = np.nan          # dropped entirely
    matrix[1, 4] = np.nan          # imputed with column 4's nanmedian
    matrix[2, 4] = np.nan

    sanitized = sanitize_ground_truth_feature_matrix(matrix)

    assert sanitized.shape == (5, N_FEATURES)
    assert not np.isnan(sanitized).any()
    expected_median = np.nanmedian(matrix[1:, 4])
    np.testing.assert_allclose(sanitized[0, 4], expected_median)
    # Rows without NaN cells are untouched.
    np.testing.assert_allclose(sanitized[2:], matrix[3:])
    # The input matrix is not mutated.
    assert np.isnan(matrix[0]).all()


def test_sanitize_all_nan_column_falls_back_to_zero():
    rng = np.random.default_rng(12)
    matrix = rng.standard_normal((4, N_FEATURES))
    matrix[:, 7] = np.nan
    sanitized = sanitize_ground_truth_feature_matrix(matrix)
    assert sanitized.shape == (4, N_FEATURES)
    np.testing.assert_allclose(sanitized[:, 7], 0.0)


def test_sanitize_empty_matrix_passes_through():
    empty = np.empty((0, N_FEATURES))
    assert sanitize_ground_truth_feature_matrix(empty).shape == (0, N_FEATURES)


def test_serialize_nan_cells_as_strict_json_null_and_round_trip():
    # A partial-NaN row must serialize to STRICT JSON (null, never the bare NaN
    # token that PostgreSQL ::jsonb rejects) and round-trip back to nan.
    corpus = _make_corpus(2, seed=9)
    corpus["doc-0"][4] = np.nan

    serialized = serialize_features_by_doc_id(corpus)
    assert "NaN" not in serialized
    # Strict parse (json.loads with the NaN constant rejected) must succeed.
    strict = json.loads(serialized, parse_constant=lambda token: pytest.fail(token))
    assert strict["doc-0"][4] is None

    rebuilt = deserialize_features_by_doc_id(serialized)
    assert np.isnan(rebuilt["doc-0"][4])
    np.testing.assert_allclose(rebuilt["doc-1"], corpus["doc-1"])


def test_recompute_ground_truth_artifacts_tolerates_nan_rows():
    # A corpus holding an all-NaN row and partial-NaN cells (degenerate quote
    # lines) must calibrate instead of raising sklearn's
    # "Input X contains NaN" — the crash that aborted whole media uploads.
    corpus = features_by_doc_id_to_arr(_make_corpus(20, seed=5))
    corpus[0, :] = np.nan
    corpus[1, 3] = np.nan
    corpus[9, 12] = np.nan

    threshold_list_str, model_b64_pkl = recompute_ground_truth_artifacts(corpus)

    threshold = np.array(json.loads(threshold_list_str)).flatten()
    assert np.isfinite(threshold).all()
    model = pickle.loads(base64.b64decode(model_b64_pkl))
    # The all-NaN row was dropped before the IsolationForest fit.
    assert model.n_features_in_ == N_FEATURES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
