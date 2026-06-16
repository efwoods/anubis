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
    FEATURE_NAMES,
    deserialize_features_by_doc_id,
    features_by_doc_id_to_arr,
    recompute_ground_truth_artifacts,
    serialize_features_by_doc_id,
)

N_FEATURES = len(FEATURE_NAMES)  # 33


def _make_corpus(n_docs: int, seed: int = 0) -> dict:
    """A {document_id: 1-D 33-feature row} dict of pseudo-random rows."""
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
