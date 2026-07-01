"""Regenerate the bundled ChatGPT-baseline stylometry artifacts.

The production authenticity check in ``graph._attach_analyzed_features`` compares
every avatar response against a fixed cloud of *unmodified ChatGPT* responses.
That cloud is shipped as two bundled artifacts under
``src/anubis/utils/dataset/``:

* ``baseline_features_arr.npy`` — an ``(n_baseline_docs, F)`` matrix, one
  :func:`extract_style_features` row per baseline assistant reply, and
* ``baseline_features_model_b64.pkl`` — an ``IsolationForest`` fit on that
  matrix, ``base64(pickle(model))`` written as raw bytes (the loader reads the
  file, ``.decode('utf-8')``s it, then ``base64.b64decode`` + ``pickle.loads``).

Both bake in the feature-vector WIDTH ``F = len(FEATURE_NAMES)``. Whenever
:data:`STYLE_FEATURE_VECTOR_VERSION` changes (a feature is added/removed/reordered)
these artifacts must be rebuilt at the new width, or the runtime Mahalanobis /
IsolationForest calls will raise on the shape mismatch. This script rebuilds both
from the baseline corpus so the regeneration is reproducible and reviewable.

Source corpus: ``data/synthetic_gpt-5-4-nano-baseline-full.jsonl`` — OpenAI
fine-tuning JSONL, one conversation per line, whose final ``assistant`` message
is the baseline reply we fingerprint.

Usage::

    python data/build_baseline_features_arr.py
"""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np

from src.anubis.utils.dataset.style_features import (
    FEATURE_NAMES,
    STYLE_FEATURE_VECTOR_VERSION,
    extract_style_features,
)

# Repo-root-relative paths (this file lives in data/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_JSONL = _REPO_ROOT / "data" / "synthetic_gpt-5-4-nano-baseline-full.jsonl"
_DATASET_DIR = _REPO_ROOT / "src" / "anubis" / "utils" / "dataset"
_ARR_PATH = _DATASET_DIR / "baseline_features_arr.npy"
_MODEL_PATH = _DATASET_DIR / "baseline_features_model_b64.pkl"


def _baseline_assistant_texts() -> List[str]:
    """Return the final assistant reply from every conversation in the baseline JSONL."""
    texts: List[str] = []
    with _BASELINE_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            conversation = json.loads(line)
            assistant_messages = [
                message["content"]
                for message in conversation.get("messages", [])
                if message.get("role") == "assistant" and message.get("content")
            ]
            if assistant_messages:
                texts.append(assistant_messages[-1])
    return texts


def build() -> None:
    """Rebuild and write both bundled baseline artifacts at the current width."""
    from sklearn.ensemble import IsolationForest

    texts = _baseline_assistant_texts()
    print(f"Loaded {len(texts)} baseline assistant replies from {_BASELINE_JSONL.name}")

    feature_rows = [
        [features[name] for name in FEATURE_NAMES]
        for features in (extract_style_features(text) for text in texts)
    ]
    feature_matrix = np.asarray(feature_rows, dtype=np.float64)  # (n_docs, F)

    # A degenerate short reply could leave a metric NaN; impute with the column
    # median so IsolationForest / StandardScaler (which reject NaN) stay well-posed
    # and the row still contributes.
    column_medians = np.nanmedian(feature_matrix, axis=0)
    column_medians = np.where(np.isnan(column_medians), 0.0, column_medians)
    nan_positions = np.isnan(feature_matrix)
    feature_matrix[nan_positions] = np.take(
        column_medians, np.where(nan_positions)[1]
    )

    print(
        f"Built baseline feature matrix {feature_matrix.shape} "
        f"(feature-vector version {STYLE_FEATURE_VECTOR_VERSION}, width {len(FEATURE_NAMES)})"
    )

    np.save(_ARR_PATH, feature_matrix, allow_pickle=False)
    print(f"Wrote {_ARR_PATH.relative_to(_REPO_ROOT)}")

    model = IsolationForest().fit(feature_matrix)
    model_b64 = base64.b64encode(pickle.dumps(model))
    _MODEL_PATH.write_bytes(model_b64)
    print(f"Wrote {_MODEL_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    build()
