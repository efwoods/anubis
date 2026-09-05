"""Regenerate the bundled ChatGPT-baseline stylometry artifacts.

The production authenticity check in ``graph._attach_analyzed_features`` compares
every avatar response against a fixed cloud of *unmodified ChatGPT* responses.
That cloud is shipped as bundled artifacts under ``src/anubis/utils/dataset/``:

* ``baseline_features_arr.npy`` — an ``(n_baseline_docs, F)`` matrix, one
  :func:`extract_style_features` row per baseline assistant reply,
* ``baseline_features_model_b64.pkl`` — an ``IsolationForest`` fit on that
  matrix, ``base64(pickle(model))`` written as raw bytes (the loader reads the
  file, ``.decode('utf-8')``s it, then ``base64.b64decode`` + ``pickle.loads``),
* ``baseline_features_explainer_b64.pkl`` — a SHAP ``KernelExplainer`` over that
  IsolationForest, same ``base64(pickle(...))`` encoding, so the runtime does not
  rebuild the (expensive) explainer on every call, and
* ``baseline_key_phrases.json`` — the signature key phrases self-discovered from
  the ChatGPT baseline corpus. The ``key_phrase_rate`` feature is avatar-relative,
  so the baseline rows are measured against the baseline's OWN discovered phrases
  (giving that column real, non-degenerate variance) rather than against an
  avatar's phrase set, which does not exist at baseline-build time.

Both matrix and model bake in the feature-vector WIDTH ``F = len(FEATURE_NAMES)``.
Whenever :data:`STYLE_FEATURE_VECTOR_VERSION` changes (a feature is
added/removed/reordered) these artifacts must be rebuilt at the new width, or the
runtime Mahalanobis / IsolationForest calls will raise on the shape mismatch. This
script rebuilds all of them from the baseline corpus so the regeneration is
reproducible and reviewable.

Source corpus: ``data/unmodified_inference_model_baseline_corpus.jsonl`` — OpenAI
fine-tuning JSONL, one conversation per line, whose final ``assistant`` message
is the baseline reply we fingerprint.

The corpus is model-specific even though its name is not: it holds whatever the
CURRENT inference model (``GlobalContext().model``) produced. Upgrading that model
invalidates these artifacts just as surely as changing the feature vector does, so
the full retrain — regenerate the corpus, then refit — belongs to
``scripts/retrain_chatgpt_baseline.py``. Run this module directly only to refit the
EXISTING corpus, which is what a feature-vector change (and nothing else) calls for.

Usage::

    # refit the existing corpus (feature-vector change)
    python data/build_baseline_features_arr.py

    # full retrain including corpus regeneration (inference-model upgrade)
    python scripts/retrain_chatgpt_baseline.py
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Allow running as a plain script from the repo root without installing the package
# (this module lives under data/, so ``src`` is not importable otherwise).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.dataset.key_phrases import discover_key_phrases  # noqa: E402
from src.anubis.utils.dataset.style_features import (  # noqa: E402
    FEATURE_NAMES,
    STYLE_FEATURE_VECTOR_VERSION,
    compute_empirical_distribution,
    extract_style_features,
)

# Repo-root-relative paths (this file lives in data/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CORPUS_PATH = (
    _REPO_ROOT / "data" / "unmodified_inference_model_baseline_corpus.jsonl"
)
_DATASET_DIR = _REPO_ROOT / "src" / "anubis" / "utils" / "dataset"
_ARR_PATH = _DATASET_DIR / "baseline_features_arr.npy"
_MODEL_PATH = _DATASET_DIR / "baseline_features_model_b64.pkl"
_EXPLAINER_PATH = _DATASET_DIR / "baseline_features_explainer_b64.pkl"
_KEY_PHRASES_PATH = _DATASET_DIR / "baseline_key_phrases.json"
# Stale, orphaned explainer from before the artifact moved under the dataset dir.
_STALE_EXPLAINER_PATH = _REPO_ROOT / "data" / "baseline_features_explainer_b64.pkl"

# SHAP background summary size — kmeans centroids used as the reference the
# KernelExplainer perturbs against. Matches the runtime rebuild in utility.py.
_SHAP_BACKGROUND_SIZE = 100


def _display_path(path: Path) -> str:
    """Render a path relative to the repo root when it is inside it, else absolutely.

    The artifact paths are module constants that tests redirect into a temp
    directory; ``relative_to`` would raise there and crash a fit that succeeded.
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _baseline_assistant_texts(corpus_path: Path) -> List[str]:
    """Return the final assistant reply from every conversation in the baseline JSONL."""
    texts: List[str] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
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


# Seed for the IsolationForest fit. An unseeded forest draws different subsamples
# and split thresholds on every fit, so two refits of the IDENTICAL corpus produced
# two different pickles (and two different SHAP explainers) — which made it
# impossible to tell "the committed artifact was regenerated from the committed
# corpus" from "someone fitted something else". With a fixed seed the committed
# corpus reproduces the committed pickle byte for byte.
_ISOLATION_FOREST_RANDOM_STATE = 0


@dataclass(frozen=True)
class BaselineBuildResult:
    """What one ``build()`` produced, for the caller to record as provenance.

    ``baseline_response_threshold`` is the value that must be written to
    ``BASELINE_RESPONSE_THRESHOLD``; the counts describe the corpus the artifacts
    were fitted on so the provenance sidecar can say what the cloud contains.
    """

    baseline_response_threshold: float
    row_count: int
    feature_width: int
    key_phrase_count: int


def build(corpus_path: Path = BASELINE_CORPUS_PATH) -> BaselineBuildResult:
    """Rebuild and write all bundled baseline artifacts at the current width.

    Returns a :class:`BaselineBuildResult` whose threshold is the recalibrated
    ``BASELINE_RESPONSE_THRESHOLD``. Callers that write the value out
    (``scripts/retrain_chatgpt_baseline.py``) use the return; running this module
    directly just prints it for a hand-copy.
    """
    import shap
    from sklearn.ensemble import IsolationForest

    texts = _baseline_assistant_texts(corpus_path)
    if not texts:
        raise ValueError(
            f"No assistant replies found in {corpus_path} — refusing to fit baseline "
            "artifacts on an empty corpus."
        )
    print(f"Loaded {len(texts)} baseline assistant replies from {corpus_path.name}")

    # Self-discover the baseline's own signature phrases so the key_phrase_rate
    # column is measured against the SAME phrase set the rows will be scored under
    # at baseline-build time (there is no avatar phrase set here). Persist the
    # phrase list for transparency/reproducibility.
    baseline_key_phrases = [
        phrase["phrase"] for phrase in discover_key_phrases(texts)
    ]
    _KEY_PHRASES_PATH.write_text(
        json.dumps(baseline_key_phrases, indent=2), encoding="utf-8"
    )
    print(
        f"Discovered {len(baseline_key_phrases)} baseline key phrases -> "
        f"{_display_path(_KEY_PHRASES_PATH)}"
    )

    feature_rows = [
        [features[name] for name in FEATURE_NAMES]
        for features in (
            extract_style_features(text, key_phrases=baseline_key_phrases)
            for text in texts
        )
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
    print(f"Wrote {_display_path(_ARR_PATH)}")

    model = IsolationForest(random_state=_ISOLATION_FOREST_RANDOM_STATE).fit(
        feature_matrix
    )
    model_b64 = base64.b64encode(pickle.dumps(model))
    _MODEL_PATH.write_bytes(model_b64)
    print(f"Wrote {_display_path(_MODEL_PATH)}")

    # Pre-build and persist the SHAP explainer so the runtime loads it instead of
    # rebuilding a KernelExplainer (kmeans + repeated model.predict) on first use.
    # Background = kmeans centroids of the matrix, same shape the runtime expects.
    background = (
        shap.kmeans(feature_matrix, _SHAP_BACKGROUND_SIZE)
        if feature_matrix.shape[0] > _SHAP_BACKGROUND_SIZE
        else feature_matrix
    )
    explainer = shap.KernelExplainer(model.predict, background)
    explainer_b64 = base64.b64encode(pickle.dumps(explainer))
    _EXPLAINER_PATH.write_bytes(explainer_b64)
    print(f"Wrote {_display_path(_EXPLAINER_PATH)}")

    # Remove the pre-move orphan so there is a single source of truth.
    if _STALE_EXPLAINER_PATH.exists():
        _STALE_EXPLAINER_PATH.unlink()
        print(f"Removed stale {_display_path(_STALE_EXPLAINER_PATH)}")

    # Recalibrate BASELINE_RESPONSE_THRESHOLD: the Tukey upper fence
    # (Q3 + 1.5*IQR) of the leave-one-out squared-Mahalanobis empirical
    # distribution over the baseline matrix — the same calibration
    # recompute_ground_truth_artifacts uses for the per-avatar cloud. The
    # Mahalanobis scale shifts whenever the feature-vector width changes, so this
    # must be re-derived on every rebuild and copied into the
    # BASELINE_RESPONSE_THRESHOLD env var (.env / .env.dev) and the
    # GlobalContext.baseline_response_threshold default.
    empirical_distribution = compute_empirical_distribution(feature_matrix)
    q3 = np.percentile(empirical_distribution, 75)
    q1 = np.percentile(empirical_distribution, 25)
    baseline_response_threshold = float(q3 + 1.5 * (q3 - q1))
    print(
        f"Recalibrated BASELINE_RESPONSE_THRESHOLD = {baseline_response_threshold!r} "
        "(update .env / .env.dev and the GlobalContext default to this value)"
    )
    return BaselineBuildResult(
        baseline_response_threshold=baseline_response_threshold,
        row_count=int(feature_matrix.shape[0]),
        feature_width=int(feature_matrix.shape[1]),
        key_phrase_count=len(baseline_key_phrases),
    )


if __name__ == "__main__":
    build()
