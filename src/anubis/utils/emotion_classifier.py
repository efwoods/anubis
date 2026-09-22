"""Shared GoEmotions sentiment helper.

Wraps the HuggingFace ``SamLowe/roberta-base-go_emotions`` text-classification
pipeline so both the avatar runtime (``src.anubis.graph``) and the
emotional-trigger analyzer (``analysis_methods``) classify text the same way.

The ``transformers`` import is intentionally lazy (inside the function) to keep
module-import cold-start cheap, per the repo's import conventions.

The pipeline is loaded once per process and reused. Building the pipeline
reads the RoBERTa weights from disk and takes seconds, while classifying one
message with an already-loaded pipeline takes tens of milliseconds; building
the pipeline per call put those seconds in front of every avatar reply. The
FastAPI lifespan calls ``warm_go_emotions_classifier`` on a worker thread so
the first reply after a restart does not pay the load either.
"""

import logging
import threading
from typing import Any, Dict

from src.anubis.utils.emotion_mapping import EMOTION_MAPPING
from src.anubis.utils.huggingface_prefetch import GO_EMOTIONS_MODEL_ID

logger = logging.getLogger(__name__)

# One lock guards both loading and inference: the pipeline is loaded exactly
# once even when two threads ask at the same moment, and inference is
# serialized because a HuggingFace pipeline is not documented as safe to call
# from several threads at once. Inference is short, so serializing costs little.
_go_emotions_classifier_lock = threading.Lock()
_go_emotions_classifier: Any = None


def _loaded_go_emotions_classifier() -> Any:
    """Return the process-wide Go Emotions pipeline, loading the pipeline on first use.

    The caller must hold ``_go_emotions_classifier_lock``.
    """
    global _go_emotions_classifier
    if _go_emotions_classifier is None:
        from transformers import pipeline

        _go_emotions_classifier = pipeline(
            "text-classification", model=GO_EMOTIONS_MODEL_ID
        )
    return _go_emotions_classifier


def warm_go_emotions_classifier() -> None:
    """Load the Go Emotions pipeline ahead of the first reply (best effort, never raises)."""
    try:
        with _go_emotions_classifier_lock:
            _loaded_go_emotions_classifier()
    except Exception as warm_error:  # pragma: no cover - defensive
        logger.warning("Go Emotions warm-up failed: %s", warm_error)


def classify_go_emotions(text: str) -> Dict[str, Any] | None:
    """Classify ``text`` into a GoEmotions label mapped to a base emotion.

    Returns ``{"base_emotion", "emotion", "score"}`` or ``None`` when the text
    is empty or the classifier fails (best-effort; never raises).
    """
    if not text or not str(text).strip():
        return None
    try:
        with _go_emotions_classifier_lock:
            classifier = _loaded_go_emotions_classifier()
            sentiment = classifier(str(text), truncation=True, max_length=512)
        label = sentiment[0]["label"]
        return {
            "base_emotion": EMOTION_MAPPING.get(label, label),
            "emotion": label,
            "score": sentiment[0]["score"],
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("classify_go_emotions failed: %s", exc)
        return None
