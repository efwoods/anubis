"""Shared GoEmotions sentiment helper.

Wraps the HuggingFace ``SamLowe/roberta-base-go_emotions`` text-classification
pipeline so both the avatar runtime (``src.anubis.graph``) and the
emotional-trigger analyzer (``analysis_methods``) classify text the same way.

The ``transformers`` import is intentionally lazy (inside the function) to keep
module-import cold-start cheap, per the repo's import conventions.
"""

import logging
from typing import Any, Dict

from src.anubis.utils.emotion_mapping import EMOTION_MAPPING
from src.anubis.utils.huggingface_prefetch import GO_EMOTIONS_MODEL_ID

logger = logging.getLogger(__name__)


def classify_go_emotions(text: str) -> Dict[str, Any] | None:
    """Classify ``text`` into a GoEmotions label mapped to a base emotion.

    Returns ``{"base_emotion", "emotion", "score"}`` or ``None`` when the text
    is empty or the classifier fails (best-effort; never raises).
    """
    if not text or not str(text).strip():
        return None
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model=GO_EMOTIONS_MODEL_ID)
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
