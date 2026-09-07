"""Assess a reference image before any emotion media is generated from it.

One vision call per uploaded reference image answers two questions:

* **Subject** — a person, a stylized character, or a non-human image with no
  face — which picks the prompt family (see ``prompts.py``).
* **Moderation risk** — whether the image-to-video vendor's content
  moderation would refuse the rendered result. The vendor moderates the
  *finished* video and charges for the rendering either way, so a reference
  that shows a trademarked character, a weapon, a fighting pose, gore,
  nudity, a minor, or a hate symbol is caught here, before a single call is
  made, and the owner is told what to change instead of paying for refusals.

The assessment is stored beside the reference image (``reference_subject``,
``reference_moderation_risk``, ``reference_moderation_reasons``,
``reference_moderation_advice`` in the ``reference_image`` store value) so a
later regeneration does not classify again. Any failure — no image model
configured, the vendor down, an unparseable answer — falls back to ``person``
with low risk, which is the behaviour that existed before, and is logged
rather than raised: an upload must not fail because a classifier did.
"""

from __future__ import annotations

import logging
from typing import Any

from src.anubis.utils.media_generation.prompts import (
    SUBJECT_NON_HUMAN,
    SUBJECT_PERSON,
    normalize_reference_subject,
)

logger = logging.getLogger(__name__)

SUBJECT_CLASSIFICATION_INFERENCE_TYPE = "reference_subject_classification"

MODERATION_RISK_LOW = "low"
MODERATION_RISK_HIGH = "high"

# What each reason means to the owner, for the sentence that replaces a charge.
MODERATION_REASON_LABELS: dict[str, str] = {
    "trademarked_character": "a well-known trademarked character",
    "weapon": "a weapon",
    "violence": "a fighting or attack pose",
    "gore": "blood or wounds",
    "nudity_or_sexual": "nudity or sexualised posing",
    "minor": "a subject who appears to be a minor",
    "hate_symbol": "a hate symbol",
}


def default_assessment() -> dict[str, Any]:
    """Return the assessment used when the classifier could not run."""
    return {
        "subject": SUBJECT_PERSON,
        "reasoning": "",
        "moderation_risk": MODERATION_RISK_LOW,
        "moderation_reasons": [],
        "moderation_advice": "",
    }


def normalize_assessment(raw: Any) -> dict[str, Any]:
    """Coerce a stored or model-returned assessment into the known shape."""
    assessment = default_assessment()
    if not isinstance(raw, dict):
        return assessment
    assessment["subject"] = normalize_reference_subject(raw.get("subject"))
    assessment["reasoning"] = str(raw.get("reasoning") or "")
    reasons = raw.get("moderation_reasons") or []
    if not isinstance(reasons, (list, tuple)):
        reasons = []
    listed_any_reason = bool(reasons)
    assessment["moderation_reasons"] = [
        str(reason) for reason in reasons if str(reason) in MODERATION_REASON_LABELS
    ]
    # A trademarked character is a recognisable body or face. An image with no
    # face at all (a lens, a panel, a spaceship, an emblem) cannot be one,
    # however famous the film it comes from; the model still reaches for the
    # label on "reminiscent of" grounds, so the rule is applied here.
    if assessment["subject"] == SUBJECT_NON_HUMAN:
        assessment["moderation_reasons"] = [
            reason
            for reason in assessment["moderation_reasons"]
            if reason != "trademarked_character"
        ]
    risk = str(raw.get("moderation_risk") or "").strip().lower()
    # The reasons decide: a listed reason is a high risk whatever the flag
    # says, and a reason dropped by the rule above takes its risk with it. A
    # bare high flag with no reasons at all is still honoured.
    if assessment["moderation_reasons"] or (
        risk == MODERATION_RISK_HIGH and not listed_any_reason
    ):
        assessment["moderation_risk"] = MODERATION_RISK_HIGH
    assessment["moderation_advice"] = str(raw.get("moderation_advice") or "")
    return assessment


def moderation_blocks_generation(assessment: dict[str, Any] | None) -> bool:
    """Whether the assessment says the vendor would refuse the rendered video."""
    if not assessment:
        return False
    return str(assessment.get("moderation_risk") or "").lower() == MODERATION_RISK_HIGH


def moderation_warning(assessment: dict[str, Any] | None) -> str:
    """Build the sentence shown instead of a charge when generation is withheld."""
    assessment = normalize_assessment(assessment)
    reasons = assessment["moderation_reasons"]
    labels = [MODERATION_REASON_LABELS[reason] for reason in reasons]
    if len(labels) > 1:
        found = ", ".join(labels[:-1]) + " and " + labels[-1]
    elif labels:
        found = labels[0]
    else:
        found = "content the vendor refuses"
    advice = assessment["moderation_advice"].strip()
    if not advice:
        advice = (
            "Upload a different reference image: a calm head-and-shoulders "
            "portrait with no weapon, no fighting pose, and no franchise "
            "character."
        )
    return (
        f"Emotion media was not generated and nothing was charged: this reference "
        f"image shows {found}, which xAI's content moderation refuses after "
        f'rendering (and bills for). {advice} Or choose "generate anyway" to '
        "attempt it at your own cost."
    )


def assessment_store_fields(assessment: dict[str, Any]) -> dict[str, Any]:
    """Return the keys kept beside the reference image in the store."""
    assessment = normalize_assessment(assessment)
    return {
        "reference_subject": assessment["subject"],
        "reference_subject_reasoning": assessment["reasoning"],
        "reference_moderation_risk": assessment["moderation_risk"],
        "reference_moderation_reasons": assessment["moderation_reasons"],
        "reference_moderation_advice": assessment["moderation_advice"],
    }


def assessment_from_store_value(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read a stored assessment back, or ``None`` when the image predates it."""
    value = value or {}
    if not value.get("reference_subject") or not value.get("reference_moderation_risk"):
        return None
    return normalize_assessment(
        {
            "subject": value.get("reference_subject"),
            "reasoning": value.get("reference_subject_reasoning"),
            "moderation_risk": value.get("reference_moderation_risk"),
            "moderation_reasons": value.get("reference_moderation_reasons"),
            "moderation_advice": value.get("reference_moderation_advice"),
        }
    )


async def classify_reference_subject(
    reference_image_data_uri: str, context: Any | None = None
) -> dict[str, Any]:
    """Return the assessment for the image: subject plus moderation risk.

    Keys: ``subject`` (always one of ``REFERENCE_SUBJECTS``), ``reasoning``,
    ``moderation_risk`` (``low`` / ``high``), ``moderation_reasons`` (a list
    of ``MODERATION_REASON_LABELS`` keys), ``moderation_advice``. The default
    assessment is returned when the classifier could not run.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_image_description_model
        from src.anubis.utils.prompts.system_prompts import (
            REFERENCE_SUBJECT_CLASSIFICATION_PROMPT,
        )
        from src.anubis.utils.schema import ReferenceSubjectClassification

        model = init_image_description_model().with_structured_output(
            schema=ReferenceSubjectClassification
        )
        messages = [
            SystemMessage(content=REFERENCE_SUBJECT_CLASSIFICATION_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": reference_image_data_uri},
                    }
                ]
            ),
        ]
        response = await model.ainvoke(messages)
        raw = (
            response
            if isinstance(response, dict)
            else {
                "subject": getattr(response, "subject", None),
                "reasoning": getattr(response, "reasoning", None),
                "moderation_risk": getattr(response, "moderation_risk", None),
                "moderation_reasons": getattr(response, "moderation_reasons", None),
                "moderation_advice": getattr(response, "moderation_advice", None),
            }
        )
        assessment = normalize_assessment(raw)
        logger.info(
            "Reference assessed as %s with %s moderation risk %s: %s",
            assessment["subject"],
            assessment["moderation_risk"],
            assessment["moderation_reasons"],
            assessment["reasoning"],
        )
        return assessment
    except Exception as classification_error:  # noqa: BLE001
        logger.warning(
            "Reference assessment failed; assuming %s with low risk: %s",
            SUBJECT_PERSON,
            classification_error,
        )
        return default_assessment()


__all__ = [
    "MODERATION_REASON_LABELS",
    "MODERATION_RISK_HIGH",
    "MODERATION_RISK_LOW",
    "SUBJECT_CLASSIFICATION_INFERENCE_TYPE",
    "assessment_from_store_value",
    "assessment_store_fields",
    "classify_reference_subject",
    "default_assessment",
    "moderation_blocks_generation",
    "moderation_warning",
    "normalize_assessment",
]
