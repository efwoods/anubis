"""The registry of psychological dimensions read from uploaded media.

One entry per dimension. Adding a dimension is one prompt plus one line here; no
node, no graph edge, and no state channel changes, because the graph fans the
registry out generically.

Every dimension produces two things from the same model call:

* **Documents** for the store, so the raw finding stays retrievable by similarity
  the way every other analyzed trait is. They carry ``metadata["namespace"]`` and
  are merged into the upload's index batch.
* **A finding dictionary** for :mod:`src.anubis.utils.psycho.profile`, which folds
  it into the avatar's consolidated profile — a confidence-weighted score for a
  graded dimension, a list of statements for a narrative one.

A dimension is one of two kinds. A GRADED dimension scores a fixed, enumerated
trait list; a NARRATIVE dimension extracts however many findings the source
supports. ``dialogue_emotional_triggers`` is the one bespoke case: it needs a
stimulus, a generalized description of the stimulus, and the target's response, so
that a live message can be matched against it at conversation time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.classes.LatentFeatureAnalysisClass import (
    LatentFeatureAnalysisClass,
    format_analysis_input_with_context,
)
from src.anubis.utils.prompts.psycho_analysis.conversation_subtleties_prompt import (
    CONVERSATION_SUBTLETIES_ANALYSIS_SYSTEM_PROMPT,
)
from src.anubis.utils.prompts.psycho_analysis.dialogue_emotional_trigger_prompt import (
    DIALOGUE_EMOTIONAL_TRIGGER_SYSTEM_PROMPT,
    DialogueEmotionalTriggerAnalysis,
)
from src.anubis.utils.prompts.psycho_analysis.love_languages_prompt import (
    LOVE_LANGUAGES_ANALYSIS_SYSTEM_PROMPT,
)
from src.anubis.utils.prompts.psycho_analysis.psychological_dimension_prompts import (
    ATTACHMENT_STYLE_ANALYSIS_SYSTEM_PROMPT,
    COGNITIVE_STYLE_ANALYSIS_SYSTEM_PROMPT,
    CORE_MOTIVATION_ANALYSIS_SYSTEM_PROMPT,
    DARK_TRAIT_ANALYSIS_SYSTEM_PROMPT,
    DEFENSE_MECHANISM_ANALYSIS_SYSTEM_PROMPT,
    EMOTIONAL_BASELINE_ANALYSIS_SYSTEM_PROMPT,
    MORAL_FOUNDATIONS_ANALYSIS_SYSTEM_PROMPT,
    MYERS_BRIGGS_ANALYSIS_SYSTEM_PROMPT,
    PERSONALITY_ARCHETYPE_ANALYSIS_SYSTEM_PROMPT,
    SCHWARTZ_VALUES_ANALYSIS_SYSTEM_PROMPT,
    ScoredPsychologicalDimension,
)
from src.anubis.utils.psycho.profile import (
    DIMENSION_KIND_GRADED,
    DIMENSION_KIND_NARRATIVE,
)

logger = logging.getLogger(__name__)

# The store namespace every dimension writes into, so its raw findings reach the
# ANALYZED TRAITS section of the system prompt by similarity like every other
# analyzed trait. The one exception is the dialogue trigger dimension, which needs
# its own namespace because it is searched at conversation time by a different
# query than the topic of the conversation.
ANALYSIS_NAMESPACE = "analysis"
EMOTIONAL_TRIGGER_NAMESPACE = "emotional_trigger"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _document_metadata(
    source_metadata: dict[str, Any],
    *,
    dimension: str,
    namespace: str,
    target_name: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata every dimension Document carries.

    The routing flags matter: the produced Documents are ``vectorstore_acceptable``
    so they are indexed, and NOT ``analysis_acceptable``, so an analysis output can
    never be queued for analysis again.
    """
    metadata: dict[str, Any] = {
        **source_metadata,
        "feature": dimension,
        "psychological_dimension": dimension,
        "target_name": target_name,
        "namespace": namespace,
        "vectorstore_acceptable": True,
        "adapter_acceptable": False,
        "analysis_acceptable": False,
        "synthetic": True,
        "created_at": _now(),
        "document_id": str(uuid4()),
        "processing_task_id": str(uuid4()),
        "model_inference_type": f"{dimension}_structured_output",
    }
    metadata.update(extra or {})
    return metadata


async def _run_graded_dimension(
    dimension: str,
    system_prompt: str,
    document: Document,
    *,
    target_name: str | None,
    source_metadata: dict[str, Any],
    situational_context: str | None,
) -> tuple[list[Document], dict[str, Any]]:
    """Score one graded dimension over one document."""
    from src.anubis.utils.model import init_model

    text = (document.page_content or "").strip()
    if not text:
        return [], {}

    try:
        prompt = system_prompt.format(target_name=target_name)
    except (KeyError, IndexError):
        prompt = system_prompt

    model = init_model(response_format=ScoredPsychologicalDimension)
    response = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=format_analysis_input_with_context(text, situational_context)
            ),
        ]
    )
    if isinstance(response, tuple):
        response = response[0]

    documents: list[Document] = []
    traits: dict[str, Any] = {}
    for trait in getattr(response, "traits", None) or []:
        trait_name = (getattr(trait, "trait", "") or "").strip()
        if not trait_name:
            continue
        statement = (getattr(trait, "first_person_statement", "") or "").strip()
        evidence = (getattr(trait, "supporting_evidence", "") or "").strip()
        try:
            score = float(getattr(trait, "score", 0.0) or 0.0)
            confidence = float(getattr(trait, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        traits[trait_name] = {
            "score": score,
            "confidence": confidence,
            "statement": statement,
            "evidence": evidence,
        }
        # Only findings the source actually supports become retrievable Documents;
        # a trait scored at zero is recorded in the profile as an absence and would
        # only be noise in the similarity index.
        if statement and score > 0.0:
            context = (
                f"{dimension.replace('_', ' ')} reading for {trait_name} "
                f"(score {score:.2f}, confidence {confidence:.2f}). {evidence}"
            ).strip()
            documents.append(
                Document(
                    page_content=LatentFeatureAnalysisClass._format_page_content(
                        context, statement
                    ),
                    metadata=_document_metadata(
                        source_metadata,
                        dimension=dimension,
                        namespace=ANALYSIS_NAMESPACE,
                        target_name=target_name,
                        extra={
                            "trait": trait_name,
                            "score": score,
                            "confidence": confidence,
                            "supporting_reason": evidence,
                            "concise_context_summary": context,
                        },
                    ),
                )
            )
    finding = {
        "dimension": dimension,
        "kind": DIMENSION_KIND_GRADED,
        "traits": traits,
        "summary": (getattr(response, "summary_statement", "") or "").strip(),
    }
    return documents, finding


async def _run_narrative_dimension(
    dimension: str,
    feature_noun: str,
    system_prompt: str,
    document: Document,
    *,
    target_name: str | None,
    source_metadata: dict[str, Any],
    situational_context: str | None,
) -> tuple[list[Document], dict[str, Any]]:
    """Extract one narrative dimension over one document via the shared analyzer."""
    analyzer = _narrative_analyzer(dimension, feature_noun, system_prompt)
    documents = await analyzer.analyze(
        document.page_content or "",
        target_name=target_name,
        source_metadata=source_metadata,
        situational_context=situational_context,
    )
    statements = [
        {
            "statement": (analyzed.metadata or {}).get(feature_noun)
            or (analyzed.metadata or {}).get("feature_statement", ""),
            "evidence": (analyzed.metadata or {}).get("supporting_reason", ""),
        }
        for analyzed in documents
    ]
    # The analyzer tags its Documents with the feature noun; retag them with the
    # dimension name so a profile section and its raw findings share one label.
    for analyzed in documents:
        analyzed.metadata["feature"] = dimension
        analyzed.metadata["psychological_dimension"] = dimension
    return documents, {
        "dimension": dimension,
        "kind": DIMENSION_KIND_NARRATIVE,
        "statements": [entry for entry in statements if entry["statement"]],
    }


_NARRATIVE_ANALYZER_CACHE: dict[str, LatentFeatureAnalysisClass] = {}


def _narrative_analyzer(
    dimension: str, feature_noun: str, system_prompt: str
) -> LatentFeatureAnalysisClass:
    """Build the analyzer lazily; ``init_model`` must not run at import time."""
    analyzer = _NARRATIVE_ANALYZER_CACHE.get(dimension)
    if analyzer is None:
        analyzer = LatentFeatureAnalysisClass(feature_noun, system_prompt)
        _NARRATIVE_ANALYZER_CACHE[dimension] = analyzer
    return analyzer


async def _run_dialogue_emotional_triggers(
    dimension: str,
    system_prompt: str,
    document: Document,
    *,
    target_name: str | None,
    source_metadata: dict[str, Any],
    situational_context: str | None,
) -> tuple[list[Document], dict[str, Any]]:
    """Find what other people say or do that moves the target, and how the target answers.

    The Documents go to their own namespace rather than to ``analysis`` because
    they are searched at conversation time against the incoming MESSAGE, not
    against the conversation's topic. Their page content leads with the
    generalized trigger description, since the store index embeds page content
    only — a trigger described only in metadata would never be found.
    """
    from src.anubis.utils.model import init_model

    text = (document.page_content or "").strip()
    if not text:
        return [], {}
    try:
        prompt = system_prompt.format(target_name=target_name)
    except (KeyError, IndexError):
        prompt = system_prompt

    model = init_model(response_format=DialogueEmotionalTriggerAnalysis)
    response = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=format_analysis_input_with_context(text, situational_context)
            ),
        ]
    )
    if isinstance(response, tuple):
        response = response[0]

    documents: list[Document] = []
    statements: list[dict[str, str]] = []
    for trigger in getattr(response, "triggers", None) or []:
        description = (getattr(trigger, "trigger_description", "") or "").strip()
        statement = (getattr(trigger, "feature_statement", "") or "").strip()
        if not description or not statement:
            continue
        response_text = (getattr(trigger, "target_response", "") or "").strip()
        emotion = (getattr(trigger, "emotion", "") or "").strip()
        occurrence = (getattr(trigger, "trigger_occurrence", "") or "").strip()
        speaker = (getattr(trigger, "trigger_speaker", "") or "").strip()
        reason = (getattr(trigger, "supporting_reason", "") or "").strip()
        # ONLY the generalized description is embedded. The store's vector index
        # embeds page_content, and this record exists to be matched against an
        # incoming message, so anything else in the page content is noise that
        # drags every trigger toward the same middle distance. Measured on one
        # avatar's triggers: with the emotion, response and restatement folded in,
        # the weakest genuine match scored 0.489 while an unrelated question about
        # the weather scored 0.487 — no threshold could tell them apart. With the
        # description alone the same probes separate 0.586 against 0.538. The rest
        # of the record stays in metadata, where the matcher reads it after the
        # match rather than embedding it.
        page_content = description
        documents.append(
            Document(
                page_content=page_content,
                metadata=_document_metadata(
                    source_metadata,
                    dimension=dimension,
                    namespace=EMOTIONAL_TRIGGER_NAMESPACE,
                    target_name=target_name,
                    extra={
                        "emotion": emotion,
                        "trigger_description": description,
                        "trigger_occurrence": occurrence,
                        "trigger_speaker": speaker,
                        "target_response": response_text,
                        "supporting_reason": reason,
                    },
                ),
            )
        )
        statements.append({"statement": statement, "evidence": reason or occurrence})
    return documents, {
        "dimension": dimension,
        "kind": DIMENSION_KIND_NARRATIVE,
        "statements": statements,
    }


@dataclass(frozen=True)
class PsychologicalDimension:
    """One registered dimension of the psychological profile."""

    name: str
    kind: str
    system_prompt: str
    # The metadata key a narrative analyzer stores its statement under; unused by
    # graded dimensions, which key on the trait names in their own prompt.
    feature_noun: str = ""
    # The name of the GlobalContext flag that must read TRUE for this dimension to
    # run. Empty means the dimension runs whenever psychological analysis runs.
    enabled_flag: str = ""
    # Whether this dimension needs BOTH sides of a conversation rather than the
    # target's words alone. The trait dimensions read target-only quote documents
    # on purpose, so another speaker's values are never attributed to the target.
    # A trigger is the opposite case: what moves the target is precisely what
    # SOMEBODY ELSE said or did, so a target-only document has the stimulus cut
    # out of it and the dimension can only recover what the target self-reports.
    reads_full_dialogue: bool = False
    runner: Callable[..., Awaitable[tuple[list[Document], dict]]] | None = field(
        default=None, compare=False
    )

    async def analyze(
        self,
        document: Document,
        *,
        target_name: str | None,
        source_metadata: dict[str, Any],
        situational_context: str | None,
    ) -> tuple[list[Document], dict[str, Any]]:
        if self.runner is not None:
            return await self.runner(
                self.name,
                self.system_prompt,
                document,
                target_name=target_name,
                source_metadata=source_metadata,
                situational_context=situational_context,
            )
        if self.kind == DIMENSION_KIND_GRADED:
            return await _run_graded_dimension(
                self.name,
                self.system_prompt,
                document,
                target_name=target_name,
                source_metadata=source_metadata,
                situational_context=situational_context,
            )
        return await _run_narrative_dimension(
            self.name,
            self.feature_noun or self.name,
            self.system_prompt,
            document,
            target_name=target_name,
            source_metadata=source_metadata,
            situational_context=situational_context,
        )


PSYCHOLOGICAL_DIMENSIONS: dict[str, PsychologicalDimension] = {
    dimension.name: dimension
    for dimension in (
        PsychologicalDimension(
            name="love_languages",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=LOVE_LANGUAGES_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="dialogue_emotional_triggers",
            kind=DIMENSION_KIND_NARRATIVE,
            system_prompt=DIALOGUE_EMOTIONAL_TRIGGER_SYSTEM_PROMPT,
            runner=_run_dialogue_emotional_triggers,
            reads_full_dialogue=True,
        ),
        PsychologicalDimension(
            name="emotional_baseline",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=EMOTIONAL_BASELINE_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="attachment_style",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=ATTACHMENT_STYLE_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="schwartz_values",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=SCHWARTZ_VALUES_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="moral_foundations",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=MORAL_FOUNDATIONS_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="cognitive_style",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=COGNITIVE_STYLE_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="myers_briggs",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=MYERS_BRIGGS_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="personality_archetypes",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=PERSONALITY_ARCHETYPE_ANALYSIS_SYSTEM_PROMPT,
        ),
        PsychologicalDimension(
            name="defense_mechanisms",
            kind=DIMENSION_KIND_NARRATIVE,
            system_prompt=DEFENSE_MECHANISM_ANALYSIS_SYSTEM_PROMPT,
            feature_noun="defense_mechanism",
        ),
        PsychologicalDimension(
            name="core_motivations",
            kind=DIMENSION_KIND_NARRATIVE,
            system_prompt=CORE_MOTIVATION_ANALYSIS_SYSTEM_PROMPT,
            feature_noun="core_motivation",
        ),
        PsychologicalDimension(
            name="conversation_subtleties",
            kind=DIMENSION_KIND_NARRATIVE,
            system_prompt=CONVERSATION_SUBTLETIES_ANALYSIS_SYSTEM_PROMPT,
            feature_noun="conversation_subtlety",
        ),
        PsychologicalDimension(
            name="dark_traits",
            kind=DIMENSION_KIND_GRADED,
            system_prompt=DARK_TRAIT_ANALYSIS_SYSTEM_PROMPT,
            enabled_flag="enable_dark_trait_analysis",
        ),
    )
}

# The dimension whose scores seed the avatar's baseline emotional temperament.
EMOTIONAL_BASELINE_DIMENSION = "emotional_baseline"


def selected_dimensions(context: Any) -> list[PsychologicalDimension]:
    """The dimensions this deployment runs, honouring the allow-list and the flags."""
    from src.anubis.utils.moderation.content_moderation import moderation_flag_enabled

    requested = (
        getattr(context, "psychological_analysis_dimensions", None) or ""
    ).strip()
    if requested:
        names = [name.strip() for name in requested.split(",") if name.strip()]
        chosen = []
        for name in names:
            dimension = PSYCHOLOGICAL_DIMENSIONS.get(name)
            if dimension is None:
                logger.warning(
                    "psychological_analysis_dimensions names an unknown dimension %r; "
                    "skipping it",
                    name,
                )
                continue
            chosen.append(dimension)
    else:
        chosen = list(PSYCHOLOGICAL_DIMENSIONS.values())
    return [
        dimension
        for dimension in chosen
        if not dimension.enabled_flag
        or moderation_flag_enabled(
            getattr(context, dimension.enabled_flag, "FALSE"), default=False
        )
    ]


__all__ = [
    "ANALYSIS_NAMESPACE",
    "EMOTIONAL_BASELINE_DIMENSION",
    "EMOTIONAL_TRIGGER_NAMESPACE",
    "PSYCHOLOGICAL_DIMENSIONS",
    "PsychologicalDimension",
    "selected_dimensions",
]
