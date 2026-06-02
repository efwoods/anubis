"""Psycho-analysis methods and the modular analyzer registry.

Houses the structured-output analyzers (OCEAN, emotional triggers, and the
narrative latent-feature analyzers) and the single
:data:`ANALYSIS_SCAFFOLD_RUNNERS` registry that ``analyze_documents`` fans out.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.classes.LatentFeatureAnalysisClass import (
    LatentFeatureAnalysisClass,
)
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.psycho_analysis.emotional_trigger_analysis_prompt import (
    EMOTIONAL_TRIGGER_ANALYSIS_SYSTEM_PROMPT,
    EmotionalTriggerAnalysis,
)
from src.anubis.utils.prompts.psycho_analysis.latent_feature_analysis_prompts import (
    BELIEFS_ANALYSIS_SYSTEM_PROMPT,
    DESCRIPTION_ANALYSIS_SYSTEM_PROMPT,
    FEARS_ANALYSIS_SYSTEM_PROMPT,
    FLAWS_ANALYSIS_SYSTEM_PROMPT,
    GOALS_ANALYSIS_SYSTEM_PROMPT,
    HISTORY_ANALYSIS_SYSTEM_PROMPT,
    IDENTITY_ANALYSIS_SYSTEM_PROMPT,
    NEEDS_ANALYSIS_SYSTEM_PROMPT,
    OPINIONS_ANALYSIS_SYSTEM_PROMPT,
    RELATIONSHIPS_ANALYSIS_SYSTEM_PROMPT,
    VALUES_ANALYSIS_SYSTEM_PROMPT,
    WANTS_ANALYSIS_SYSTEM_PROMPT,
)
from src.anubis.utils.prompts.psycho_analysis.OCEAN_analysis import (
    AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION,
    AGREEABLENESS_OCEAN_ANALYSIS_PROMPT,
    CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION,
    CONSCIENTIOUSNESS_OCEAN_ANALYSIS_PROMPT,
    EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION,
    EXTRAVERSION_OCEAN_ANALYSIS_PROMPT,
    NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION,
    NEUROTICISM_OCEAN_ANALYSIS_PROMPT,
    OPENNESS_OCEAN_ANALYSIS_EXTRACTION,
    OPENNESS_OCEAN_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)

""" OCEAN ANALYSIS """


async def perform_ocean_analysis(
    human_message: HumanMessage,
    additional_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Analyze a human message and return five OCEAN-trait Documents.

    Ingests a human message containing content to analyze and returns a list of
    Documents containing the five results of the analysis.
    """
    logger.info("breakpoint")
    # TODO: CALCULATE TOKEN USAGE response['response_metadata'], latency_ms, cost

    # TODO: response_metrics_aggregation
    openness_model_ocean_analysis = init_model(
        model_without_tools=False, response_format=OPENNESS_OCEAN_ANALYSIS_EXTRACTION
    )
    # TODO: response_metrics_aggregation
    conscientiousness_model_ocean_analysis = init_model(
        model_without_tools=False,
        response_format=CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION,
    )
    # TODO: response_metrics_aggregation
    extraversion_model_ocean_analysis = init_model(
        model_without_tools=False,
        response_format=EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION,
    )
    # TODO: response_metrics_aggregation
    agreeableness_model_ocean_analysis = init_model(
        model_without_tools=False,
        response_format=AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION,
    )
    # TODO: response_metrics_aggregation
    neuroticism_model_ocean_analysis = init_model(
        model_without_tools=False, response_format=NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION
    )

    openness_ocean_analysis_prompt = SystemMessage(
        content=OPENNESS_OCEAN_ANALYSIS_PROMPT
    )
    conscientiousness_ocean_analysis_prompt = SystemMessage(
        content=CONSCIENTIOUSNESS_OCEAN_ANALYSIS_PROMPT
    )
    extraversion_ocean_analysis_prompt = SystemMessage(
        content=EXTRAVERSION_OCEAN_ANALYSIS_PROMPT
    )
    agreeableness_ocean_analysis_prompt = SystemMessage(
        content=AGREEABLENESS_OCEAN_ANALYSIS_PROMPT
    )
    neuroticism_ocean_analysis_prompt = SystemMessage(
        content=NEUROTICISM_OCEAN_ANALYSIS_PROMPT
    )

    (
        openness_ocean_analysis_results,
        conscientiousness_ocean_analysis_results,
        extraversion_ocean_analysis_results,
        agreeableness_ocean_analysis_results,
        neuroticism_ocean_analysis_results,
    ) = await asyncio.gather(
        openness_model_ocean_analysis.ainvoke(
            [openness_ocean_analysis_prompt, human_message]
        ),
        conscientiousness_model_ocean_analysis.ainvoke(
            [conscientiousness_ocean_analysis_prompt, human_message]
        ),
        extraversion_model_ocean_analysis.ainvoke(
            [extraversion_ocean_analysis_prompt, human_message]
        ),
        agreeableness_model_ocean_analysis.ainvoke(
            [agreeableness_ocean_analysis_prompt, human_message]
        ),
        neuroticism_model_ocean_analysis.ainvoke(
            [neuroticism_ocean_analysis_prompt, human_message]
        ),
    )

    results = [
        Document(
            page_content=openness_ocean_analysis_results.openness_description,
            metadata={
                "trait": "openness",
                "score": openness_ocean_analysis_results.openness,
                "reasons": openness_ocean_analysis_results.openness_reasons_and_examples,
                "processing_task_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            id=str(uuid4()),
        ),
        Document(
            page_content=conscientiousness_ocean_analysis_results.conscientiousness_description,
            metadata={
                "trait": "conscientiousness",
                "score": conscientiousness_ocean_analysis_results.conscientiousness,
                "reasons": conscientiousness_ocean_analysis_results.conscientiousness_reasons_and_examples,
                "processing_task_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            id=str(uuid4()),
        ),
        Document(
            page_content=extraversion_ocean_analysis_results.extraversion_description,
            metadata={
                "trait": "extraversion",
                "score": extraversion_ocean_analysis_results.extraversion,
                "reasons": extraversion_ocean_analysis_results.extraversion_reasons_and_examples,
                "processing_task_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            id=str(uuid4()),
        ),
        Document(
            page_content=agreeableness_ocean_analysis_results.agreeableness_description,
            metadata={
                "trait": "agreeableness",
                "score": agreeableness_ocean_analysis_results.agreeableness,
                "reasons": agreeableness_ocean_analysis_results.agreeableness_reasons_and_examples,
                "processing_task_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            id=str(uuid4()),
        ),
        Document(
            page_content=neuroticism_ocean_analysis_results.neuroticism_description,
            metadata={
                "trait": "neuroticism",
                "score": neuroticism_ocean_analysis_results.neuroticism,
                "reasons": neuroticism_ocean_analysis_results.neuroticism_reasons_and_examples,
                "processing_task_id": str(uuid4()),
                "document_id": str(uuid4()),
            },
            id=str(uuid4()),
        ),
    ]

    if additional_metadata:
        for doc in results:
            doc.metadata.update(additional_metadata)

    return results


""" MODULAR ANALYSIS FRAMEWORK

Every analyzer is an ``async (doc: Document) -> List[Document]`` callable
registered in :data:`ANALYSIS_SCAFFOLD_RUNNERS`. ``analyze_documents``
(``src/subgraphs/process_media_graph/utils/nodes.py``) fans every applicable
analyzer out over every ``analysis_acceptable`` Document in parallel, gathers
the produced Documents, and merges them into the vector-store index batch.

Every analyzer output is an ``analysis``-namespace Document
(``vectorstore_acceptable=True``, ``adapter_acceptable=False``,
``analysis_acceptable=False``) whose ``page_content`` follows the
``<FACT_CONTEXT_AND_FACT>`` structure.

To add a feature: write a system prompt, then either
  * narrative feature → add a ``LatentFeatureAnalysisClass`` entry to
    ``_NARRATIVE_ANALYZERS`` (one line), or
  * bespoke framework → write a ``perform_*`` function + a thin runner,
and register it in ``ANALYSIS_SCAFFOLD_RUNNERS``.
"""

# Metadata keys that are queue-routing only — stripped from analysis outputs so
# the produced Documents aren't re-queued for analysis or mis-flagged.
_ANALYSIS_QUEUE_METADATA_KEYS = frozenset(
    {
        "analysis_scaffolds",
        "analysis_job_kind",
        "analysis_acceptable",
        "vectorstore_acceptable",
        "adapter_acceptable",
    }
)


def metadata_for_analysis_outputs(doc: Document) -> dict[str, Any]:
    """Carry source-Document metadata forward to analysis outputs (minus routing keys)."""
    return {
        k: v
        for k, v in (doc.metadata or {}).items()
        if k not in _ANALYSIS_QUEUE_METADATA_KEYS
    }


def _analysis_output_metadata(
    source_meta: dict[str, Any], feature: str, model_inference_type: str
) -> dict[str, Any]:
    """Build the common metadata stamp for analysis-namespace Documents."""
    return {
        **source_meta,
        "feature": feature,
        "namespace": "analysis",
        "vectorstore_acceptable": True,
        "adapter_acceptable": False,
        "analysis_acceptable": False,
        "synthetic": True,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "document_id": str(uuid4()),
        "processing_task_id": str(uuid4()),
        "model_inference_type": model_inference_type,
    }


""" EMOTIONAL TRIGGER ANALYSIS """


async def perform_emotional_trigger_analysis(
    human_message: HumanMessage,
    target_name: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Detect base-six emotion shifts + triggers, corroborated by GoEmotions.

    Returns one ``analysis``-namespace Document per detected shift. Each
    Document also carries the GoEmotions classifier label/score for the source
    text under ``go_emotions`` metadata.
    """
    from src.anubis.utils.emotion_classifier import classify_go_emotions

    content = human_message.content
    text = content.strip() if isinstance(content, str) else content
    if not text:
        return []

    source_metadata = source_metadata or {}

    model = init_model(response_format=EmotionalTriggerAnalysis)
    response = await model.ainvoke(
        [
            SystemMessage(
                content=EMOTIONAL_TRIGGER_ANALYSIS_SYSTEM_PROMPT.format(
                    target_name=target_name
                )
            ),
            human_message,
        ]
    )

    # GoEmotions runs on the CPU-bound transformers pipeline; offload so the
    # event loop isn't blocked while the structured-output calls overlap.
    go_emotions = await asyncio.to_thread(classify_go_emotions, str(text))

    documents: list[Document] = []
    for shift in response.shifts or []:
        statement = (shift.feature_statement or "").strip()
        if not statement:
            continue
        context = (
            f"The target's emotion shifted to '{shift.emotion}', triggered by: "
            f"{shift.trigger_event}"
        )
        page_content = LatentFeatureAnalysisClass._format_page_content(
            context, statement
        )
        metadata = _analysis_output_metadata(
            source_metadata,
            feature="emotional_trigger",
            model_inference_type="emotional_trigger_structured_output",
        )
        metadata.update(
            {
                "emotion": shift.emotion,
                "trigger_event": shift.trigger_event,
                "supporting_reason": shift.supporting_reason,
                "concise_context_summary": context,
                "original_statement": str(text),
                "target_name": target_name,
                "go_emotions": go_emotions,
            }
        )
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


""" ANALYZER RUNNERS + REGISTRY """


# Narrative latent-feature specs: registry key → (statement noun, system prompt).
# Analyzers are built lazily on first use (``init_model`` is not called at import).
_NARRATIVE_ANALYZER_SPECS: dict[str, tuple[str, str]] = {
    "beliefs": ("belief", BELIEFS_ANALYSIS_SYSTEM_PROMPT),
    "relationships": ("relationship", RELATIONSHIPS_ANALYSIS_SYSTEM_PROMPT),
    # Registered stubs (generic prompt body; refine into bespoke prompts later).
    "values": ("value", VALUES_ANALYSIS_SYSTEM_PROMPT),
    "opinions": ("opinion", OPINIONS_ANALYSIS_SYSTEM_PROMPT),
    "goals": ("goal", GOALS_ANALYSIS_SYSTEM_PROMPT),
    "wants": ("want", WANTS_ANALYSIS_SYSTEM_PROMPT),
    "needs": ("need", NEEDS_ANALYSIS_SYSTEM_PROMPT),
    "fears": ("fear", FEARS_ANALYSIS_SYSTEM_PROMPT),
    "flaws": ("flaw", FLAWS_ANALYSIS_SYSTEM_PROMPT),
    "description": ("description", DESCRIPTION_ANALYSIS_SYSTEM_PROMPT),
    "identity": ("identity", IDENTITY_ANALYSIS_SYSTEM_PROMPT),
    "history": ("history", HISTORY_ANALYSIS_SYSTEM_PROMPT),
}

# Lazily-instantiated singletons keyed by registry name.
_NARRATIVE_ANALYZER_CACHE: dict[str, LatentFeatureAnalysisClass] = {}


def _get_narrative_analyzer(name: str) -> LatentFeatureAnalysisClass:
    analyzer = _NARRATIVE_ANALYZER_CACHE.get(name)
    if analyzer is None:
        feature_noun, prompt = _NARRATIVE_ANALYZER_SPECS[name]
        analyzer = LatentFeatureAnalysisClass(feature_noun, prompt)
        _NARRATIVE_ANALYZER_CACHE[name] = analyzer
    return analyzer


def _make_narrative_runner(
    name: str,
) -> Callable[[Document], Awaitable[list[Document]]]:
    async def runner(doc: Document) -> list[Document]:
        analyzer = _get_narrative_analyzer(name)
        return await analyzer.analyze(
            doc.page_content or "",
            target_name=(doc.metadata or {}).get("target_name"),
            source_metadata=metadata_for_analysis_outputs(doc),
        )

    return runner


async def _run_ocean(doc: Document) -> list[Document]:
    text = (doc.page_content or "").strip()
    if not text:
        return []
    source_meta = metadata_for_analysis_outputs(doc)
    raw = await perform_ocean_analysis(HumanMessage(content=text))
    out: list[Document] = []
    for d in raw:
        trait = d.metadata.get("trait", "trait")
        score = d.metadata.get("score")
        # Per OCEAN schema, ``reasons`` holds the first-person trait description;
        # ``page_content`` holds the supporting reasons/examples (the context).
        first_person = (d.metadata.get("reasons") or d.page_content or "").strip()
        context = (
            f"Big Five (OCEAN) {trait} analysis (score {score}): "
            f"{(d.page_content or '').strip()}"
        )
        metadata = _analysis_output_metadata(
            source_meta,
            feature=f"ocean_{trait}",
            model_inference_type="ocean_structured_output",
        )
        metadata.update(
            {
                "trait": trait,
                "score": score,
                "concise_context_summary": context,
                "target_name": source_meta.get("target_name"),
            }
        )
        out.append(
            Document(
                page_content=LatentFeatureAnalysisClass._format_page_content(
                    context, first_person
                ),
                metadata=metadata,
            )
        )
    return out


async def _run_emotional_triggers(doc: Document) -> list[Document]:
    return await perform_emotional_trigger_analysis(
        HumanMessage(content=doc.page_content or ""),
        target_name=(doc.metadata or {}).get("target_name"),
        source_metadata=metadata_for_analysis_outputs(doc),
    )


# Single modular registry: feature name → analyzer runner. Default set runs on
# every analysis-acceptable doc unless a doc narrows it via
# ``metadata["analysis_scaffolds"]``.
ANALYSIS_SCAFFOLD_RUNNERS: dict[str, Any] = {
    "ocean": _run_ocean,
    "emotional_triggers": _run_emotional_triggers,
    **{name: _make_narrative_runner(name) for name in _NARRATIVE_ANALYZER_SPECS},
}
