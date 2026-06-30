"""Psycho-analysis methods and the modular analyzer registry.

Houses the structured-output analyzers (OCEAN, emotional triggers, and the
narrative latent-feature analyzers) and the single
:data:`ANALYSIS_SCAFFOLD_RUNNERS` registry that ``analyze_documents`` fans out.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.classes.LatentFeatureAnalysisClass import (
    LatentFeatureAnalysisClass,
    format_analysis_input_with_context,
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
from src.anubis.utils.prompts.psycho_analysis.standardized_question_analysis_prompt import (
    STANDARDIZED_QUESTION_ANALYSIS_SYSTEM_PROMPT,
    StandardizedQuestionAnswer,
)

logger = logging.getLogger(__name__)

""" OCEAN ANALYSIS """


async def perform_ocean_analysis(
    human_message: HumanMessage,
    additional_metadata: dict[str, Any] | None = None,
    situational_context: str | None = None,
) -> list[Document]:
    """Analyze a human message and return five OCEAN-trait Documents.

    Ingests a human message containing content to analyze and returns a list of
    Documents containing the five results of the analysis. ``situational_context``
    (spec Step 2), when supplied, frames the target statement with the
    whole-scene summary and the preceding "user" turn.
    """
    if situational_context:
        _ocean_content = human_message.content
        _ocean_text = (
            _ocean_content.strip()
            if isinstance(_ocean_content, str)
            else _ocean_content
        )
        human_message = HumanMessage(
            content=format_analysis_input_with_context(
                str(_ocean_text), situational_context
            )
        )
    logger.info("breakpoint")
    # TODO: CALCULATE TOKEN USAGE response['response_metadata'], latency_ms, cost

    # TODO: response_metrics_aggregation
    openness_model_ocean_analysis = init_model(response_format=OPENNESS_OCEAN_ANALYSIS_EXTRACTION)
    # TODO: response_metrics_aggregation
    conscientiousness_model_ocean_analysis = init_model(
        response_format=CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION,
    )
    # TODO: response_metrics_aggregation
    extraversion_model_ocean_analysis = init_model(response_format=EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION)
    # TODO: response_metrics_aggregation
    agreeableness_model_ocean_analysis = init_model(
        response_format=AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION,
    )
    # TODO: response_metrics_aggregation
    neuroticism_model_ocean_analysis = init_model(
        response_format=NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION
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


def _situational_context_from_doc(doc: Document) -> str | None:
    """Build the per-target situational-context block from a source Document.

    Spec Step 2: a target statement is analyzed using the preceding "user" turn
    (``user_context``, falling back to ``adapter_prompt``) and the one
    whole-scene summary (``scene_summary``) as context. Documents that carry
    neither (e.g. biographical/monologue sources) return ``None`` and are
    analyzed in isolation exactly as before.
    """
    md = doc.metadata or {}
    scene = (md.get("scene_summary") or "").strip()
    user_ctx = (md.get("user_context") or md.get("adapter_prompt") or "").strip()
    parts: list[str] = []
    if scene:
        parts.append(f"Scene summary (the overall situation): {scene}")
    if user_ctx:
        parts.append(
            "The statement the target is responding to (the preceding 'user' "
            "turn): " + user_ctx
        )
    return "\n".join(parts) if parts else None


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
        "created_at": datetime.now(tz=UTC).isoformat(),
        "document_id": str(uuid4()),
        "processing_task_id": str(uuid4()),
        "model_inference_type": model_inference_type,
    }


""" EMOTIONAL TRIGGER ANALYSIS """


async def perform_emotional_trigger_analysis(
    human_message: HumanMessage,
    target_name: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    situational_context: str | None = None,
) -> list[Document]:
    """Detect base-six emotion shifts + triggers, corroborated by GoEmotions.

    Returns one ``analysis``-namespace Document per detected shift. Each
    Document also carries the GoEmotions classifier label/score for the source
    text under ``go_emotions`` metadata. ``situational_context`` (spec Step 2),
    when supplied, frames the target statement with scene + preceding-user
    context; GoEmotions still runs over the raw statement.
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
            HumanMessage(
                content=format_analysis_input_with_context(
                    str(text), situational_context
                )
            ),
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
            context, statement, shift.supporting_reason
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


""" STANDARDIZED QUESTION ANALYSIS """


async def perform_standardized_question_analysis(
    human_message: HumanMessage,
    target_name: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    situational_context: str | None = None,
) -> list[Document]:
    """Search the content for an answer to each standardized identity question.

    Each question in ``ALL_STANDARDIZED_QUESTIONS``
    (``data/standardized_questions.py``) is asked in its own structured-output
    model call that searches the analyzed content for an answer about the
    target — stated directly by the target or inferable from information present
    about the target. The bank is fanned out concurrently (bounded by
    ``GlobalContext().standardized_question_analysis_concurrency``). Returns one
    ``analysis``-namespace Document per question/answer pair found; questions
    with no answer in the content produce no document.

    ``situational_context`` (spec Step 2), when supplied, frames the analyzed
    statement with the whole-scene summary and the preceding "user" turn the
    target was responding to. Provenance (``original_statement``) stays the raw
    statement; only the model input is contextualized.
    """
    from src.anubis.utils.analysis.standardized_questions import ALL_STANDARDIZED_QUESTIONS
    from src.anubis.utils.context import GlobalContext

    content = human_message.content
    text = content.strip() if isinstance(content, str) else content
    if not text:
        return []

    source_metadata = source_metadata or {}

    analyzer_human_message = HumanMessage(
        content=format_analysis_input_with_context(str(text), situational_context)
    )

    # One model instance, reused across every per-question call (stateless).
    model = init_model(response_format=StandardizedQuestionAnswer)
    concurrency = max(1, GlobalContext().standardized_question_analysis_concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def _answer_one(question: str) -> Document | None:
        async with semaphore:
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content=STANDARDIZED_QUESTION_ANALYSIS_SYSTEM_PROMPT.format(
                            target_name=target_name,
                            question=question,
                        )
                    ),
                    analyzer_human_message,
                ]
            )
        if not getattr(response, "answer_found", False):
            return None
        answer = (response.answer or "").strip()
        if not answer:
            return None
        context = f'In response to the question "{question}"'
        page_content = LatentFeatureAnalysisClass._format_page_content(
            context, answer, response.supporting_reason
        )
        metadata = _analysis_output_metadata(
            source_metadata,
            feature="standardized_question",
            model_inference_type="standardized_question_structured_output",
        )
        metadata.update(
            {
                "question": question,
                "answer": answer,
                "supporting_reason": response.supporting_reason,
                "concise_context_summary": context,
                "original_statement": str(text),
                "target_name": target_name,
            }
        )
        return Document(page_content=page_content, metadata=metadata)

    results = await asyncio.gather(
        *(_answer_one(q) for q in ALL_STANDARDIZED_QUESTIONS),
        return_exceptions=True,
    )

    documents: list[Document] = []
    for question, result in zip(ALL_STANDARDIZED_QUESTIONS, results):
        if isinstance(result, Exception):
            logger.warning(
                "standardized_question analyzer: question %r failed: %s; skipping",
                question,
                result,
            )
            continue
        if result is not None:
            documents.append(result)

    return documents


""" ANALYZER RUNNERS + REGISTRY """


# Narrative latent-feature specs: registry key → (statement noun, system prompt).
# Analyzers are built lazily on first use (``init_model`` is not called at import).
_NARRATIVE_ANALYZER_SPECS: dict[str, tuple[str, str]] = {
    "beliefs": ("belief", BELIEFS_ANALYSIS_SYSTEM_PROMPT),
    "relationships": ("relationship", RELATIONSHIPS_ANALYSIS_SYSTEM_PROMPT),
    # DSM-5 disorder indications (screening characterization, not diagnosis).
    # "dsm5": ("dsm5_indication", DSM5_ANALYSIS_SYSTEM_PROMPT),
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
            situational_context=_situational_context_from_doc(doc),
        )

    return runner


async def _run_ocean(doc: Document) -> list[Document]:
    text = (doc.page_content or "").strip()
    if not text:
        return []
    source_meta = metadata_for_analysis_outputs(doc)
    raw = await perform_ocean_analysis(
        HumanMessage(content=text),
        situational_context=_situational_context_from_doc(doc),
    )
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
        situational_context=_situational_context_from_doc(doc),
    )


async def _run_standardized_questions(doc: Document) -> list[Document]:
    return await perform_standardized_question_analysis(
        HumanMessage(content=doc.page_content or ""),
        target_name=(doc.metadata or {}).get("target_name"),
        source_metadata=metadata_for_analysis_outputs(doc),
        situational_context=_situational_context_from_doc(doc),
    )


# Single modular registry: feature name → analyzer runner. Default set runs on
# every analysis-acceptable doc unless a doc narrows it via
# ``metadata["analysis_scaffolds"]``.
ANALYSIS_SCAFFOLD_RUNNERS: dict[str, Any] = {
    "ocean": _run_ocean,
    "emotional_triggers": _run_emotional_triggers,
    "standardized_questions": _run_standardized_questions,
    **{name: _make_narrative_runner(name) for name in _NARRATIVE_ANALYZER_SPECS},
}
