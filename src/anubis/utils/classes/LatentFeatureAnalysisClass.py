"""Modular latent-feature analyzer for the psycho-analysis pipeline.

This is the narrative-feature counterpart of
:class:`~src.anubis.utils.classes.FactRewriterClass.FactRewriterClass`. It
scans a chunk of (target-focused) source text for ONE latent psychological
feature — a belief, value, opinion, goal, want, need, fear, flaw,
relationship, or a piece of description / identity / history — and emits
analysis Documents that can be indexed into the ``analysis`` store namespace
and prompt-injected at inference time.

Design goals
------------
* **Modular.** One class, instantiated once per feature with a distinct
  system prompt. Wire any ``(feature_name, system_prompt)`` pair to scan for
  any feature; the structured-output schema is generic and shared.
* **Provenance is appended in Python**, never invented by the model — exactly
  like ``RewrittenFactsWithProvenance`` in :mod:`FactRewriterClass`. The
  verbatim ``original_statement``, the caller-supplied ``target_name``, and a
  ``concise_context_summary`` (built by the same context-summary model the
  fact rewriter uses) are stitched on after the model returns.
* **Output is vectorstore-acceptable analysis.** The produced Documents have
  already been through the analysis stage of the graph, so they are tagged
  ``namespace="analysis"``, ``vectorstore_acceptable=True``,
  ``adapter_acceptable=False``, ``analysis_acceptable=False``. Their
  ``page_content`` follows the ``<FACT_CONTEXT_AND_FACT>`` structure so the
  embedded text carries both the finding and its grounding context.
"""

import asyncio
import logging
from datetime import datetime, timezone
from time import time_ns
from typing import Any, Dict, List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.classes.FactRewriterClass import (
    ConciseContextOfTheSourceOfFacts,
)
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.concise_context_summary_prompt import (
    CONCISE_CONTEXT_SUMMARY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def format_analysis_input_with_context(
    statement: str, situational_context: str | None
) -> str:
    """Wrap a target statement with optional situational context for analysis.

    When ``situational_context`` is empty/None the statement is returned
    unchanged (backward-compatible — analyzers that receive isolated text behave
    exactly as before). When context is supplied (the whole-scene summary and/or
    the preceding "user" turn the target is responding to), it is rendered as a
    clearly-labelled block so the analyzer model interprets the target statement
    in situ while still extracting features ONLY about the target. The caller
    keeps the raw ``statement`` for provenance (``original_statement``); only the
    model input is wrapped.
    """
    statement = (statement or "").strip()
    ctx = (situational_context or "").strip()
    if not ctx:
        return statement
    return (
        "<SITUATIONAL_CONTEXT>\n"
        + ctx
        + "\n</SITUATIONAL_CONTEXT>\n\n<TARGET_STATEMENT>\n"
        + statement
        + "\n</TARGET_STATEMENT>\n\n"
        "Analyze ONLY the target's statement above. Use the situational context "
        "solely to interpret what the target means; do not extract features "
        "about anyone other than the target."
    )


class ExtractedLatentFeature(BaseModel):
    """One atomic latent feature detected about the target — model output."""

    feature_statement: str = Field(
        description=(
            "A single, first-person statement expressing the detected feature "
            "about the target (e.g. the belief, value, opinion, goal, want, "
            "need, fear, flaw, relationship, or descriptive trait). Written "
            "as the target would express it about themselves. Capture only "
            "what the source supports; do not invent or embellish."
        )
    )
    supporting_reason: str = Field(
        description=(
            "The reason drawn from the source material that supports this "
            "finding — the evidence and the overall context behind why the "
            "target holds or exhibits this feature. Grounded strictly in the "
            "source; no new facts."
        )
    )


class ExtractedLatentFeatureList(BaseModel):
    """Structured-output schema the analyzer model is constrained to return."""

    features: List[ExtractedLatentFeature] = Field(
        default_factory=list,
        description=(
            "All atomic instances of the requested feature found about the "
            "target. Empty list if the source contains none."
        ),
    )


class LatentFeatureAnalysisClass:
    """Scan source text for one latent feature and emit analysis Documents.

    Parameters
    ----------
    feature_name:
        Short key for the feature being scanned (e.g. ``"belief"``,
        ``"relationship"``). Used as the metadata key holding the finding and
        as the ``feature`` tag on every produced Document.
    system_prompt:
        Feature-specific system prompt. May contain a ``{target_name}``
        placeholder, which is formatted in at call time (mirrors
        ``FACT_REWRITER_SYSTEM_PROMPT``).
    model_inference_type:
        Label recorded for token/cost metrics. Defaults to
        ``"<feature_name>_analysis_structured_output"``.
    """

    def __init__(
        self,
        feature_name: str,
        system_prompt: str,
        model_inference_type: str | None = None,
    ):
        """Build an analyzer for one feature from a feature name and prompt."""
        self.feature_name = feature_name
        self.system_prompt = system_prompt
        self.model = init_model(response_format=ExtractedLatentFeatureList)
        self.model_inference_type = (
            model_inference_type or f"{feature_name}_analysis_structured_output"
        )

    @staticmethod
    def _format_page_content(
        concise_context_summary: str,
        feature_statement: str,
        supporting_reason: str | None = None,
    ) -> str:
        """Wrap a finding in the ``<FACT_CONTEXT_AND_FACT>`` structure.

        When ``supporting_reason`` is supplied it is folded into the context
        block — *in addition to* the concise context summary — so the
        embedded/searchable ``page_content`` carries the evidence behind the
        finding. Because this same text is what gets embedded and stored, the
        supporting reason then participates both in the dedup similarity search
        (verifying a finding does not already exist) and in the stored fact.
        """
        context = (concise_context_summary or "").strip()
        reason = (supporting_reason or "").strip()
        if reason:
            context = f"{context} {reason}".strip()
        return (
            "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>"
            + context
            + "</FACT_CONTEXT><FACT>"
            + feature_statement.strip()
            + "</FACT></FACT_CONTEXT_AND_FACT>"
        )

    def _format_system_prompt(self, target_name: str | None) -> str:
        """Format ``{target_name}`` into the prompt when present."""
        try:
            return self.system_prompt.format(target_name=target_name)
        except (KeyError, IndexError):
            # Prompt has no/odd format fields; use as-is.
            return self.system_prompt

    async def analyze(
        self,
        input_str: str,
        target_name: str | None = None,
        source_metadata: Dict[str, Any] | None = None,
        situational_context: str | None = None,
    ) -> List[Document]:
        """Detect ``feature_name`` instances in ``input_str``.

        Runs the analyzer model and the concise-context-summary model in
        parallel, then builds one analysis Document per detected feature with
        provenance stitched on in Python.

        ``source_metadata`` carries forward identifying fields from the source
        Document (``user_id``, ``assistant_id``, ``filename``,
        ``filename_uuid5``, ``classified_situation``, ...) so the produced
        analysis Documents stay attributable.
        """
        text = (input_str or "").strip()
        if not text:
            return []

        start_time = time_ns()
        source_metadata = source_metadata or {}

        analyzer_messages = [
            SystemMessage(content=self._format_system_prompt(target_name)),
            HumanMessage(
                content=format_analysis_input_with_context(
                    text, situational_context
                )
            ),
        ]
        summary_model = init_model(response_format=ConciseContextOfTheSourceOfFacts)
        summary_messages = [
            SystemMessage(
                content=CONCISE_CONTEXT_SUMMARY_SYSTEM_PROMPT.format(
                    target_name=target_name
                )
            ),
            HumanMessage(content=text),
        ]

        response, summary_response = await asyncio.gather(
            self.model.ainvoke(analyzer_messages),
            summary_model.ainvoke(summary_messages),
        )

        concise_context_summary = summary_response.concise_context_summary or ""
        current_timestamp = datetime.now(tz=timezone.utc).isoformat()

        documents: List[Document] = []
        for feature in response.features or []:
            statement = (feature.feature_statement or "").strip()
            if not statement:
                continue
            page_content = self._format_page_content(
                concise_context_summary, statement, feature.supporting_reason
            )
            metadata: Dict[str, Any] = {
                **source_metadata,
                self.feature_name: statement,
                "feature": self.feature_name,
                "supporting_reason": feature.supporting_reason,
                "concise_context_summary": concise_context_summary,
                "original_statement": text,
                "target_name": target_name,
                "namespace": "analysis",
                "vectorstore_acceptable": True,
                "adapter_acceptable": False,
                "analysis_acceptable": False,
                "synthetic": True,
                "created_at": current_timestamp,
                "document_id": str(uuid4()),
                "processing_task_id": str(uuid4()),
                "model_inference_type": self.model_inference_type,
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

        logger.info(
            "LatentFeatureAnalysisClass[%s]: %d findings in %.1f ms",
            self.feature_name,
            len(documents),
            (time_ns() - start_time) / 1e6,
        )
        return documents
