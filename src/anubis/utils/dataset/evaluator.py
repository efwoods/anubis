"""Evaluator orchestrator: stylistic + knowledge + content-quality, all from stored profiles.

Loads the precomputed stylistic profile and knowledge profile from the
LangGraph store and runs:

* :func:`evaluate_authenticity_against_profile` — Burrows Delta, lexicon
  Jaccard, sentence-length z, syntax POS divergence, style rate gaps.

* :func:`evaluate_knowledge` — atomic-claim FP rate via bounded vector
  retrieval against the precomputed atomic-fact index.

* :func:`evaluate` (existing module) — BERT-score, ROUGE, optional
  LLM-as-judge content-quality scores.

The corpus itself is NEVER touched here; we only read the stored profile
artifacts and query the bounded index. This keeps evaluation O(profile)
rather than O(corpus).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.dataset.authenticity_evaluator import (
    evaluate_authenticity_against_profile,
)
from src.anubis.utils.dataset.build_knowledge_profile import load_knowledge_profile
from src.anubis.utils.dataset.build_profile import load_stylistic_profile
from src.anubis.utils.dataset.knowledge_evaluator import evaluate_knowledge
from src.anubis.utils.dataset.quality import evaluate as evaluate_content_quality

logger = logging.getLogger(__name__)


async def run_evaluation(
    *,
    candidate_text: str,
    creator_id: str,
    assistant_id: str,
    store,
    reference_text: Optional[str] = None,
    use_llm_as_a_judge: bool = False,
    context: Optional[GlobalContext] = None,
) -> Dict[str, Any]:
    """Run the full evaluation suite against precomputed profile artifacts.

    Parameters
    ----------
    candidate_text:
        The model output to evaluate.
    creator_id, assistant_id:
        Identify the avatar whose stored profiles should be used.
    store:
        ``BaseStore`` instance from the LangGraph runtime.
    reference_text:
        Optional ground-truth reference for BERT/ROUGE/LLM-judge metrics.
        If not supplied those metrics are skipped.
    use_llm_as_a_judge:
        Whether to run the LLM-as-judge content-quality scorer (slower /
        more expensive). Off by default.
    context:
        Optional pre-built ``GlobalContext`` (avoids redundant env reads).
    """
    ctx = context or GlobalContext()

    stylistic_profile = await load_stylistic_profile(
        creator_id=creator_id, assistant_id=assistant_id, store=store
    )
    knowledge_profile = await load_knowledge_profile(
        creator_id=creator_id, assistant_id=assistant_id, store=store
    )

    authenticity_report = evaluate_authenticity_against_profile(
        candidate_text, stylistic_profile or {}
    )

    if knowledge_profile is None:
        knowledge_report: Dict[str, Any] = {
            "status": "skipped",
            "reason": "no_knowledge_profile_available",
        }
    else:
        try:
            knowledge_report = await evaluate_knowledge(
                candidate_text=candidate_text,
                creator_id=creator_id,
                assistant_id=assistant_id,
                store=store,
                context=ctx,
            )
        except Exception as exc:
            logger.exception("evaluate_knowledge failed: %s", exc)
            knowledge_report = {"status": "error", "error": str(exc)}

    if reference_text:
        try:
            content_quality_report = await evaluate_content_quality(
                source_text=reference_text,
                generated_response=candidate_text,
                use_llm_as_a_judge=use_llm_as_a_judge,
            )
        except Exception as exc:
            logger.exception("evaluate_content_quality failed: %s", exc)
            content_quality_report = {"status": "error", "error": str(exc)}
    else:
        content_quality_report = {
            "status": "skipped",
            "reason": "no_reference_text_supplied",
        }

    return {
        "creator_id": creator_id,
        "assistant_id": assistant_id,
        "candidate_text_excerpt": (candidate_text or "")[:500],
        "stylistic_profile_built": stylistic_profile is not None,
        "knowledge_profile_built": knowledge_profile is not None,
        "authenticity": authenticity_report,
        "knowledge": knowledge_report,
        "content_quality": content_quality_report,
    }
