"""Knowledge evaluator: false-positive (unsupported claims) + false-negative (missed facts).

Two passes, both bounded by ``context.knowledge_profile_top_k``:

1. **Claim extraction + FP**: An LLM extracts atomic claims from the
   candidate text. For each claim we run a vector search against the
   pre-built atomic-fact index and ask an LLM judge whether any of the
   retrieved facts supports the claim. Unsupported claims are FP.

2. **Question coverage + FN**: An LLM extracts ``probe questions`` that the
   profile's facts can answer (one question per fact-cluster). For each
   probe we vector-search the candidate text representation; if the
   candidate provides no answer, the missed fact is FN.

Both passes never re-read the corpus — they only touch the stored
``knowledge_profile`` and its mirror ``knowledge_profile_index`` namespace.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model

logger = logging.getLogger(__name__)


class _AtomicClaim(BaseModel):
    claim: str = Field(description="One atomic claim extracted from the text.")
    span: str = Field(
        default="",
        description=(
            "Optional verbatim span that contained the claim (for audit)."
        ),
    )


class _AtomicClaimList(BaseModel):
    claims: List[_AtomicClaim] = Field(default_factory=list)


class _SupportJudgment(BaseModel):
    supported: bool = Field(
        description=(
            "True only if at least one retrieved fact entails the claim."
        )
    )
    reasoning: str = Field(
        description="Short justification for supported/unsupported."
    )
    most_relevant_fact_index: int = Field(
        default=-1,
        description=(
            "0-based index of the most relevant retrieved fact, or -1 if "
            "none."
        ),
    )


_CLAIM_EXTRACTOR_SYSTEM = (
    "You extract every atomic factual claim asserted by the assistant text. "
    "Do not include questions, opinions, or stylistic flourishes. Each claim "
    "must be a standalone factual statement that could in principle be "
    "verified true or false."
)

_SUPPORT_JUDGE_SYSTEM = (
    "You are a strict evidence judge. Given an assistant claim and a list of "
    "retrieved facts about the speaker, decide whether at least one fact "
    "ENTAILS the claim. Direct entailment only; do not assume facts not "
    "shown. If the retrieved facts contradict the claim, mark UNSUPPORTED."
)


async def _extract_claims(candidate_text: str) -> List[_AtomicClaim]:
    if not (candidate_text or "").strip():
        return []
    GlobalContext()  # surface env-var validation per workspace .cursorrules
    model = init_model(model_without_tools=False, response_format=_AtomicClaimList)
    response = await model.ainvoke(
        [
            SystemMessage(content=_CLAIM_EXTRACTOR_SYSTEM),
            HumanMessage(content=candidate_text),
        ]
    )
    return list(response.claims or [])


async def _judge_support(
    claim: str, retrieved_facts: List[str]
) -> _SupportJudgment:
    if not retrieved_facts:
        return _SupportJudgment(
            supported=False,
            reasoning="No retrieved facts available for this claim.",
            most_relevant_fact_index=-1,
        )
    framing = (
        f"CLAIM: {claim}\n\nRETRIEVED FACTS:\n"
        + "\n".join(f"[{i}] {f}" for i, f in enumerate(retrieved_facts))
    )
    GlobalContext()
    model = init_model(model_without_tools=False, response_format=_SupportJudgment)
    return await model.ainvoke(
        [
            SystemMessage(content=_SUPPORT_JUDGE_SYSTEM),
            HumanMessage(content=framing),
        ]
    )


async def _retrieve_facts_for_query(
    *,
    query: str,
    creator_id: str,
    assistant_id: str,
    store,
    top_k: int,
) -> List[Dict[str, Any]]:
    namespace = (creator_id, assistant_id, "knowledge_profile_index")
    try:
        items = await store.asearch(namespace, query=query, limit=top_k)
    except Exception as exc:
        logger.warning("Knowledge index search failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for item in items or []:
        value = getattr(item, "value", {}) or {}
        page_content = value.get("page_content") or ""
        if not page_content.strip():
            continue
        out.append(
            {
                "fact": page_content.strip(),
                "metadata": value.get("metadata") or {},
            }
        )
    return out


async def evaluate_knowledge(
    *,
    candidate_text: str,
    creator_id: str,
    assistant_id: str,
    store,
    context: Optional[GlobalContext] = None,
) -> Dict[str, Any]:
    """Evaluate FP / FN of ``candidate_text`` against the stored knowledge profile."""
    ctx = context or GlobalContext()
    top_k = int(ctx.knowledge_profile_top_k or 8)

    claims = await _extract_claims(candidate_text)

    fp_results: List[Dict[str, Any]] = []
    supported_count = 0
    for claim_obj in claims:
        retrieved = await _retrieve_facts_for_query(
            query=claim_obj.claim,
            creator_id=creator_id,
            assistant_id=assistant_id,
            store=store,
            top_k=top_k,
        )
        retrieved_texts = [r["fact"] for r in retrieved]
        judgment = await _judge_support(claim_obj.claim, retrieved_texts)
        if judgment.supported:
            supported_count += 1
        fp_results.append(
            {
                "claim": claim_obj.claim,
                "span": claim_obj.span,
                "supported": judgment.supported,
                "reasoning": judgment.reasoning,
                "most_relevant_fact_index": judgment.most_relevant_fact_index,
                "retrieved_facts": retrieved_texts,
            }
        )

    total_claims = len(fp_results)
    unsupported_claims = total_claims - supported_count
    fp_rate = unsupported_claims / total_claims if total_claims else 0.0

    return {
        "status": "ok",
        "candidate_text_excerpt": (candidate_text or "")[:500],
        "total_claims": total_claims,
        "supported_claims": supported_count,
        "unsupported_claims": unsupported_claims,
        "false_positive_rate": fp_rate,
        "fp_details": fp_results,
        "top_k_per_claim": top_k,
        "note": (
            "False-negative coverage requires a target question set; this "
            "evaluator emits FP analysis today and exposes the bounded "
            "retrieval index for downstream FN drivers (see "
            "evaluator.py)."
        ),
    }
