"""Lawsuit-safe fact extractor + rewriter for biographical source text.

Pipeline shape:
1. Caller passes a chunk of biographical/conversational text plus an optional
   ``target_name`` hint.
2. The class returns a list of ``ExtractedFact`` objects, each with the
   verbatim ``original_statement`` from the source, an atomic
   ``extracted_fact``, a lawsuit-safer ``rewritten_statement``, and a short
   evidence justification used by the audit trail metadata.

Routing wired into ``process_text_to_document``:
* ``classified_situation == "biographical_facts"`` → run this class, then run
  :class:`FirstPersonRewriterClass` on each ``rewritten_statement``, then
  emit one ``identity`` Document per first-person statement.
* Dialogue / multi-speaker → also run this class on the full transcript with
  ``target_name`` set, to extract facts said *about* the target by other
  speakers.
"""

import json
from time import time_ns
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.fact_rewriter_prompt import (
    FACT_REWRITER_SYSTEM_PROMPT,
)


class ExtractedFact(BaseModel):
    """One atomic fact about the target with full provenance."""

    rewritten_statement: str = Field(
        description=(
            "Lawsuit-safer rephrasing of the original statement. Preserves "
            "every fact, materially changes wording and sentence structure."
        )
    )


class ExtractedAndRewrittenFacts(BaseModel):
    """Container for the structured-output response."""

    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description=(
            "All atomic facts extracted from the source text. Empty list if "
            "the source contains no facts about the target."
        ),
    )


class FactRewriterClass:
    """Extract atomic facts and produce lawsuit-safer rewrites of source text."""

    def __init__(self):
        self.model = init_model(response_format=ExtractedAndRewrittenFacts)
        self.system_prompt = FACT_REWRITER_SYSTEM_PROMPT
        self.system_message = SystemMessage(content=FACT_REWRITER_SYSTEM_PROMPT)
        self.system_prompt_tokens = 846
        self.model_name = "gpt-5.4-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million =0.0000004
        self.model_inference_type = "fact_rewriter_structured_output"

    async def extract(
        self, input_str: str, target_name: Optional[str] = None
    ) -> dict:
        """Extract atomic facts about ``target_name`` from ``input_str``.

        Returns the ``ExtractedAndRewrittenFacts`` model dump augmented with
        token / cost / latency metadata so the caller can record it.
        """
        start_time = time_ns()

        framing = (
            f"TARGET: {target_name}\n\n" if target_name else ""
        ) + "SOURCE TEXT:\n\n" + input_str

        human_message = HumanMessage(content=framing)
        input_tokens = (
            count_tokens(framing) + self.system_prompt_tokens
        )
        messages = [self.system_message, human_message]

        response = await self.model.ainvoke(messages)
        response_dict = response.model_dump()

        output_tokens = count_tokens(json.dumps(response_dict))
        total_tokens = input_tokens + output_tokens
        total_cost = (
            input_tokens * self.model_input_token_cost_per_million
            + output_tokens * self.model_output_token_cost_per_million
        )

        response_dict.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name,
                "total_cost": total_cost,
                "model_inference_type": self.model_inference_type,
                "latency_ms": (time_ns() - start_time) / 1e6,
                "target_name": target_name,
            }
        )
        return response_dict
