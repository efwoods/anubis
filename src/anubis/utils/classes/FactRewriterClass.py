"""Lawsuit-safe fact extractor + rewriter for biographical source text.

Pipeline shape:
1. Caller passes a chunk of biographical/conversational text plus an optional
   ``target_name`` hint.
2. The model returns a list of ``ExtractedFact`` objects, each carrying ONLY
   a lawsuit-safer ``rewritten_statement``. Nothing else is asked of the model
   so the structured-output schema stays minimal.
3. AFTER the model call, the class wraps the response in a
   ``RewrittenFactsWithProvenance`` container that appends the verbatim
   ``original_statement`` (the caller's ``input_str``) and the ``target_name``
   so provenance is added by Python and never invented by the model.

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
from src.anubis.utils.tokenizer import count_tokens
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.fact_rewriter_prompt import (
    FACT_REWRITER_SYSTEM_PROMPT,
)

class ExtractedFact(BaseModel):
    """One atomic fact extracted from biographical source text — model output.

    A single fact extracted from the source text. Given the source text, rewrite the statement. Do not exclude any facts. Inlude all information of the source text in the rewritten statement. Do not create any new facts. Do not change the meaning of the statement.
    """

    rewritten_statement: str = Field(
        description=(
            "A rephrasing of the original statement. A single fact extracted from the source text. Given the source text, rewrite the statement. Do not exclude any facts. Inlude all information of the source text in the rewritten statement. Do not create any new facts. Do not change the meaning of the statement. Preserves every fact, materially changes wording and sentence structure."
        )
    )


class ExtractedAndRewrittenFacts(BaseModel):
    """Structured-output schema the LLM is constrained to return.

    What it represents
        The complete shape of one model reply: a list of
        :class:`ExtractedFact` items, each with only a
        ``rewritten_statement``. 

    How it is used
        * Passed to :func:`init_model` as ``response_format`` in
          :method:`FactRewriterClass.__init__`, which forces the OpenAI-
          compatible client to validate the model reply against this
          schema.
        * Returned by ``model.ainvoke`` inside
          :method:`FactRewriterClass.extract`; its ``facts`` list is then
          forwarded into a :class:`RewrittenFactsWithProvenance` so the
          downstream pipeline gets both the rewrites and the verbatim
          source text in a single object.
    """

    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description=(
            "All atomic facts extracted from the source text. Empty list if "
            "the source contains no facts about the target."
        ),
    )

class RewrittenFactsWithProvenance(BaseModel):
    """Post-call container produced by :class:`FactRewriterClass`.

    Built by Python AFTER the model returns, so provenance is verbatim and
    cannot be invented by the LLM. ``original_statement`` is the caller's
    ``input_str`` and ``target_name`` is the caller-supplied target hint.
    """

    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="The model's lawsuit-safer rewrites, untouched.",
    )
    original_statement: str = Field(
        description=(
            "Verbatim source text the model was given. Appended in code "
            "after the model call to preserve provenance for downstream "
            "metadata."
        ),
    )
    target_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the individual the facts are about. Appended in code "
            "after the model call from the caller-supplied hint."
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

        Returns the dump of a :class:`RewrittenFactsWithProvenance` (built in
        code AFTER the model call from the model's
        :class:`ExtractedAndRewrittenFacts` plus the caller-supplied
        ``input_str`` and ``target_name``) augmented with token / cost /
        latency metadata so the caller can record it.
        """
        start_time = time_ns()

        framing = (
            f"TARGET: {target_name}\n\n" if target_name else ""
        ) + "SOURCE TEXT:\n\n" + input_str

        human_message = HumanMessage(content=framing)
        input_tokens = (
            count_tokens(framing) + self.system_prompt_tokens
        )
        messages = [SystemMessage(content=self.system_message), HumanMessage(content=human_message)]

        response = await self.model.ainvoke(messages)

        provenanced = RewrittenFactsWithProvenance(
            facts=list(response.facts or []),
            original_statement=input_str,
            target_name=target_name,
        )
        response_dict = provenanced.model_dump()

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
