"""Third-person → first-person identity rewriter.

Pipeline shape:
1. Caller passes a list of lawsuit-safer ``rewritten_statement`` strings
   produced by :class:`FactRewriterClass`.
2. For each input string, this class fires a separate model call whose
   message list is ``[system_message, HumanMessage(content=stmt)]`` — one
   statement per call, never a batch. All calls run concurrently via
   :func:`asyncio.gather` so wall-clock latency is roughly one model call.
3. Each model reply is a single :class:`FirstPersonStatement` (one field,
   ``first_person_statement``).
4. AFTER all calls return, Python pairs each first-person output with its
   originating third-person statement to build a list of
   :class:`FirstPersonStatementWithProvenance`, returned inside a
   :class:`FirstPersonStatementsWithProvenance` container along with token
   / cost / latency metadata. Provenance is stitched in code so the LLM
   cannot alter it.

Each ``FirstPersonStatementWithProvenance`` becomes one ``Document`` in
the vectorstore with the full provenance chain (``original_statement``,
``rewritten_statement``, ``first_person_statement``, ``synthetic=True``,
``original_source``).
"""

import asyncio
import json
from time import time_ns
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from src.anubis.utils.tokenizer import count_tokens
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.first_person_rewriter_prompt import (
    FIRST_PERSON_REWRITER_SYSTEM_PROMPT,
)

class FirstPersonStatement(BaseModel):
    """Single first-person rewrite of one input statement — model output.

    What it represents
        The complete structured-output schema the LLM is constrained to
        return for ONE call. Each call sees the system prompt plus a
        human message containing exactly one third-person biographical
        statement about the TARGET, and emits one first-person sentence
        the target could plausibly say about themselves.

    Rewriting rules (mirroring
    :data:`FIRST_PERSON_REWRITER_SYSTEM_PROMPT`):
        * Use first-person pronouns ("I", "my", "me"). Use "we" only when
          the source clearly refers to a group that includes the target.
        * Preserve every fact in the input — numbers, dates, names,
          places, relationships, professions, and qualifiers
          ("approximately", "around", "reportedly") — exactly as written.
          No additions, no omissions, no softening.
        * Do not shift tense or aspect in ways that change meaning
          (e.g. "was born" must not become "am born").
        * Do not invent motivation, feelings, or back-story that are not
          in the source.
        * If the input contains multiple distinct facts, keep them all
          inside the single ``first_person_statement`` field (split into
          multiple sentences within that one field if it reads more
          naturally).
        * If the input is not actually about the target, return the input
          verbatim; the downstream pipeline filters those out by
          comparing the output to the input.

    How it is used
        * Set as ``response_format`` on the LLM in
          :meth:`FirstPersonRewriterClass.__init__`, so every parallel
          per-statement call is validated against this schema.
        * Returned by ``model.ainvoke`` inside
          :meth:`FirstPersonRewriterClass.rewrite`; the
          ``first_person_statement`` field is then paired with its
          originating third-person statement inside a
          :class:`FirstPersonStatementWithProvenance`.
    """

    first_person_statement: str = Field(
        description=(
            "The same content rephrased in first person. Preserves every "
            "fact, no additions, no omissions."
        )
    )

class FirstPersonStatementWithProvenance(BaseModel):
    """One first-person statement paired in code with its original input."""

    first_person_statement: str = Field(
        description="Model output: the statement rephrased in first person.",
    )
    original_statement: str = Field(
        description=(
            "The third-person statement that was passed into the rewriter. "
            "Appended in code AFTER the model call to preserve provenance."
        ),
    )


class FirstPersonStatementsWithProvenance(BaseModel):
    """Post-call container produced by :class:`FirstPersonRewriterClass`.

    Built by Python AFTER all parallel per-statement model calls return,
    by zipping each input statement with its corresponding model output.
    The model never sees or writes ``original_statement``.
    """

    statements: List[FirstPersonStatementWithProvenance] = Field(
        default_factory=list,
        description=(
            "First-person rewrites paired with the originating third-person "
            "statement, in input order."
        ),
    )


class FirstPersonRewriterClass:
    """Convert third-person biographical statements into first-person identity.

    One model call per input statement, fanned out concurrently with
    :func:`asyncio.gather`. Each call sees only the system prompt and a
    human message containing exactly that one statement, so the model can
    never confuse facts across statements or batch-process them.
    """

    # Sentinel substituted into the prompt when the caller passes an empty
    # context summary. The prompt's <escape_hatches> tells the model to
    # default to literal modality in this case, but we still want a
    # human-readable placeholder rather than a blank section.
    _EMPTY_CONTEXT_PLACEHOLDER = (
        "(no context summary provided; treat the source as literal "
        "modality and apply no modality wrap.)"
    )

    def __init__(self):
        self.model = init_model(response_format=FirstPersonStatement)
        self.system_prompt_template = FIRST_PERSON_REWRITER_SYSTEM_PROMPT
        self.system_prompt_tokens = 2300
        self.model_name = "gpt-5.4-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "first_person_rewriter_structured_output"

    def _build_system_message(
        self, concise_context_summary: str
    ) -> SystemMessage:
        """Format the prompt template with the per-call context summary.

        A new ``SystemMessage`` is built per ``rewrite()`` call (not per
        statement, since the context is shared across the batch) so we
        never mutate shared instance state. Concurrent ``rewrite()``
        calls from different requests therefore cannot race on the
        system message contents.
        """
        summary = (concise_context_summary or "").strip()
        if not summary:
            summary = self._EMPTY_CONTEXT_PLACEHOLDER
        formatted_prompt = self.system_prompt_template.format(
            concise_context_summary=summary
        )
        return SystemMessage(content=formatted_prompt)

    async def _rewrite_one(
        self, statement: str, system_message: SystemMessage
    ) -> FirstPersonStatement:
        """One model call: system prompt + human message of a single statement."""
        human_message = HumanMessage(content=statement)
        return await self.model.ainvoke([system_message, human_message])

    async def rewrite(
        self,
        statements: List[str],
        concise_context_summary: str = "",
    ) -> dict:
        """Rewrite each third-person statement into first person, in parallel.

        Fans out one model call per statement via :func:`asyncio.gather`.
        Each call's input is exactly ``[system_message, HumanMessage(stmt)]``
        — never a batch framing. The ``concise_context_summary`` is
        formatted into the system prompt once per ``rewrite()`` call (it
        is shared across the batch since all facts came from the same
        source text). The model uses it to detect the source modality
        (dream / memory / imagined / wished / literal) and to apply the
        matching first-person modality wrap; see
        :data:`FIRST_PERSON_REWRITER_SYSTEM_PROMPT` for the rules.

        After all calls return, each output is paired with its
        originating input to form a
        :class:`FirstPersonStatementWithProvenance`, all collected
        inside a :class:`FirstPersonStatementsWithProvenance` container.
        The dump is augmented with aggregated token / cost / latency
        metadata so the caller can record it.
        """
        start_time = time_ns()
        system_message = self._build_system_message(concise_context_summary)

        cleaned_statements: List[str] = [
            (s or "").strip() for s in (statements or [])
        ]
        cleaned_statements = [s for s in cleaned_statements if s]

        if not cleaned_statements:
            return {
                "statements": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_name": self.model_name,
                "total_cost": 0.0,
                "model_inference_type": self.model_inference_type,
                "latency_ms": 0.0,
                "model_calls": 0,
            }

        model_outputs: List[FirstPersonStatement] = await asyncio.gather(
            *[
                self._rewrite_one(s, system_message)
                for s in cleaned_statements
            ]
        )

        provenanced_statements: List[FirstPersonStatementWithProvenance] = []
        per_call_input_tokens = 0
        per_call_output_tokens = 0
        for original, model_stmt in zip(cleaned_statements, model_outputs):
            provenanced_statements.append(
                FirstPersonStatementWithProvenance(
                    first_person_statement=model_stmt.first_person_statement,
                    original_statement=original,
                )
            )
            per_call_input_tokens += (
                count_tokens(original) + self.system_prompt_tokens
            )
            per_call_output_tokens += count_tokens(
                json.dumps(model_stmt.model_dump())
            )

        provenanced = FirstPersonStatementsWithProvenance(
            statements=provenanced_statements
        )
        response_dict = provenanced.model_dump()

        total_tokens = per_call_input_tokens + per_call_output_tokens
        total_cost = (
            per_call_input_tokens * self.model_input_token_cost_per_million
            + per_call_output_tokens * self.model_output_token_cost_per_million
        )

        response_dict.update(
            {
                "input_tokens": per_call_input_tokens,
                "output_tokens": per_call_output_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name,
                "total_cost": total_cost,
                "model_inference_type": self.model_inference_type,
                "latency_ms": (time_ns() - start_time) / 1e6,
                "model_calls": len(cleaned_statements),
            }
        )
        return response_dict
