"""Third-person → first-person identity rewriter.

Takes the lawsuit-safer ``rewritten_statement`` produced by
:class:`FactRewriterClass` and converts it into a first-person identity
statement suitable for storage under the ``identity`` namespace.

Each statement becomes one ``Document`` in the vectorstore with the full
provenance chain (`original_statement`, `extracted_fact`,
`rewritten_statement`, `first_person_statement`, `synthetic=True`,
`original_source`).
"""

import json
from time import time_ns
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.prompts.first_person_rewriter_prompt import (
    FIRST_PERSON_REWRITER_SYSTEM_PROMPT,
)

class FirstPersonStatement(BaseModel):
    """One first-person identity statement with provenance."""


    first_person_statement: str = Field(
        description=(
            "The same content rephrased in first person. Preserves every "
            "fact, no additions, no omissions."
        )
    )


class FirstPersonIdentityStatements(BaseModel):
    """Container for the structured-output response."""

    statements: List[FirstPersonStatement] = Field(
        default_factory=list,
        description=(
            "First-person rewrites of every input statement, preserving "
            "input order."
        ),
    )


class FirstPersonRewriterClass:
    """Convert third-person biographical statements into first-person identity."""

    def __init__(self):
        self.model = init_model(response_format=FirstPersonIdentityStatements)
        self.system_prompt = FIRST_PERSON_REWRITER_SYSTEM_PROMPT
        self.system_message = SystemMessage(
            content=FIRST_PERSON_REWRITER_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = 748
        self.model_name = "gpt-5.4-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "first_person_rewriter_structured_output"

    async def rewrite(self, statements: List[str]) -> dict:
        """Rewrite a batch of third-person statements into first person."""
        start_time = time_ns()

        if not statements:
            return {
                "statements": [],
                "input_tokens": self.system_prompt_tokens,
                "output_tokens": 0,
                "total_tokens": self.system_prompt_tokens,
                "model_name": self.model_name,
                "total_cost": 0.0,
                "model_inference_type": self.model_inference_type,
                "latency_ms": 0.0,
            }

        framing = "REWRITE EACH STATEMENT INTO FIRST PERSON:\n\n" + "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(statements)
        )

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
            }
        )
        return response_dict
