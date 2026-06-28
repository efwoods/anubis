"""Structured-output classifier judging useful content vs boilerplate fragments."""

import json
from time import time_ns

from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    USEFUL_CONTENT_CLASSIFICATION_SYSTEM_PROMPT,
    UsefulContentClassification,
)
from src.anubis.utils.tokenizer import count_tokens


class UsefulContentClassificationClass:
    """LLM fallback for the fragment filter: useful content vs boilerplate noise.

    Only invoked on borderline chunks the cheap heuristic cannot decide, so the
    per-run cost stays small. Mirrors the structured-output classifier pattern
    used elsewhere (see ``ReferenceDocumentClassificationClass``).
    """

    def __init__(self):
        """Initialize the model, prompt, and pricing/metric metadata."""
        self.model = init_model(response_format=UsefulContentClassification)
        self.system_message = SystemMessage(
            content=USEFUL_CONTENT_CLASSIFICATION_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = count_tokens(
            USEFUL_CONTENT_CLASSIFICATION_SYSTEM_PROMPT
        )
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "useful_content_structured_output"

    async def classify(self, input_str: str):
        """Return the usefulness verdict dict (with token/cost/latency metrics)."""
        start_time = time_ns()
        human_message = HumanMessage(content=input_str)
        input_tokens = count_tokens(input_str) + self.system_prompt_tokens
        messages = [self.system_message, human_message]
        response = await self.model.ainvoke(messages)
        response_dict = response.model_dump()

        output_tokens = count_tokens(json.dumps(response_dict))
        total_tokens = input_tokens + output_tokens

        total_cost = (
            input_tokens * self.model_input_token_cost_per_million
            + output_tokens * self.model_output_token_cost_per_million
        )

        response_dict.update({"input_tokens": input_tokens})
        response_dict.update({"output_tokens": output_tokens})
        response_dict.update({"total_tokens": total_tokens})
        response_dict.update({"model_name": self.model_name})
        response_dict.update({"total_cost": total_cost})
        response_dict.update({"model_inference_type": self.model_inference_type})
        end_time = time_ns()
        duration_ms = (end_time - start_time) / 1e6
        response_dict.update({"latency_ms": duration_ms})
        return response_dict
