"""Structured-output classifier judging whether a line is the document title.

Safeguard for the fragment filter: when the cheap embedding-similarity check
between an extracted line and the page title lands in the ambiguous band, this
judge decides — by meaning and role rather than lexical overlap — whether the
line is page furniture (the title / a running header restating it) or real body
content that merely shares words with the title.
"""

import json
from time import time_ns

from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    TITLE_FRAGMENT_CLASSIFICATION_SYSTEM_PROMPT,
    TitleFragmentClassification,
)
from src.anubis.utils.tokenizer import count_tokens


class TitleFragmentClassificationClass:
    """LLM safeguard for the fragment filter: title furniture vs real content.

    Only invoked on the ambiguous similarity band the embedding check cannot
    decide, so the per-run cost stays small. Mirrors the structured-output
    classifier pattern used elsewhere (see ``UsefulContentClassificationClass``).
    The ``classify`` method is synchronous because the caller
    (``classify_fragment_heuristic``) runs in a synchronous per-line loop.
    """

    def __init__(self):
        """Initialize the model, prompt, and pricing/metric metadata."""
        self.model = init_model(response_format=TitleFragmentClassification)
        self.system_message = SystemMessage(
            content=TITLE_FRAGMENT_CLASSIFICATION_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = count_tokens(
            TITLE_FRAGMENT_CLASSIFICATION_SYSTEM_PROMPT
        )
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "title_fragment_structured_output"

    def classify(self, fragment: str, title: str):
        """Return the title verdict dict (with token/cost/latency metrics).

        ``title`` is the actual page/document title the line is compared against;
        sending it gives the judge the context the embedding check could not
        resolve on lexical overlap alone.
        """
        start_time = time_ns()
        input_str = f"TITLE:\n{title}\n\nLINE:\n{fragment}"
        human_message = HumanMessage(content=input_str)
        input_tokens = count_tokens(input_str) + self.system_prompt_tokens
        messages = [self.system_message, human_message]
        response = self.model.invoke(messages)
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
