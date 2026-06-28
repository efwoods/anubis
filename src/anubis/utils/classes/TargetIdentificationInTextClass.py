"""Structured-output classifier attributing prose to speakers and the target."""

import json
from time import time_ns

from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    TARGET_IDENTIFICATION_IN_TEXT_SYSTEM_PROMPT,
    TargetIdentificationInText,
)
from src.anubis.utils.tokenizer import count_tokens


class TargetIdentificationInTextClass:
    """Attribute long-form prose to speakers and flag the target's statements.

    Reuses the existing ``TargetIdentificationInText`` schema (Phase 4 / Step 5a).
    Given a body of unstructured prose (Bible passage, script, wiki, article) and
    an optional target name, it returns ``statements: [{speaker, content,
    is_target}]`` — the same speaker-attributed shape ``coalesce_segments_by_speaker``
    and ``process_dialogue_json_to_documents`` already consume from diarized audio.
    This is what lets text/URL/PDF flow into the golden-format pipeline so adapter
    datasets (prompt-completion + multi-turn) get produced per
    ``_OVERALL_PREPROCESSING_PROCESS.md``.
    """

    def __init__(self):
        """Initialize the model, prompt, and pricing/metric metadata."""
        self.model = init_model(response_format=TargetIdentificationInText)
        self.system_message = SystemMessage(
            content=TARGET_IDENTIFICATION_IN_TEXT_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = count_tokens(
            TARGET_IDENTIFICATION_IN_TEXT_SYSTEM_PROMPT
        )
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "target_identification_structured_output"

    async def identify(self, input_str: str, target_name: str | None = None):
        """Return the speaker-attribution dict (statements + metrics) for the text."""
        start_time = time_ns()
        # The target name (when supplied) anchors attribution; otherwise the model
        # best-identifies the primary persona from the text itself.
        if target_name:
            human_content = f"TARGET individual: {target_name}\n\nTEXT:\n{input_str}"
        else:
            human_content = f"TARGET individual: (identify the primary persona)\n\nTEXT:\n{input_str}"
        human_message = HumanMessage(content=human_content)
        input_tokens = count_tokens(human_content) + self.system_prompt_tokens
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
