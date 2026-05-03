import json
from time import time_ns

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION,
    ReferenceDocumentOrBiographicalConversationalInformation,
)


class ReferenceDocumentClassificationClass:
    """Structured reference vs biographical classification (menus, holy texts, etc.)."""

    def __init__(self):
        context = GlobalContext()
        self.model = init_model(
            response_format=ReferenceDocumentOrBiographicalConversationalInformation
        )
        self.system_message = SystemMessage(
            content=REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION
        )
        self.system_prompt_tokens = count_tokens_approximately(
            REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION
        )
        self.model_name = context.classification_model
        self.model_input_token_cost_per_million = float(
            context.classification_model_prompt_cost
        )
        self.model_output_token_cost_per_million = float(
            context.classification_model_completion_cost
        )
        self.model_inference_type = "reference_document_structured_output"

    async def classify(self, input_str: str):
        start_time = time_ns()
        human_message = HumanMessage(content=input_str)
        input_tokens = (
            count_tokens_approximately(input_str) + self.system_prompt_tokens
        )
        messages = [self.system_message, human_message]
        response = await self.model.ainvoke(messages)
        response_dict = response.model_dump()

        output_tokens = count_tokens_approximately(json.dumps(response_dict))
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
