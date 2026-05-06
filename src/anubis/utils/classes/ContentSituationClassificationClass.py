from src.anubis.utils.schema import ContentSituationClassification, CONTENT_SITUATION_CLASSIFICATION_SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage
from src.anubis.utils.model import init_model
from src.anubis.utils.tokenizer import count_tokens
from time import time_ns
import json
from src.anubis.utils.context import GlobalContext
class ContentSituationClassificationClass():
    def __init__(self):

        self.model = init_model(response_format=ContentSituationClassification)
        self.system_message = SystemMessage(content=CONTENT_SITUATION_CLASSIFICATION_SYSTEM_PROMPT)
        self.system_prompt_tokens = 527
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "model_with_structured_output"

    async def classify(self, input_str: str):
        start_time = time_ns()
        human_message = HumanMessage(content=input_str)
        input_tokens = count_tokens(input_str) + self.system_prompt_tokens
        messages = [self.system_message, human_message]
        response = await self.model.ainvoke(messages)
        response_dict = response.model_dump()

        output_tokens = count_tokens(json.dumps(response_dict))
        total_tokens = input_tokens + output_tokens

        total_cost = input_tokens * self.model_input_token_cost_per_million + output_tokens * self.model_output_token_cost_per_million

        response_dict.update({"input_tokens":input_tokens})
        response_dict.update({"output_tokens":output_tokens})
        response_dict.update({"total_tokens":total_tokens})
        response_dict.update({"model_name":self.model_name})
        response_dict.update({"total_cost":total_cost})
        response_dict.update({"model_inference_type":self.model_inference_type})
        end_time = time_ns()
        duration_ms = (end_time - start_time) / 1e6
        response_dict.update({"latency_ms":duration_ms})
        return response_dict

