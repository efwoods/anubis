from time import time_ns

from langchain_core.messages import SystemMessage

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_image_description_model
from src.anubis.utils.schema import DESCRIBE_IMAGE_PROMPT


class ImageDescriptionClass:
    """Vision model call for plain-text image descriptions and usage metadata."""

    def __init__(self):
        context = GlobalContext()
        self.model = init_image_description_model()
        self.system_message = SystemMessage(content=DESCRIBE_IMAGE_PROMPT)
        self.model_name = context.image_model
        self.model_input_token_cost = float(context.image_model_prompt_cost)
        self.model_output_token_cost = float(context.image_model_completion_cost)
        self.model_inference_type = "image_description"

    async def describe(self, image_data: str, filename: str) -> dict:
        start_time = time_ns()
        payload = [
            self.system_message,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{image_data}"},
                    }
                ],
            },
        ]

        response = await self.model.ainvoke(input=payload)
        description = getattr(response, "content", None) or ""
        if not description:
            response = await self.model.ainvoke(input=payload)
            description = getattr(response, "content", None) or ""
            if not description:
                raise ValueError("No content found in response")

        response_dict = response.model_dump()
        token_usage = (
            response_dict.get("response_metadata", {}).get("token_usage") or {}
        )
        input_tokens = int(token_usage.get("prompt_tokens", 0))
        output_tokens = int(token_usage.get("completion_tokens", 0))
        total_tokens = input_tokens + output_tokens

        total_cost = (
            input_tokens * self.model_input_token_cost
            + output_tokens * self.model_output_token_cost
        )
        model_name = response_dict.get("response_metadata", {}).get(
            "model_name", self.model_name
        )

        end_time = time_ns()
        latency_ms = (end_time - start_time) / 1e6

        return {
            "description": description,
            "source": filename,
            "inference_type": self.model_inference_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model_name": model_name,
            "latency_ms": latency_ms,
        }
