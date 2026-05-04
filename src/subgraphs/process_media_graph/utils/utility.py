import logging
from time import time_ns
from typing import Optional

from src.anubis.utils.prompts.system_prompts import (
    MULTI_IMAGE_PROMPT,
    TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION,
)

logger = logging.getLogger(__name__)

from src.anubis.utils.context import GlobalContext
from langchain_core.documents import Document
from src.anubis.utils.model import init_image_description_model
from langchain_core.messages import SystemMessage

from langgraph.store.base import BaseStore


def _sniff_image_mime_from_bytes(chunk: bytes) -> Optional[str]:
    if not chunk:
        return None
    if chunk[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if chunk[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if chunk[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(chunk) >= 12 and chunk[:4] == b"RIFF" and chunk[8:12] == b"WEBP":
        return "image/webp"
    return None

async def extract_personality_from_image(
    image_data: str,
    reference_image: bool,
    filename: str,
    store: BaseStore,
    user_id: str,
    assistant_id: str,
    context: GlobalContext = GlobalContext(),
) -> Document:
    start_time = time_ns()
    """Extract personality description from image using vision LLM."""
    logger.info(
        "extract_personality_from_image entrypoint (reference from store if present)"
    )

    model = init_image_description_model()

    assistant_reference_image_identity_namespace = (
        user_id,
        assistant_id,
        "reference_image",
    )
    key = assistant_id
    reference_image_item = await store.aget(
        assistant_reference_image_identity_namespace, key
    )
    reference_image_data = None
    if reference_image_item:
        reference_image_data = getattr(reference_image_item, "value", {}).get(
            "reference_image_data", None
        )

    if reference_image_data and not reference_image:
        image_to_target_textual_description_payload = [
            SystemMessage(content=MULTI_IMAGE_PROMPT),
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": reference_image_data}},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ]
    else:
        image_to_target_textual_description_payload = [
            SystemMessage(
                content=TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
            ),
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_data}}],
            },
        ]

    input_payload = image_to_target_textual_description_payload

    response = await model.ainvoke(input=input_payload)

    logger.info(f"response: {response}")

    contextual_description = getattr(response, "content")
    if not contextual_description:
        response = await model.ainvoke(input=input_payload)
        logger.info(f"response: {response}")
        contextual_description = getattr(response, "content")
        if not contextual_description:
            raise ValueError("No content found in response")

    response_dict = response.model_dump()

    input_tokens = response_dict["response_metadata"]["token_usage"]["prompt_tokens"]
    output_tokens = response_dict["response_metadata"]["token_usage"][
        "completion_tokens"
    ]
    total_tokens = input_tokens + output_tokens

    total_cost = (
        input_tokens * context.image_model_prompt_cost
        + output_tokens * context.image_model_completion_cost
    )
    model_name = response_dict.get("response_metadata", {}).get("model_name", None)

    end_time = time_ns()
    latency_ms = (end_time - start_time) / 1e6
    logger.info(f"latency_ms: {latency_ms} ms")

    doc = Document(
        page_content=contextual_description,
        metadata={
            "source": filename,
            "inference_type": "image_description",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model_name": model_name,
            "latency_ms": latency_ms,
        },
    )
    return doc
