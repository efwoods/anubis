
from typing import Optional
import hashlib
import json
import time
from src.anubis.utils.prompts.system_prompts import MULTI_IMAGE_PROMPT, TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION

from time import time_ns

import logging
logger = logging.getLogger(__name__)

# region agent log
_DEBUG_LOG_PATH = "/home/user/gh/anubis/.cursor/debug-a51747.log"
_DEBUG_SESSION_ID = "a51747"


def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _url_fingerprint(url: str, n: int = 400) -> str:
    sample = (url or "")[:n].encode("utf-8", errors="replace")
    return hashlib.sha256(sample).hexdigest()[:16]

# endregion

from src.anubis.utils.context import GlobalContext
from langchain_core.documents import Document
from src.anubis.utils.model import init_image_description_model
from langchain_core.messages import SystemMessage


async def extract_personality_from_image(
    image_data: str, 
    filename: str, 
    store: str, 
    user_id: str, 
    assistant_id: str, 
    context: GlobalContext
    ) -> Document:
    start_time = time_ns()
    """Extract personality description from image using vision LLM."""
    logger.info(f"needs reference image from storage for target identification (possibly object bounding box of the target)")
    # base64_image = self._image_to_base64(image_source)
    # base64_image = self.image_to_base64(image_path)

    logger.info(f"extract_personality_from_image entrypoint")    
    # Use reference image to help identify the person in the image.

    from src.anubis.utils.prompts.system_prompts import TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
    
    # TODO: response_metrics_aggregation
    model = init_image_description_model()

    # use reference image if available to target individual
    assistant_reference_image_identity_namespace = (user_id, assistant_id, "reference_image")
    key=assistant_id
    reference_image_item = await store.aget(assistant_reference_image_identity_namespace, key)
    reference_image_data = None
    if reference_image_item:
        reference_image_data = getattr(reference_image_item,'value', {}).get("reference_image_data", None)
        if reference_image_data:
            # Use reference image to help identify the person in the image.
            image_to_target_textual_description_payload = [
                            SystemMessage(content=MULTI_IMAGE_PROMPT),
                            {"role": "user", 
                            "content": [{
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{reference_image_data}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{image_data}"
                                },
                            },
                        ]}]
    if not reference_image_data or not reference_image_data:
            # No reference image data available, use the image data to help identify the person in the image.
            image_to_target_textual_description_payload = [
                    SystemMessage(content=TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION),
                    {"role": 
                    "user", 
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_data}"
                        },
                    }                ]},
                ]
    input = image_to_target_textual_description_payload

    # region agent log
    first_msg = input[0]
    uses_multi = isinstance(first_msg, SystemMessage) and getattr(first_msg, "content", None) == MULTI_IMAGE_PROMPT
    user_block = input[1] if len(input) > 1 else {}
    parts = user_block.get("content") if isinstance(user_block, dict) else None
    n_images = len(parts) if isinstance(parts, list) else 0
    ref_fp = _url_fingerprint(parts[0]["image_url"]["url"]) if uses_multi and n_images >= 1 else None
    cur_fp = _url_fingerprint(parts[-1]["image_url"]["url"]) if n_images else None
    _agent_debug_log(
        "H1",
        "utility.py:extract_personality_pre_invoke",
        "vision payload shape",
        {
            "filename": filename,
            "uses_multi_image_prompt": uses_multi,
            "n_user_images": n_images,
            "ref_url_fp": ref_fp,
            "subject_url_fp": cur_fp,
            "had_reference_in_store": bool(reference_image_data),
        },
    )
    _agent_debug_log(
        "H3",
        "utility.py:extract_personality_pre_invoke",
        "branch",
        {"uses_multi": uses_multi, "reference_loaded": bool(reference_image_data)},
    )
    # endregion

    response = await model.ainvoke(input=input)

    logger.info(f"response: {response}")

    contextual_description = getattr(response, 'content')
    if not contextual_description:
            response = await model.ainvoke(input=input)
            logger.info(f"response: {response}")
            contextual_description = getattr(response, 'content')
            if not contextual_description:
                raise ValueError("No content found in response")

    # region agent log
    _agent_debug_log(
        "H2",
        "utility.py:extract_personality_post_invoke",
        "model response stats",
        {
            "filename": filename,
            "output_char_len": len(contextual_description or ""),
            "is_target_not_visible": (contextual_description or "").strip() == "TARGET_NOT_VISIBLE",
        },
    )
    # endregion

    response_dict = response.model_dump()
    
    input_tokens = response_dict['response_metadata']['token_usage']['prompt_tokens']
    output_tokens = response_dict['response_metadata']['token_usage']['completion_tokens']
    total_tokens = input_tokens + output_tokens
    
    total_cost = input_tokens * context.image_model_prompt_cost + output_tokens * context.image_model_completion_cost
    model_name = response_dict.get('response_metadata', {}).get('model_name', None)

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
        }
    )
    return doc

