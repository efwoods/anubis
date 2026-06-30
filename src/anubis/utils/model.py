# src/anubis/utils/model

import logging
logger = logging.getLogger(__name__)

from src.anubis.utils.context import GlobalContext
from typing import Optional

from pydantic import BaseModel

# NOTE: ``ChatTogether``, ``ChatNVIDIA``, ``ChatOpenAI``, and ``AsyncLlamaAPIClient``
# are imported lazily inside the branches that use them.  Eagerly importing all four
# at module scope adds ~3-4 s to every cold start of any module that transitively
# imports model.py (notably retrieval_graph.py and graph.py).  Each provider's SDK
# is only needed for its own ``model_provider`` branch, so the chosen provider pays
# its import cost on the first model call; the other three SDKs are never loaded.

from typing import TypedDict



from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Any
from typing import Literal
from pydantic import BaseModel, field_validator, Field
from typing import Optional

from src.anubis.utils.tokenizer import count_tokens
import json 

# TODO: identify all model call token usage


class TokenUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int
class ResponseMetadata(TypedDict):
    model_name: str
    token_usage: TokenUsage


""" TODO: Prevent Rate Limiting and Token Limiting Errors and Handle Message Failures """

def init_model(context: Optional[GlobalContext] = GlobalContext(), 
               tools=[], 
               tool_choice: str = "auto", 
               response_format = None, 
               ):
    
    context = GlobalContext()
    model_name = context.model
    base_url = context.llm_provider_base_url
    api_key = context.llm_provider_api_key
    dev = context.dev
    model_provider = context.model_provider

    logger.info(f"dev: {dev}")
    logger.info(f"api_key: {api_key}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    if response_format is not None:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model = context.classification_model,
                base_url = context.classification_model_base_url,
                temperature=0.1,
                api_key = context.classification_model_api_key,
            )
            model = model.with_structured_output(schema=response_format)
            return model
    
    if model_provider == "OPEN_AI":
        from langchain_openai import ChatOpenAI

        if response_format is None:
            model = ChatOpenAI(
                        model = model_name,
                        base_url = base_url,
                        temperature=0.1,
                        top_p=0.1,
                        api_key = api_key,
                    ).bind_tools(
                        # method='json_schema', 
                        tools=tools, 
                        tool_choice=tool_choice, # auto: zero or more tools
                        # strict=True, # model output will be guaranteed to match the schema
                        # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
                    )
        else: 
            model = ChatOpenAI(
                model = model_name,
                base_url = base_url,
                temperature=0.1,
                top_p=0.1,
                api_key = api_key,
            )
            model = model.with_structured_output(schema=response_format)

    if model_provider == "TOGETHER":
        from langchain_together import ChatTogether

        if response_format is None:
            model = ChatTogether(
                        model = model_name,
                        base_url = base_url,
                        temperature=0.1,
                        top_p=0.1,
                        api_key = api_key,
                    ).bind_tools(
                        # method='json_schema', 
                        tools=tools, 
                        tool_choice=tool_choice, # auto: zero or more tools
                        # strict=True, # model output will be guaranteed to match the schema
                        # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
                    )
        else: 
            model = ChatTogether(
                model = model_name,
                base_url = base_url,
                temperature=0.1,
                top_p=0.1,
                api_key = api_key,
            )
            model = model.with_structured_output(schema=response_format)
    elif model_provider == "NVIDIA":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        if response_format is None:
                model = ChatNVIDIA(
                            model = model_name,
                            temperature=0.1,
                            top_p=0.1,
                            api_key = api_key,
                        ).bind_tools(
                            # method='json_schema', 
                            tools=tools, 
                            tool_choice=tool_choice, # auto: zero or more tools
                            # strict=True, # model output will be guaranteed to match the schema
                            # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
                        )
        else: 
            model = ChatNVIDIA(
                model = model_name,
                temperature=0.1,
                top_p=0.1,
                api_key = api_key,
            )
            model = model.with_structured_output(schema=response_format)
    elif model_provider == "META":
        from langchain_openai import ChatOpenAI

        if response_format is None:
            model = ChatOpenAI(
                            model = model_name,
                            base_url = base_url,
                            temperature=0.1,
                            top_p=0.1,
                            api_key = api_key,
                        ).bind_tools(
                            # method='json_schema', 
                            tools=tools, 
                            tool_choice=tool_choice, # auto: zero or more tools
                            # strict=True, # model output will be guaranteed to match the schema
                            # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
                        )
        else: 
            model = ChatOpenAI(
                model = model_name,
                base_url = base_url,
                temperature=0.1,
                top_p=0.1,
                api_key = api_key,
            )
            model = model.with_structured_output(schema=response_format)
    

    return model


def init_chat_model_unbound(context: Optional[GlobalContext] = None):
    """Return a raw `BaseChatModel` instance for the configured provider, with no tools bound.

    The deep agent (`create_deep_agent`) needs an unbound chat model so it can
    manage tool binding internally via its middleware stack. `init_model`
    always wraps the provider client in `.bind_tools(...)`, which produces a
    `RunnableBinding` rather than a `BaseChatModel`. This helper mirrors the
    provider-routing logic of `init_model` but returns the bare client.
    """
    context = context or GlobalContext()
    model_name = context.model
    base_url = context.llm_provider_base_url
    api_key = context.llm_provider_api_key
    model_provider = context.model_provider

    logger.info(f"init_chat_model_unbound provider={model_provider} model={model_name}")

    if model_provider == "OPEN_AI" or model_provider == "META":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            temperature=0.1,
            top_p=0.1,
            api_key=api_key,
        )

    if model_provider == "TOGETHER":
        from langchain_together import ChatTogether

        return ChatTogether(
            model=model_name,
            base_url=base_url,
            temperature=0.1,
            top_p=0.1,
            api_key=api_key,
        )

    if model_provider == "NVIDIA":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=model_name,
            temperature=0.1,
            top_p=0.1,
            api_key=api_key,
        )

    msg = f"Unsupported MODEL_PROVIDER for unbound chat model: {model_provider!r}"
    raise ValueError(msg)


def init_image_description_model():
    from langchain_openai import ChatOpenAI

    context = GlobalContext()
    model_name = context.image_model
    base_url = context.image_model_base_url
    api_key = context.image_model_api_key
    dev = context.dev
    model_provider = context.model_provider

    logger.info(f"dev: {dev}")
    logger.info(f"api_key: {api_key}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    model = ChatOpenAI(
                model = model_name,
                base_url = base_url,
                temperature=0.1,
                api_key = api_key,
            )
    return model


async def calculate_token_usage_description_model(model_structured_output_response: any, input_str: str):
    from src.anubis.utils.tokenizer import count_tokens    
    class TokenUsage(TypedDict):
        prompt_tokens: int
        total_tokens: int
        completion_tokens: int

    input_tokens = count_tokens(input_str)
    completion_tokens = sum([count_tokens(str(value)) for value in model_structured_output_response.model_dump().values()])
    total_tokens = input_tokens + completion_tokens

    token_usage = TokenUsage(prompt_tokens=input_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return token_usage

