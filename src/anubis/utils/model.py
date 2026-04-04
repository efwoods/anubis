# src/anubis/utils/model

import logging
logger = logging.getLogger(__name__)

from src.anubis.utils.context import GlobalContext
from typing import Optional

from pydantic import BaseModel
from langchain_together import ChatTogether
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

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

    # from langchain_openai import ChatOpenAI

    if model_provider == "TOGETHER":
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

# from together import Together

# client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")

# response = client.chat.completions.create(
#     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
#     messages=[
#       {
#         "role": "user",
#         "content": "What are some fun things to do in New York?"
#       }
#     ]
# )