# src/anubis/utils/model

import logging
logger = logging.getLogger(__name__)

from src.anubis.utils.context import GlobalContext
from typing import Optional

""" TODO: Prevent Rate Limiting and Token Limiting Errors and Handle Message Failures """

def init_model(context: Optional[GlobalContext] = GlobalContext(), 
               tools=[], tool_choice={}, response_format = None):
    
    context = GlobalContext()
    model_name = context.model
    base_url = context.llama_api_base_url
    api_key = context.llama_api_key
    dev = context.dev

    logger.info(f"dev: {dev}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    # if dev == 'TRUE':
    from langchain_openai import ChatOpenAI
    
    if response_format is None:
        model = ChatOpenAI(
                    model = model_name,
                    base_url = base_url,
                    temperature=0.1,
                    top_p=0.1,
                    api_key = api_key,
<<<<<<< HEAD
                ).bind_tools(tools=tools, tool_choice="auto") # zero or more tools
=======
                ).bind_tools(
                    method='json_schema', 
                    tools=tools, 
                    tool_choice="auto", # auto: zero or more tools
                    strict=True, 
                    include_raw=True
                )
>>>>>>> test_bak
    else: 
        model = ChatOpenAI(
            model = model_name,
            base_url = base_url,
            temperature=0.1,
            top_p=0.1,
            api_key = api_key,
        )

        model = model.with_structured_output(schema=response_format, include_raw=False, strict=True)
    # else: 
    #     from langchain_together import ChatTogether
    #     model = ChatTogether(model=model_name, temperature=0.1)
    return model
