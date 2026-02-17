# src/anubis/utils/model

# from src.anubis.utils.tools import search, health_check, add_to_vectorstore, retrieve_from_vectorstore

# tools = [search, health_check, add_to_vectorstore, retrieve_from_vectorstore]

from pydantic import BaseModel

from src.anubis.utils.configuration import GlobalConfiguration

def init_model(configuration: GlobalConfiguration, 
               tools=[], response_format = None):
    
    provider_model = configuration.provider_model
    base_url = configuration.llama_api_base_url
    api_key = configuration.llama_api_key
    dev = configuration.dev

    logger.info(f"dev: {dev}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"provider_model: {provider_model}")
    
    provider, model_name = provider_model.split("/", maxsplit=1) 


    # if dev == 'TRUE':
    from langchain_openai import ChatOpenAI
    
    if response_format is None:
        model = ChatOpenAI(
                    model = model_name,
                    base_url = base_url,
                    temperature=0.1,
                    api_key = api_key,
                ).bind_tools(tools=tools)
    else: 
        model = ChatOpenAI(
            model = model_name,
            base_url = base_url,
            temperature=0.1,
            api_key = api_key,
        ).bind_tools(tools=tools)
        model = model.with_structured_output(response_format)
    # else: 
    #     from langchain_together import ChatTogether
    #     model = ChatTogether(model=model_name, temperature=0.1)
    return model

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.helper_functions import format_docs
from datetime import datetime, timezone
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime
from langchain.agents import create_agent

import logging

logger = logging.getLogger(__name__)
