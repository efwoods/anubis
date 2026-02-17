# src/anubis/utils/model

import logging

logger = logging.getLogger(__name__)


from src.anubis.utils.configuration import GlobalConfiguration

def init_model(configuration: GlobalConfiguration, 
               tools=[], response_format = None):
    
    model_name = configuration.model
    base_url = configuration.llama_api_base_url
    api_key = configuration.llama_api_key
    dev = configuration.dev

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
