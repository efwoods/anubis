# src/anubis/utils/model

# from src.anubis.utils.tools import search, health_check, add_to_vectorstore, retrieve_from_vectorstore

# tools = [search, health_check, add_to_vectorstore, retrieve_from_vectorstore]

def init_model(provider_model, base_url, api_key, tools, dev):
    provider, model_name = provider_model.split("/", maxsplit=1) 
    if dev == 'TRUE':
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
                    model = model_name,
                    base_url = base_url,
                    temperature=0.1,
                    api_key = api_key,
                ).bind_tools(tools=tools)
    else: 
        from langchain_together import ChatTogether
        model = ChatTogether(model=model_name, temperature=0.1)
    return model
