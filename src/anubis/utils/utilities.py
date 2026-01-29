import os

def init_model():
    # if os.getenv("DEV") == 'TRUE':
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
                model = os.getenv("MODEL"),
                base_url = os.getenv("LLAMA_API_BASE_URL"),
                temperature=0.1,
                api_key = os.environ.get("LLAMA_API_KEY"),
            )
    # else: 
    #     from langchain_together import ChatTogether
    #     model = ChatTogether(model=os.getenv("MODEL"), temperature=0.1)
    return model

