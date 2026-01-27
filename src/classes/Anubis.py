import os
from langchain.tools import tool
from langchain.agents import create_agent
from prompts.system_prompt import SYSTEM_PROMPT

class Anubis:
    def __init__(self):
        if os.getenv("DEV") == 'TRUE':
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model = os.getenv("MODEL"),
                base_url = os.getenv("LLAMA_API_BASE_URL"),
                temperature=0.1,
                api_key = os.environ.get("LLAMA_API_KEY"),
            )
        else: 
            from langchain_together import ChatTogether
            llm = ChatTogether(model=os.getenv("MODEL"), temperature=0.1)
        self.llm = llm
        self.tools = []

    def get_agent(self):
        return create_agent(model=self.llm, tools = self.tools, system_prompt=SYSTEM_PROMPT)


