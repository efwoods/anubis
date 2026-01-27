from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain.agents import create_agent

class Avatar:
    def __init__(self):
        self.model