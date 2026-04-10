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
               image_model: Optional[bool] = False, 
               model_without_tools: Optional[bool] = False
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
    if model_without_tools:
        if response_format is None:
            model = AsyncLlamaAPIClientWrapper()
        else:
            model = AsyncLlamaAPIClientWrapper(response_format=response_format)
        return model 
    
    if model_provider == "OPEN_AI":
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

def init_image_description_model():
    context = GlobalContext()
    model_name = context.image_model
    base_url = context.llm_provider_base_url
    api_key = context.llm_provider_api_key
    dev = context.dev
    model_provider = context.model_provider

    logger.info(f"dev: {dev}")
    logger.info(f"api_key: {api_key}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    # from langchain_openai import ChatOpenAI
    model = ChatNVIDIA(
                model = model_name,
                temperature=0.1,
                top_p=0.1,
                api_key = api_key,
            )
    
    # if model_provider == "TOGETHER":
    #     model = ChatTogether(
    #                 model = model_name,
    #                 base_url = base_url,
    #                 temperature=0.1,
    #                 top_p=0.1,
    #                 api_key = api_key,
    #             )

    # elif model_provider == "NVIDIA":
    #         model = ChatNVIDIA(
    #                     model = model_name,
    #                     temperature=0.1,
    #                     top_p=0.1,
    #                     api_key = api_key,
    #                 )
    # elif model_provider == "META":
    #     model = ChatOpenAI(
    #                     model = model_name,
    #                     base_url = base_url,
    #                     temperature=0.1,
    #                     top_p=0.1,
    #                     api_key = api_key,
    #                 )
    return model

from llama_api_client import AsyncLlamaAPIClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Any
from typing import Literal
from pydantic import BaseModel, field_validator, Field
from typing import Optional

from langchain_core.messages.utils import count_tokens_approximately

class AsyncLlamaAPIClientWrapper:
    def __init__(self, response_format = None):
        context = GlobalContext()
        self.llama_api_key = context.llama_api_key
        self.pydantic_model = response_format

    async def ainvoke(self, messages: List[Literal[HumanMessage, SystemMessage, AIMessage, dict]]):
      """ Accept a list of langchain messages and a pydantic_model 
      and formats the messages for use as a model 
      with structured output for analysis 
      or returns an AI message 
      if no pydantic model is accepted
      """
      client = AsyncLlamaAPIClient(api_key=self.llama_api_key)
      class LlamaMessage(BaseModel):
          role: Literal["human","user", "system", "assistant"] = Field(validation_alias="type")
          content: str

          @field_validator('role', mode="before")
          @classmethod
          def map_role(cls, value: str) -> str:
              mapping = {"human": "user", "user":"user", "system":"system", "assistant":"assistant"}
              return mapping.get(value, "user")

      if type(messages[0]) is not dict:
        formatted_messages = [(LlamaMessage.model_validate(message.model_dump()).model_dump()) for message in messages]
      else:
          formatted_messages = messages

      if self.pydantic_model is not None:
          if self.pydantic_model.__name__ == "TextualSituationalAwareness":
              approximate_message_length = count_tokens_approximately(formatted_messages[1]['content'])
              if approximate_message_length > 4000:
                  formatted_messages[1]['content'] = formatted_messages[1]['content'][:4000] # truncate messages for situational analysis classification
      
      if self.pydantic_model is not None:
        response = await client.chat.completions.create(
            messages=formatted_messages,
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            stream=False,
            temperature=0.1,
            # max_completion_tokens=4096,
            top_p=0.1,
            repetition_penalty=1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.pydantic_model.__name__,
                    "schema": self.pydantic_model.model_json_schema()
                }
            }
        )
        return self.pydantic_model.model_validate_json(response.completion_message.content.text)
    
      else:
        response = await client.chat.completions.create(
              messages=formatted_messages,
              model="Llama-4-Maverick-17B-128E-Instruct-FP8",
              stream=False,
              temperature=0.1,
              max_completion_tokens=16000,
              top_p=0.1,
              repetition_penalty=1,
          )
        return AIMessage(content=response.completion_message.content.text)
     