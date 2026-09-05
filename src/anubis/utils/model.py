# src/anubis/utils/model

import logging

logger = logging.getLogger(__name__)

import json

# NOTE: ``ChatTogether``, ``ChatNVIDIA``, ``ChatOpenAI``, and ``AsyncLlamaAPIClient``
# are imported lazily inside the branches that use them.  Eagerly importing all four
# at module scope adds ~3-4 s to every cold start of any module that transitively
# imports model.py (notably retrieval_graph.py and graph.py).  Each provider's SDK
# is only needed for its own ``model_provider`` branch, so the chosen provider pays
# its import cost on the first model call; the other three SDKs are never loaded.
from typing import Any, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tokenizer import count_tokens

# Runnable tag applied to every structured-output model (``response_format`` set). The
# streaming layer (``_stream_deep_agent`` in graph.py) uses it to positively exclude these
# internal JSON-producing calls from the user-facing ``assistant_token`` stream — otherwise
# their raw structured output leaks into the chat (e.g. interleaved fact-correction JSON).
STRUCTURED_OUTPUT_STREAM_TAG = "structured_output_no_user_stream"

# Inference models that reject the ``top_p`` sampling parameter: OpenAI answers
# every request with HTTP 400 "Unsupported parameter: 'top_p' is not supported
# with this model" (observed for gpt-5.6-luna on 2026-09-03; ``temperature`` is
# still accepted). Matched by prefix on MODEL so point releases stay covered.
TOP_P_UNSUPPORTED_MODEL_PREFIXES = ("gpt-5.6-luna",)
# Inference models that only accept function tools on the chat-completions
# endpoint when reasoning is switched off: OpenAI answers a tool-bound request
# with HTTP 400 "Function tools with reasoning_effort are not supported for
# gpt-5.6-luna in /v1/chat/completions. To use function tools, use /v1/responses
# or set reasoning_effort to 'none'" (observed 2026-09-03; the default effort,
# and 'low', both fail). Every avatar turn binds the identity tools, so these
# models are always called with reasoning_effort='none'.
REASONING_EFFORT_NONE_MODEL_PREFIXES = ("gpt-5.6-luna",)


def openai_sampling_parameters(model_name: str | None) -> dict[str, Any]:
    """Return the per-model ``ChatOpenAI`` keyword arguments the inference path sends.

    The low-temperature, low-top_p pairing is the sampling regime every avatar
    reply has used; it is kept wherever the model accepts it. For a model in
    :data:`TOP_P_UNSUPPORTED_MODEL_PREFIXES` only ``temperature`` is sent, since
    sending ``top_p`` fails the whole call rather than being ignored; for a model
    in :data:`REASONING_EFFORT_NONE_MODEL_PREFIXES` ``reasoning_effort="none"`` is
    added so tool-bound calls stay on the chat-completions endpoint.
    """
    parameters: dict[str, Any] = {"temperature": 0.1}
    normalized_model_name = (model_name or "").strip()
    if not normalized_model_name.startswith(TOP_P_UNSUPPORTED_MODEL_PREFIXES):
        parameters["top_p"] = 0.1
    if normalized_model_name.startswith(REASONING_EFFORT_NONE_MODEL_PREFIXES):
        parameters["reasoning_effort"] = "none"
    return parameters


def describe_api_key_for_logging(api_key: Optional[str]) -> str:
    """Describe a provider credential without writing the credential itself.

    These log lines exist to answer one question while debugging a provider
    call: was a key configured for this model, and roughly which one. Printing
    the key answered that question and also published a live secret to anyone
    who could read the container logs — ``docker logs``, Grafana, or a support
    bundle. The last four characters are enough to tell two configured keys
    apart, and are not enough to authenticate with.

    :param api_key: The provider credential, or None when none is configured.
    :returns: A description safe to write to the log.
    """
    if not api_key:
        return "not configured"
    return f"configured (ends {api_key[-4:]}, {len(api_key)} characters)"


# TODO: identify all model call token usage


class TokenUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int


class ResponseMetadata(TypedDict):
    model_name: str
    token_usage: TokenUsage


""" TODO: Prevent Rate Limiting and Token Limiting Errors and Handle Message Failures """


def init_model(
    context: Optional[GlobalContext] = GlobalContext(),
    tools=[],
    tool_choice: str = "auto",
    response_format=None,
    model_without_tools: Optional[bool] = False,
):

    context = GlobalContext()
    model_name = context.model
    base_url = context.llm_provider_base_url
    api_key = context.llm_provider_api_key
    dev = context.dev
    model_provider = context.model_provider

    logger.info(f"dev: {dev}")
    logger.info(f"api_key: {describe_api_key_for_logging(api_key)}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    # from langchain_openai import ChatOpenAI
    if model_without_tools:
        if response_format is None:
            model = AsyncLlamaAPIClientWrapper()
        else:
            model = AsyncLlamaAPIClientWrapper(response_format=response_format)
        return model

    if response_format is not None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=context.classification_model,
            base_url=context.classification_model_base_url,
            temperature=0.1,
            api_key=context.classification_model_api_key,
        )
        model = model.with_structured_output(schema=response_format)
        # Tag so the streaming layer never forwards this call's tokens to the user as
        # ``assistant_token`` — structured output is internal JSON, not a reply.
        return model.with_config(tags=[STRUCTURED_OUTPUT_STREAM_TAG])

    if model_provider == "OPEN_AI":
        from langchain_openai import ChatOpenAI

        if response_format is None:
            model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                **openai_sampling_parameters(model_name),
                api_key=api_key,
                # Report token usage on streamed responses for the metering layer.
                stream_usage=True,
            ).bind_tools(
                # method='json_schema',
                tools=tools,
                tool_choice=tool_choice,  # auto: zero or more tools
                # strict=True, # model output will be guaranteed to match the schema
                # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
            )
        else:
            model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                **openai_sampling_parameters(model_name),
                api_key=api_key,
            )
            model = model.with_structured_output(schema=response_format)

    if model_provider == "TOGETHER":
        from langchain_together import ChatTogether

        if response_format is None:
            model = ChatTogether(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                top_p=0.1,
                api_key=api_key,
            ).bind_tools(
                # method='json_schema',
                tools=tools,
                tool_choice=tool_choice,  # auto: zero or more tools
                # strict=True, # model output will be guaranteed to match the schema
                # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
            )
        else:
            model = ChatTogether(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                top_p=0.1,
                api_key=api_key,
            )
            model = model.with_structured_output(schema=response_format)
    elif model_provider == "NVIDIA":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        if response_format is None:
            model = ChatNVIDIA(
                model=model_name,
                temperature=0.1,
                top_p=0.1,
                api_key=api_key,
            ).bind_tools(
                # method='json_schema',
                tools=tools,
                tool_choice=tool_choice,  # auto: zero or more tools
                # strict=True, # model output will be guaranteed to match the schema
                # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
            )
        else:
            model = ChatNVIDIA(
                model=model_name,
                temperature=0.1,
                top_p=0.1,
                api_key=api_key,
            )
            model = model.with_structured_output(schema=response_format)
    elif model_provider == "META":
        from langchain_openai import ChatOpenAI

        if response_format is None:
            model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                **openai_sampling_parameters(model_name),
                api_key=api_key,
            ).bind_tools(
                # method='json_schema',
                tools=tools,
                tool_choice=tool_choice,  # auto: zero or more tools
                # strict=True, # model output will be guaranteed to match the schema
                # include_raw=True # model response (JSON e.g.) and the parsed response (Pydantic e.g.) will be returned
            )
        else:
            model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                **openai_sampling_parameters(model_name),
                api_key=api_key,
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
            **openai_sampling_parameters(model_name),
            api_key=api_key,
            # Include token usage on the final streamed chunk so per-turn
            # usage_metadata reaches the metering layer (Stripe billing meters,
            # api_metrics rows, Prometheus counters). Without stream_options the
            # OpenAI streaming API omits usage entirely. Only set for the real
            # OpenAI endpoint: OpenAI-compatible providers (META/Llama) may
            # reject the stream_options parameter.
            stream_usage=(model_provider == "OPEN_AI"),
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
    logger.info(f"api_key: {describe_api_key_for_logging(api_key)}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=0.1,
        api_key=api_key,
    )
    return model


async def calculate_token_usage_description_model(
    model_structured_output_response: any, input_str: str
):
    from src.anubis.utils.tokenizer import count_tokens

    class TokenUsage(TypedDict):
        prompt_tokens: int
        total_tokens: int
        completion_tokens: int

    input_tokens = count_tokens(input_str)
    completion_tokens = sum(
        [
            count_tokens(str(value))
            for value in model_structured_output_response.model_dump().values()
        ]
    )
    total_tokens = input_tokens + completion_tokens

    token_usage = TokenUsage(
        prompt_tokens=input_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return token_usage


class AsyncLlamaAPIClientWrapper:
    def __init__(self, response_format=None):
        context = GlobalContext()
        self.llama_api_key = context.llama_api_key
        self.pydantic_model = response_format
        self.model_name = context.llama_model

    async def ainvoke(
        self, messages: List[Literal[HumanMessage, SystemMessage, AIMessage, dict]]
    ):
        """Accept a list of langchain messages and a pydantic_model
        and formats the messages for use as a model
        with structured output for analysis
        or returns an AI message with token usage metadata
        if no pydantic model is accepted
        """
        from llama_api_client import AsyncLlamaAPIClient

        client = AsyncLlamaAPIClient(api_key=self.llama_api_key)

        class LlamaMessage(BaseModel):
            role: Literal["human", "user", "system", "assistant"] = Field(
                validation_alias="type"
            )
            content: str

            @field_validator("role", mode="before")
            @classmethod
            def map_role(cls, value: str) -> str:
                mapping = {
                    "human": "user",
                    "user": "user",
                    "system": "system",
                    "assistant": "assistant",
                }
                return mapping.get(value, "user")

        if type(messages[0]) is not dict:
            formatted_messages = [
                (LlamaMessage.model_validate(message.model_dump()).model_dump())
                for message in messages
            ]
        else:
            formatted_messages = messages

        if self.pydantic_model is not None:
            if self.pydantic_model.__name__ == "TextualSituationalAwareness":
                approximate_message_length = count_tokens(
                    formatted_messages[1]["content"]
                )
                if approximate_message_length > 4000:
                    formatted_messages[1]["content"] = formatted_messages[1]["content"][
                        :4000
                    ]  # truncate messages for situational analysis classification

        if self.pydantic_model is not None:
            response = await client.chat.completions.create(
                messages=formatted_messages,
                model=self.model_name,
                stream=False,
                temperature=0.1,
                # max_completion_tokens=4096,
                top_p=0.1,
                repetition_penalty=1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.pydantic_model.__name__,
                        "schema": self.pydantic_model.model_json_schema(),
                    },
                },
            )

            model = self.pydantic_model.model_validate_json(
                response.completion_message.content.text
            )
            formatted_messages_content_str = json.dumps(formatted_messages)
            token_usage = await calculate_token_usage_description_model(
                model_structured_output_response=model,
                input_str=formatted_messages_content_str,
            )

            result = (
                model,
                ResponseMetadata(model_name=self.model_name, token_usage=token_usage),
            )
            return result

        else:
            response = await client.chat.completions.create(
                messages=formatted_messages,
                model=self.model_name,
                stream=False,
                temperature=0.1,
                max_completion_tokens=16000,
                top_p=0.1,
                repetition_penalty=1,
            )
            # return AIMessage(content=response.completion_message.content.text)
            result = (
                AIMessage(content=response.completion_message.content.text),
                ResponseMetadata(
                    model_name=self.model_name,
                    token_usage=TokenUsage(
                        prompt_tokens=response.metrics.num_prompt_tokens,
                        total_tokens=response.metrics.num_total_tokens,
                        completion_tokens=response.metrics.num_completion_tokens,
                    ),
                ),
            )
            return result
