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
from contextvars import ContextVar
from typing import Any, List, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tokenizer import count_tokens

# Runnable tag applied to every structured-output model (``response_format`` set). The
# streaming layer (``_stream_deep_agent`` in graph.py) uses it to positively exclude these
# internal JSON-producing calls from the user-facing ``assistant_token`` stream — otherwise
# their raw structured output leaks into the chat (e.g. interleaved fact-correction JSON).
STRUCTURED_OUTPUT_STREAM_TAG = "structured_output_no_user_stream"

# NVIDIA Integrate serves Llama 3.2 90B Vision at 32,768 input tokens even
# though the base model is 128k. Observed 2026-09-14: HTTP 400
# "maximum context length is 32768 tokens. However, you requested 44576
# tokens in the messages".
_NVIDIA_HOSTED_INPUT_TOKEN_LIMITS = {
    "meta/llama-3.2-90b-vision-instruct": 32768,
}


def hosted_inference_input_token_limit(context: GlobalContext | None = None) -> int:
    """The input-token ceiling the inference endpoint will accept.

    ``MODEL_TOKEN_LIMIT`` is the operator's configured window (OpenAI-scale
    defaults are hundreds of thousands). NVIDIA's hosted 90B NIM is smaller
    than that default, and a request over the NIM ceiling is a 400 rather
    than a truncated prompt.
    """
    context = context or GlobalContext()
    configured = int(context.model_token_limit or 0) or 400000
    provider = (context.model_provider or "").strip().upper()
    model_name = (context.model or "").strip().lower()
    if provider != "NVIDIA":
        return configured
    for catalog_name, token_limit in _NVIDIA_HOSTED_INPUT_TOKEN_LIMITS.items():
        if model_name == catalog_name or model_name.startswith(catalog_name):
            return min(configured, token_limit)
    return configured


# What the model vendors say when the operator's own account is out of credit.
# None of these are the reader's doing. The stream layer reports them as 503
# only after a NVIDIA NIM retry in development has also failed.
VENDOR_CREDIT_EXHAUSTED_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "billing_hard_limit_reached",
    "insufficient credits",
    "insufficient_credits",
    "out of credits",
)

# A refused vendor key is not an empty credit balance. The stream and speech
# routes report these as ``vendor_key_refused``, never as the out-of-funds overlay.
VENDOR_KEY_REFUSED_MARKERS = (
    "invalid_api_key",
    "incorrect_api_key",
    "incorrect api key provided",
    "missing_api_key",
    "authentication_error",
)
VENDOR_KEY_REFUSED_CODE = "vendor_key_refused"

NVIDIA_NIM_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"

# The text model that actually answered this turn (primary or NVIDIA NIM
# fallback). Read by ``text_inference_record`` when the reply is stamped.
_active_text_inference: ContextVar[dict[str, Any] | None] = ContextVar(
    "active_text_inference", default=None
)


def development_mode_enabled(context: GlobalContext | None = None) -> bool:
    """Whether this process is running with ``DEV=TRUE``."""
    context = context or GlobalContext()
    return str(getattr(context, "dev", None) or "").strip().upper() == "TRUE"


def vendor_credit_is_exhausted(error: BaseException) -> bool:
    """Whether ``error`` is a vendor refusing the call for lack of credit."""
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(marker in error_text for marker in VENDOR_CREDIT_EXHAUSTED_MARKERS)


def vendor_key_is_refused(error: BaseException) -> bool:
    """Whether ``error`` is a vendor refusing the operator's API key.

    A refused key is not a spent reader allotment and not an empty vendor
    credit balance. Check ``vendor_credit_is_exhausted`` first when both
    could apply.
    """
    if vendor_credit_is_exhausted(error):
        return False
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(marker in error_text for marker in VENDOR_KEY_REFUSED_MARKERS)


def record_text_inference(
    *,
    model_provider: str | None,
    model: str | None,
    used_nvidia_nim_credit_fallback: bool = False,
) -> None:
    """Remember the text model that is answering this turn."""
    _active_text_inference.set(
        {
            "text_model": model,
            "text_model_provider": model_provider,
            "text_model_credit_fallback": used_nvidia_nim_credit_fallback,
        }
    )


def text_inference_record(context: GlobalContext | None = None) -> dict[str, Any]:
    """The text model to stamp on a reply, including a DEV credit fallback."""
    context = context or GlobalContext()
    recorded = _active_text_inference.get() or {}
    record: dict[str, Any] = {
        "text_model": recorded.get("text_model") or context.model,
        "text_model_provider": recorded.get("text_model_provider")
        or context.model_provider,
    }
    if recorded.get("text_model_credit_fallback"):
        record["text_model_credit_fallback"] = True
    return record


def resolve_nvidia_nim_settings(context: GlobalContext | None = None) -> dict[str, str]:
    """The NVIDIA NIM endpoint used when the primary text provider is out of credit.

    ``NVIDIA_NIM_*`` wins when set. When the key is unset and the configured
    ``LLM_PROVIDER_BASE_URL`` (or ``IMAGE_MODEL_BASE_URL``) is already NVIDIA,
    that key is reused so a DEV stack that only has ``LLM_PROVIDER_API_KEY``
    still has a fallback after the operator points ``MODEL_PROVIDER`` at OpenAI.
    """
    context = context or GlobalContext()
    base_url = (
        (getattr(context, "nvidia_nim_base_url", None) or "").strip()
        or NVIDIA_NIM_DEFAULT_BASE_URL
    )
    model_name = (
        (getattr(context, "nvidia_nim_model", None) or "").strip()
        or NVIDIA_NIM_DEFAULT_MODEL
    )
    api_key = (getattr(context, "nvidia_nim_api_key", None) or "").strip()
    if not api_key:
        llm_url = (context.llm_provider_base_url or "").strip().lower()
        image_url = (context.image_model_base_url or "").strip().lower()
        if "nvidia.com" in llm_url:
            api_key = (context.llm_provider_api_key or "").strip()
            if not (getattr(context, "nvidia_nim_model", None) or "").strip():
                model_name = (context.model or "").strip() or model_name
        elif "nvidia.com" in image_url:
            api_key = (context.image_model_api_key or "").strip()
    return {
        "model_provider": "NVIDIA",
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
    }


def primary_text_inference_is_nvidia_nim(context: GlobalContext | None = None) -> bool:
    """Whether the configured text model is already NVIDIA NIM."""
    context = context or GlobalContext()
    provider = (context.model_provider or "").strip().upper()
    base_url = (context.llm_provider_base_url or "").strip().lower()
    return provider == "NVIDIA" or "nvidia.com" in base_url


def should_retry_inference_on_nvidia_nim(
    error: BaseException, context: GlobalContext | None = None
) -> bool:
    """Whether a failed primary call should be retried on NVIDIA NIM.

    Development only. A stack that is already on NVIDIA NIM has nowhere else
    to go. A missing NIM key is logged rather than retried.
    """
    context = context or GlobalContext()
    if not development_mode_enabled(context):
        return False
    if not vendor_credit_is_exhausted(error):
        return False
    if primary_text_inference_is_nvidia_nim(context):
        return False
    if not resolve_nvidia_nim_settings(context)["api_key"]:
        logger.warning(
            "Primary text model is out of credit and NVIDIA NIM fallback "
            "has no API key (set NVIDIA_NIM_API_KEY)"
        )
        return False
    return True


def _build_nvidia_nim_chat_model(
    context: GlobalContext, *, stream_usage: bool = False
):
    """An unbound ChatOpenAI pointed at NVIDIA NIM."""
    from langchain_openai import ChatOpenAI

    settings = resolve_nvidia_nim_settings(context)
    return ChatOpenAI(
        model=settings["model"],
        base_url=settings["base_url"],
        **openai_sampling_parameters(settings["model"]),
        api_key=settings["api_key"],
        stream_usage=stream_usage,
    )


def attach_nvidia_nim_credit_fallback(
    model: Any,
    context: GlobalContext | None = None,
    *,
    record_as_text_inference: bool = True,
):
    """Wrap ``model`` so a DEV credit refusal retries on NVIDIA NIM."""
    context = context or GlobalContext()
    if record_as_text_inference:
        record_text_inference(
            model_provider=context.model_provider,
            model=context.model,
            used_nvidia_nim_credit_fallback=False,
        )
    if not development_mode_enabled(context):
        return model
    if primary_text_inference_is_nvidia_nim(context):
        return model
    if not resolve_nvidia_nim_settings(context)["api_key"]:
        return model
    return NvidiaNimCreditFallbackChatModel(
        primary=model,
        fallback=_build_nvidia_nim_chat_model(context),
        context=context,
    )


class NvidiaNimCreditFallbackChatModel(BaseChatModel):
    """Try the configured text model; on vendor credit exhaustion in DEV, use NVIDIA NIM.

    ``create_deep_agent`` requires a ``BaseChatModel``. After ``bind_tools``
    the inner clients become runnables; public ``invoke`` / ``astream``
    delegate to those runnables and retry the fallback when the primary
    refuses for lack of credit.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _primary: Any = PrivateAttr()
    _fallback: Any = PrivateAttr()
    _context: GlobalContext = PrivateAttr()

    def __init__(self, primary: Any, fallback: Any, context: GlobalContext, **kwargs):
        """Hold the primary client and the NVIDIA NIM fallback client."""
        super().__init__(**kwargs)
        self._primary = primary
        self._fallback = fallback
        self._context = context

    @property
    def _llm_type(self) -> str:
        return "nvidia_nim_credit_fallback"

    @property
    def model_name(self) -> str | None:
        """The catalog id of the text model that is answering this turn."""
        recorded = _active_text_inference.get() or {}
        return recorded.get("text_model") or getattr(self._primary, "model_name", None)

    def bind_tools(self, tools, **kwargs):
        """Bind the same tools on the primary client and the NVIDIA NIM fallback."""
        return NvidiaNimCreditFallbackChatModel(
            primary=self._primary.bind_tools(tools, **kwargs),
            fallback=self._fallback.bind_tools(tools, **kwargs),
            context=self._context,
        )

    def with_structured_output(self, schema, **kwargs):
        """Ask both clients for the same structured-output schema."""
        return NvidiaNimCreditFallbackChatModel(
            primary=self._primary.with_structured_output(schema, **kwargs),
            fallback=self._fallback.with_structured_output(schema, **kwargs),
            context=self._context,
        )

    def with_config(self, *args, **kwargs):
        """Copy runnable config onto both clients."""
        primary = (
            self._primary.with_config(*args, **kwargs)
            if hasattr(self._primary, "with_config")
            else self._primary
        )
        fallback = (
            self._fallback.with_config(*args, **kwargs)
            if hasattr(self._fallback, "with_config")
            else self._fallback
        )
        return NvidiaNimCreditFallbackChatModel(
            primary=primary, fallback=fallback, context=self._context
        )

    def _remember_primary(self) -> None:
        record_text_inference(
            model_provider=self._context.model_provider,
            model=self._context.model,
            used_nvidia_nim_credit_fallback=False,
        )

    def _remember_fallback(self) -> None:
        settings = resolve_nvidia_nim_settings(self._context)
        record_text_inference(
            model_provider=settings["model_provider"],
            model=settings["model"],
            used_nvidia_nim_credit_fallback=True,
        )
        logger.warning(
            "Primary text model %s is out of credit; retrying on NVIDIA NIM %s",
            self._context.model,
            settings["model"],
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            result = self._delegate_generate(
                self._primary, messages, stop, run_manager, **kwargs
            )
            self._remember_primary()
            return result
        except Exception as error:
            if not should_retry_inference_on_nvidia_nim(error, self._context):
                raise
            self._remember_fallback()
            return self._delegate_generate(
                self._fallback, messages, stop, run_manager, **kwargs
            )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            result = await self._adelegate_generate(
                self._primary, messages, stop, run_manager, **kwargs
            )
            self._remember_primary()
            return result
        except Exception as error:
            if not should_retry_inference_on_nvidia_nim(error, self._context):
                raise
            self._remember_fallback()
            return await self._adelegate_generate(
                self._fallback, messages, stop, run_manager, **kwargs
            )

    def _delegate_generate(self, model, messages, stop, run_manager, **kwargs):
        generate = getattr(model, "_generate", None)
        if callable(generate):
            return generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        message = model.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _adelegate_generate(self, model, messages, stop, run_manager, **kwargs):
        generate = getattr(model, "_agenerate", None)
        if callable(generate):
            return await generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        message = await model.ainvoke(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def invoke(self, input, config=None, **kwargs):
        """Invoke the primary client; on credit exhaustion, invoke NVIDIA NIM."""
        try:
            result = self._primary.invoke(input, config=config, **kwargs)
            self._remember_primary()
            return result
        except Exception as error:
            if not should_retry_inference_on_nvidia_nim(error, self._context):
                raise
            self._remember_fallback()
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        """Await the primary client; on credit exhaustion, await NVIDIA NIM."""
        try:
            result = await self._primary.ainvoke(input, config=config, **kwargs)
            self._remember_primary()
            return result
        except Exception as error:
            if not should_retry_inference_on_nvidia_nim(error, self._context):
                raise
            self._remember_fallback()
            return await self._fallback.ainvoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        """Stream the primary client; on a first-chunk credit refusal, stream NVIDIA NIM."""
        yielded = False
        try:
            for chunk in self._primary.stream(input, config=config, **kwargs):
                yielded = True
                yield chunk
            self._remember_primary()
        except Exception as error:
            if yielded or not should_retry_inference_on_nvidia_nim(
                error, self._context
            ):
                raise
            self._remember_fallback()
            yield from self._fallback.stream(input, config=config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        """Stream the primary client; on a first-chunk credit refusal, stream NVIDIA NIM."""
        yielded = False
        try:
            async for chunk in self._primary.astream(input, config=config, **kwargs):
                yielded = True
                yield chunk
            self._remember_primary()
        except Exception as error:
            if yielded or not should_retry_inference_on_nvidia_nim(
                error, self._context
            ):
                raise
            self._remember_fallback()
            async for chunk in self._fallback.astream(input, config=config, **kwargs):
                yield chunk


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
        model = attach_nvidia_nim_credit_fallback(
            model, context, record_as_text_inference=False
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
        from langchain_openai import ChatOpenAI

        # NVIDIA's hosted catalog is OpenAI-compatible at LLM_PROVIDER_BASE_URL.
        # ChatNVIDIA's local table still routes llama-3.2-11b-vision-instruct to
        # a VLM GR URL with supports_tools=False, so avatar tool binding never
        # hits integrate.api.nvidia.com. ChatOpenAI uses the same knobs as META.
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

    return attach_nvidia_nim_credit_fallback(model, context)


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

        openai_model = ChatOpenAI(
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
        return attach_nvidia_nim_credit_fallback(openai_model, context)

    if model_provider == "TOGETHER":
        from langchain_together import ChatTogether

        return attach_nvidia_nim_credit_fallback(
            ChatTogether(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                top_p=0.1,
                api_key=api_key,
            ),
            context,
        )

    if model_provider == "NVIDIA":
        from langchain_openai import ChatOpenAI

        nvidia_model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            **openai_sampling_parameters(model_name),
            api_key=api_key,
            stream_usage=False,
        )
        return attach_nvidia_nim_credit_fallback(nvidia_model, context)

    msg = f"Unsupported MODEL_PROVIDER for unbound chat model: {model_provider!r}"
    raise ValueError(msg)


def init_image_description_model():
    from langchain_openai import ChatOpenAI

    context = GlobalContext()
    model_name = context.image_model
    base_url = context.image_model_base_url
    api_key = context.image_model_api_key
    dev = context.dev

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
